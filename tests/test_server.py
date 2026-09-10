import time

import numpy as np
from starlette.testclient import TestClient

from asr_test import config
from asr_test.storage import SessionStore
from asr_test.utils import float32_to_pcm16
from fakes import FakeLlm, FakeStt, FakeTts, FakeVad
from server import create_app, _default_llm_models


def _make_client(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    fake_llm = FakeLlm(reply="hi there")
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: fake_llm,
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=SessionStore(":memory:"),
    )
    return TestClient(app), fake_llm


def test_ws_endpoint_sends_ready_event_then_audio_reply(monkeypatch):
    client, _ = _make_client(monkeypatch)
    with client.websocket_connect("/ws") as ws:
        ready = ws.receive_json()
        assert ready["event"] == "ready"

        frame = np.zeros(config.FRAME, dtype=np.float32)
        for _ in range(3):
            ws.send_bytes(float32_to_pcm16(frame))

        reply = ws.receive_bytes()
        assert len(reply) > 0


def test_two_concurrent_sessions_have_independent_history(monkeypatch):
    client, fake_llm = _make_client(monkeypatch)
    frame = np.zeros(config.FRAME, dtype=np.float32)

    with client.websocket_connect("/ws") as ws_a, client.websocket_connect("/ws") as ws_b:
        ws_a.receive_json()
        ws_b.receive_json()
        for ws in (ws_a, ws_b):
            for _ in range(3):
                ws.send_bytes(float32_to_pcm16(frame))
            ws.receive_bytes()

    assert len(fake_llm.calls) == 2
    assert fake_llm.calls[0] == fake_llm.calls[1] == [{"role": "user", "content": "hello"}]


def test_options_endpoint_lists_tts_voices_and_llm_models(monkeypatch):
    monkeypatch.setattr("server._default_llm_models", lambda base_url, fallback: ["model-a", "model-b"])
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm(),
        default_tts_engine="kokoro",
        session_store=SessionStore(":memory:"),
    )
    client = TestClient(app)

    resp = client.get("/options")

    assert resp.status_code == 200
    body = resp.json()
    assert body["tts"] == {"kokoro": ["voice-a", "voice-b"]}
    assert body["llm_models"] == ["model-a", "model-b"]
    assert body["defaults"]["tts_engine"] == "kokoro"


def test_default_llm_models_falls_back_when_lm_studio_unreachable(monkeypatch):
    class _BoomSession:
        def get(self, *a, **kw):
            raise ConnectionError("no server")

    monkeypatch.setattr("server.requests", _BoomSession())

    assert _default_llm_models("http://localhost:1234/v1", "fallback-model") == ["fallback-model"]


def test_create_app_eagerly_warms_default_tts_and_llm():
    before_tts = FakeTts.instances_created
    llm_factory_calls = []

    def llm_factory(model):
        llm_factory_calls.append(model)
        return FakeLlm()

    create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=llm_factory,
        default_tts_engine="kokoro",
        default_llm_model="default-model",
        session_store=SessionStore(":memory:"),
    )

    assert FakeTts.instances_created - before_tts == 1
    assert llm_factory_calls == ["default-model"]


def test_tts_engine_cache_reuses_instance_for_same_voice(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    fake_llm = FakeLlm(reply="hi there")
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: fake_llm,
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=SessionStore(":memory:"),
    )
    # Snapshot *after* create_app() — it now eagerly warms the default
    # (engine, voice="") combo at startup, which is a different cache key
    # from voice="voice-a" below and shouldn't count toward this delta.
    before = FakeTts.instances_created
    client = TestClient(app)

    with client.websocket_connect("/ws?voice=voice-a") as ws1:
        ws1.receive_json()
    with client.websocket_connect("/ws?voice=voice-a") as ws2:
        ws2.receive_json()

    assert FakeTts.instances_created - before == 1


def test_ws_query_params_select_voice_and_llm_model(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm(),
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=SessionStore(":memory:"),
    )
    client = TestClient(app)

    with client.websocket_connect("/ws?tts=kokoro&voice=voice-b&llm_model=custom-model") as ws:
        ready = ws.receive_json()

    assert ready["llm_model"] == "custom-model"
    assert "voice-b" in FakeTts.created_voices


def test_ws_vad_query_params_are_passed_to_vad_factory(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    captured = {}

    def spy_vad_factory(**kwargs):
        captured.update(kwargs)
        return FakeVad(start_at=1, end_at=3)

    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm(),
        vad_factory=spy_vad_factory,
        default_tts_engine="kokoro",
        session_store=SessionStore(":memory:"),
    )
    client = TestClient(app)

    with client.websocket_connect(
        "/ws?vad_threshold=0.3&vad_min_silence_ms=800&vad_speech_pad_ms=100"
    ) as ws:
        ws.receive_json()

    assert captured == {"threshold": 0.3, "min_silence_ms": 800, "speech_pad_ms": 100}


def test_ws_rejects_out_of_range_vad_threshold(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm(),
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=SessionStore(":memory:"),
    )
    client = TestClient(app)

    with client.websocket_connect("/ws?vad_threshold=1.5") as ws:
        msg = ws.receive_json()
        assert msg["event"] == "error"
        assert "vad_threshold" in msg["message"]


def test_ws_rejects_non_numeric_vad_param(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm(),
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=SessionStore(":memory:"),
    )
    client = TestClient(app)

    with client.websocket_connect("/ws?vad_threshold=not-a-number") as ws:
        msg = ws.receive_json()
        assert msg["event"] == "error"
        assert "vad_" in msg["message"]


def test_ws_rejects_negative_vad_min_silence_ms(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm(),
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=SessionStore(":memory:"),
    )
    client = TestClient(app)

    with client.websocket_connect("/ws?vad_min_silence_ms=-100") as ws:
        msg = ws.receive_json()
        assert msg["event"] == "error"


def test_ws_text_mode_accepts_json_text_and_replies_with_bot_text(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm("hi there"),
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=SessionStore(":memory:"),
    )
    client = TestClient(app)

    with client.websocket_connect("/ws?mode=text") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "hello there"})

        user_event = ws.receive_json()
        bot_event = ws.receive_json()

    assert user_event == {"event": "user_text", "text": "hello there"}
    assert bot_event["event"] == "bot_text"
    assert bot_event["text"] == "hi there "


def test_ws_text_mode_never_sends_binary_audio(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm("hi there"),
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=SessionStore(":memory:"),
    )
    client = TestClient(app)

    with client.websocket_connect("/ws?mode=text") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "hello there"})
        ws.receive_json()  # user_text
        ws.receive_json()  # bot_text
        # Starlette's TestClient raises if a message arrives in an
        # unexpected shape, so a stray binary frame here would surface
        # as a failure on one of the two receive_json() calls above.


def test_ws_rejects_invalid_mode(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm(),
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=SessionStore(":memory:"),
    )
    client = TestClient(app)

    with client.websocket_connect("/ws?mode=carrier-pigeon") as ws:
        msg = ws.receive_json()
        assert msg["event"] == "error"
        assert "mode" in msg["message"]


def test_ws_text_mode_never_calls_vad_factory(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    calls = []

    def spy_vad_factory(**kwargs):
        calls.append(kwargs)
        return FakeVad(start_at=1, end_at=3)

    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm("hi there"),
        vad_factory=spy_vad_factory,
        default_tts_engine="kokoro",
        session_store=SessionStore(":memory:"),
    )
    client = TestClient(app)

    with client.websocket_connect("/ws?mode=text") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "hello there"})
        ws.receive_json()  # user_text
        ws.receive_json()  # bot_text

    assert calls == []


def test_ws_text_mode_never_constructs_a_real_tts_engine(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    before = FakeTts.instances_created
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm("hi there"),
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=SessionStore(":memory:"),
    )
    # create_app() eagerly warms the default (engine, voice="") combo at
    # startup regardless of mode -- snapshot after that, not before.
    before = FakeTts.instances_created
    client = TestClient(app)

    with client.websocket_connect("/ws?mode=text") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "hello there"})
        ws.receive_json()  # user_text
        ws.receive_json()  # bot_text

    assert FakeTts.instances_created == before


def test_ready_event_includes_session_id(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    store = SessionStore(":memory:")
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm(),
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=store,
    )
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ready = ws.receive_json()

    assert "session_id" in ready and ready["session_id"]


def test_completed_turn_is_persisted_to_session_store(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    store = SessionStore(":memory:")
    fake_usage = {"input_tokens": 5, "output_tokens": 3}
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm("hi there", fake_usage=fake_usage),
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=store,
    )
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ready = ws.receive_json()
        session_id = ready["session_id"]
        frame = np.zeros(config.FRAME, dtype=np.float32)
        for _ in range(3):
            ws.send_bytes(float32_to_pcm16(frame))
        ws.receive_bytes()  # first TTS chunk — the bot_text event (and its
        # store write) can race slightly behind this, since TTS synthesis
        # runs concurrently with respond()'s own final bookkeeping; poll
        # below rather than assuming this call already implies the write.

        deadline = time.time() + 2.0
        result = store.get_session(session_id)
        while (not result or len(result["turns"]) < 2) and time.time() < deadline:
            time.sleep(0.02)
            result = store.get_session(session_id)

    assert result is not None
    assert result["turns"] == [
        {"role": "user", "text": "hello", "usage": None},
        {"role": "assistant", "text": "hi there ", "usage": fake_usage},
    ]


def test_get_sessions_lists_created_sessions(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    store = SessionStore(":memory:")
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm(),
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=store,
    )
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ready = ws.receive_json()

    resp = client.get("/sessions")

    assert resp.status_code == 200
    ids = [s["id"] for s in resp.json()]
    assert ready["session_id"] in ids


def test_get_session_by_id_returns_detail(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    store = SessionStore(":memory:")
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm(),
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=store,
    )
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ready = ws.receive_json()

    resp = client.get(f"/sessions/{ready['session_id']}")

    assert resp.status_code == 200
    assert resp.json()["session"]["id"] == ready["session_id"]


def test_get_session_by_unknown_id_returns_404(monkeypatch):
    store = SessionStore(":memory:")
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm(),
        default_tts_engine="kokoro",
        session_store=store,
    )
    client = TestClient(app)

    resp = client.get("/sessions/does-not-exist")

    assert resp.status_code == 404

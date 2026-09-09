import numpy as np
from starlette.testclient import TestClient

from asr_test import config
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
        vad_factory=lambda: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
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
        vad_factory=lambda: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
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
        vad_factory=lambda: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
    )
    client = TestClient(app)

    with client.websocket_connect("/ws?tts=kokoro&voice=voice-b&llm_model=custom-model") as ws:
        ready = ws.receive_json()

    assert ready["llm_model"] == "custom-model"
    assert "voice-b" in FakeTts.created_voices


def test_ws_echo_mode_query_param_is_applied_and_echoed_back(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm(),
        vad_factory=lambda: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
    )
    client = TestClient(app)

    with client.websocket_connect("/ws?echo_mode=headphones") as ws:
        ready = ws.receive_json()

    assert ready["echo_mode"] == "headphones"


def test_ws_rejects_invalid_echo_mode(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm(),
        vad_factory=lambda: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
    )
    client = TestClient(app)

    with client.websocket_connect("/ws?echo_mode=bogus") as ws:
        msg = ws.receive_json()
        assert msg["event"] == "error"
        assert "echo_mode" in msg["message"]

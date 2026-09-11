import json
import time

import numpy as np
from fakes import FakeLlm, FakeStt, FakeTts, FakeVad
from starlette.testclient import TestClient

from asr_test import config
from asr_test.cli.server import _default_llm_models, create_app
from asr_test.storage import SessionStore
from asr_test.utils import float32_to_pcm16


def _receive_json_skipping_audio(ws, max_messages=200):
    """WebSocketAudioSink streams a continuous background silence
    block the instant it's constructed (see its own docstring/tests) --
    in voice mode a real reply's binary frames can legitimately arrive
    interleaved with any JSON event, at any time. Tests that care about
    a specific JSON event (not just "some audio eventually arrives")
    need to skip past those, not assume the very next message is it."""
    for _ in range(max_messages):
        message = ws.receive()
        if "text" in message:
            return json.loads(message["text"])
    raise AssertionError("no JSON message arrived within max_messages")


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
    monkeypatch.setattr("asr_test.cli.server._default_llm_models", lambda base_url, fallback: ["model-a", "model-b"])
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


def test_options_endpoint_includes_provider_and_tools_for_connections_diagram(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr("asr_test.cli.server._default_llm_models", lambda base_url, fallback: ["model-a"])
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
    assert body["provider"]["name"] == "local"
    assert body["provider"]["model"]
    tool_names = {t["name"] for t in body["tools"]}
    assert tool_names == {"get_current_time", "web_search"}
    web_search = next(t for t in body["tools"] if t["name"] == "web_search")
    assert web_search["enabled"] is False  # no TAVILY_API_KEY set


def test_default_llm_models_falls_back_when_lm_studio_unreachable(monkeypatch):
    class _BoomSession:
        def get(self, *a, **kw):
            raise ConnectionError("no server")

    monkeypatch.setattr("asr_test.cli.server.requests", _BoomSession())

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


def test_ws_rejects_wake_word_mode_without_trigger_word(monkeypatch):
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

    with client.websocket_connect("/ws?voice_input_mode=wake_word") as ws:
        msg = ws.receive_json()
        assert msg["event"] == "error"
        assert "trigger_word" in msg["message"]


def test_ws_rejects_invalid_voice_input_mode(monkeypatch):
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

    with client.websocket_connect("/ws?voice_input_mode=telepathy") as ws:
        msg = ws.receive_json()
        assert msg["event"] == "error"
        assert "voice_input_mode" in msg["message"]


def test_ws_wake_word_mode_with_trigger_word_is_accepted(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    app = create_app(
        stt=FakeStt("computer, hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm("hi there"),
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=SessionStore(":memory:"),
    )
    client = TestClient(app)

    with client.websocket_connect("/ws?voice_input_mode=wake_word&trigger_word=computer") as ws:
        ready = ws.receive_json()
        assert ready["event"] == "ready"

        frame = np.zeros(config.FRAME, dtype=np.float32)
        for _ in range(3):
            ws.send_bytes(float32_to_pcm16(frame))
        reply = ws.receive_bytes()

    assert len(reply) > 0


def test_ws_push_to_talk_mode_never_calls_vad_factory(monkeypatch):
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

    with client.websocket_connect("/ws?voice_input_mode=push_to_talk") as ws:
        ws.receive_json()  # ready
        ws.send_json({"event": "ptt_start"})
        frame = np.zeros(config.FRAME, dtype=np.float32)
        ws.send_bytes(float32_to_pcm16(frame))
        ws.send_json({"event": "ptt_stop"})
        _receive_json_skipping_audio(ws)  # user_text

    assert calls == []


def test_ws_push_to_talk_produces_a_response_on_ptt_stop(monkeypatch):
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

    with client.websocket_connect("/ws?voice_input_mode=push_to_talk") as ws:
        ws.receive_json()  # ready
        ws.send_json({"event": "ptt_start"})
        frame = np.zeros(config.FRAME, dtype=np.float32)
        for _ in range(3):
            ws.send_bytes(float32_to_pcm16(frame))
        ws.send_json({"event": "ptt_stop"})

        user_event = _receive_json_skipping_audio(ws)

    assert user_event == {"event": "user_text", "text": "hello"}


def test_ws_push_to_talk_ignores_trigger_word(monkeypatch):
    # trigger_word only applies in wake_word mode -- push_to_talk should
    # respond to plain speech even if a trigger_word happens to be sent.
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

    with client.websocket_connect("/ws?voice_input_mode=push_to_talk&trigger_word=computer") as ws:
        ws.receive_json()  # ready
        ws.send_json({"event": "ptt_start"})
        frame = np.zeros(config.FRAME, dtype=np.float32)
        for _ in range(3):
            ws.send_bytes(float32_to_pcm16(frame))
        ws.send_json({"event": "ptt_stop"})

        user_event = _receive_json_skipping_audio(ws)

    assert user_event == {"event": "user_text", "text": "hello"}


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


def test_session_keys_endpoint_returns_a_key_token(monkeypatch):
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

    resp = client.post("/session-keys", json={"openai_api_key": "sk-test"})

    assert resp.status_code == 200
    assert resp.json()["key_token"]


def test_ws_key_token_is_applied_via_llm_env_factory(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    captured = {}

    def spy_llm_env_factory(model, env):
        captured["model"] = model
        captured["env"] = dict(env)
        return FakeLlm("hi there")

    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm(),
        llm_env_factory=spy_llm_env_factory,
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=SessionStore(":memory:"),
    )
    client = TestClient(app)

    token = client.post("/session-keys", json={"openai_api_key": "sk-override"}).json()["key_token"]
    with client.websocket_connect(f"/ws?key_token={token}") as ws:
        ws.receive_json()  # ready

    assert captured["model"] == "lfm2.5-230m"
    assert captured["env"]["OPENAI_API_KEY"] == "sk-override"


def test_session_keys_maps_azure_and_tavily_fields(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    captured = {}

    def spy_llm_env_factory(model, env):
        captured["env"] = dict(env)
        return FakeLlm("hi there")

    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm(),
        llm_env_factory=spy_llm_env_factory,
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=SessionStore(":memory:"),
    )
    client = TestClient(app)

    token = client.post("/session-keys", json={
        "azure_api_key": "az-key",
        "azure_endpoint": "https://example.openai.azure.com/",
        "azure_deployment": "my-deployment",
        "tavily_api_key": "tvly-key",
    }).json()["key_token"]
    with client.websocket_connect(f"/ws?key_token={token}") as ws:
        ws.receive_json()  # ready

    assert captured["env"]["AZURE_OPENAI_API_KEY"] == "az-key"
    assert captured["env"]["AZURE_OPENAI_ENDPOINT"] == "https://example.openai.azure.com/"
    assert captured["env"]["AZURE_OPENAI_DEPLOYMENT"] == "my-deployment"
    assert captured["env"]["TAVILY_API_KEY"] == "tvly-key"


def test_ws_key_token_is_single_use(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm(),
        llm_env_factory=lambda model, env: FakeLlm("hi there"),
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=SessionStore(":memory:"),
    )
    client = TestClient(app)

    token = client.post("/session-keys", json={"openai_api_key": "sk-test"}).json()["key_token"]
    with client.websocket_connect(f"/ws?key_token={token}") as ws:
        ws.receive_json()  # ready

    with client.websocket_connect(f"/ws?key_token={token}") as ws:
        msg = ws.receive_json()
        assert msg["event"] == "error"
        assert "key_token" in msg["message"]


def test_ws_unknown_key_token_gets_error(monkeypatch):
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

    with client.websocket_connect("/ws?key_token=nonexistent-token") as ws:
        msg = ws.receive_json()
        assert msg["event"] == "error"
        assert "key_token" in msg["message"]


def test_ws_without_key_token_never_calls_llm_env_factory(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    calls = []
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm("hi there"),
        llm_env_factory=lambda model, env: calls.append((model, env)) or FakeLlm(),
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=SessionStore(":memory:"),
    )
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready

    assert calls == []


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
    assert ready["resumed"] is False


def test_ws_resume_session_id_seeds_conversation_and_keeps_writing_to_it(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    store = SessionStore(":memory:")
    store.create_session("s1", mode="text", tts_engine=None, llm_model="lfm2.5-230m")
    store.add_turn("s1", "user", "what's your name")
    store.add_turn("s1", "assistant", "Assistant.")
    fake_llm = FakeLlm("hi there")
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: fake_llm,
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=store,
    )
    client = TestClient(app)

    with client.websocket_connect("/ws?mode=text&resume_session_id=s1") as ws:
        ready = ws.receive_json()
        assert ready["session_id"] == "s1"
        assert ready["resumed"] is True

        ws.send_json({"text": "hello again"})
        ws.receive_json()  # user_text
        ws.receive_json()  # bot_text

    # the LLM call should have seen the prior turns as context
    assert fake_llm.calls[0][0] == {"role": "user", "content": "what's your name"}
    assert fake_llm.calls[0][1] == {"role": "assistant", "content": "Assistant."}
    assert fake_llm.calls[0][-1] == {"role": "user", "content": "hello again"}

    # and the new turns were appended to the SAME stored session, not a new one
    result = store.get_session("s1")
    assert [t["text"] for t in result["turns"]] == [
        "what's your name", "Assistant.", "hello again", "hi there ",
    ]


def test_ws_resuming_with_a_different_mode_updates_the_stored_mode(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    store = SessionStore(":memory:")
    store.create_session("s1", mode="voice", tts_engine="kokoro", llm_model="lfm2.5-230m")
    store.add_turn("s1", "user", "hello")
    store.add_turn("s1", "assistant", "hi there")
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm("hi there"),
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=store,
    )
    client = TestClient(app)

    with client.websocket_connect("/ws?mode=text&resume_session_id=s1") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "switched to text"})
        ws.receive_json()  # user_text
        ws.receive_json()  # bot_text

    assert store.get_session("s1")["session"]["mode"] == "text"
    assert store.list_sessions()[0]["mode"] == "text"


def test_ws_resuming_with_the_same_mode_leaves_stored_mode_unchanged(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    store = SessionStore(":memory:")
    store.create_session("s1", mode="text", tts_engine=None, llm_model="lfm2.5-230m")
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm("hi there"),
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=store,
    )
    client = TestClient(app)

    with client.websocket_connect("/ws?mode=text&resume_session_id=s1") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "hi"})
        ws.receive_json()  # user_text
        ws.receive_json()  # bot_text

    assert store.get_session("s1")["session"]["mode"] == "text"


def test_ws_resume_unknown_session_id_gets_error(monkeypatch):
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

    with client.websocket_connect("/ws?resume_session_id=does-not-exist") as ws:
        msg = ws.receive_json()
        assert msg["event"] == "error"
        assert "does-not-exist" in msg["message"]


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


def test_delete_session_removes_it(monkeypatch):
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

    resp = client.delete(f"/sessions/{ready['session_id']}")

    assert resp.status_code == 200
    assert store.get_session(ready["session_id"]) is None


def test_delete_session_unknown_id_returns_404():
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm(),
        default_tts_engine="kokoro",
        session_store=SessionStore(":memory:"),
    )
    client = TestClient(app)

    resp = client.delete("/sessions/does-not-exist")

    assert resp.status_code == 404


def test_first_exchange_generates_a_session_title(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    store = SessionStore(":memory:")
    fake_llm = FakeLlm("hi there", fake_title="Weekend trip planning")
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: fake_llm,
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=store,
    )
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ready = ws.receive_json()
        frame = np.zeros(config.FRAME, dtype=np.float32)
        for _ in range(3):
            ws.send_bytes(float32_to_pcm16(frame))
        ws.receive_bytes()

        deadline = time.time() + 2.0
        result = store.get_session(ready["session_id"])
        while result["session"]["title"] is None and time.time() < deadline:
            time.sleep(0.02)
            result = store.get_session(ready["session_id"])

    assert result["session"]["title"] == "Weekend trip planning"
    assert fake_llm.title_calls == [("hello", "hi there ")]


def test_title_generation_is_not_retriggered_on_later_turns(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    store = SessionStore(":memory:")
    fake_llm = FakeLlm("hi there", fake_title="Weekend trip planning")
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: fake_llm,
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=store,
    )
    client = TestClient(app)

    # Text mode, not voice: two turns sent as raw PCM frames back-to-back
    # raced against WebSocketAudioSink's real playback/barge-in timing --
    # the second turn's frames could land inside the first reply's
    # barge-in grace window and get silently dropped as a false barge-in
    # rather than queued as a new segment (found via a real intermittent
    # full-suite failure: "assert 1 == 2"). Text mode's on_text_message()
    # has no such timing dependency, so two turns here are deterministic.
    with client.websocket_connect("/ws?mode=text") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "hi"})
        ws.receive_json()  # user_text
        ws.receive_json()  # bot_text
        ws.send_json({"text": "another question"})
        ws.receive_json()  # user_text
        ws.receive_json()  # bot_text

    assert len(fake_llm.calls) == 2
    assert len(fake_llm.title_calls) == 1


def test_resuming_a_session_does_not_regenerate_its_title(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    store = SessionStore(":memory:")
    store.create_session("s1", mode="text", tts_engine=None, llm_model="lfm2.5-230m")
    store.add_turn("s1", "user", "hi")
    store.add_turn("s1", "assistant", "hello")
    store.set_title("s1", "Original title")
    fake_llm = FakeLlm("hi there", fake_title="Should not be used")
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: fake_llm,
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=store,
    )
    client = TestClient(app)

    with client.websocket_connect("/ws?mode=text&resume_session_id=s1") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "one more thing"})
        ws.receive_json()  # user_text
        ws.receive_json()  # bot_text

    assert fake_llm.title_calls == []
    assert store.get_session("s1")["session"]["title"] == "Original title"


def test_index_page_is_served():
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm(),
        default_tts_engine="kokoro",
        session_store=SessionStore(":memory:"),
    )
    client = TestClient(app)

    resp = client.get("/")

    assert resp.status_code == 200
    assert "Voice Agent" in resp.text

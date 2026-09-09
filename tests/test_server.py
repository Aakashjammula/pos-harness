import numpy as np
from starlette.testclient import TestClient

from asr_test import config
from asr_test.utils import float32_to_pcm16
from fakes import FakeLlm, FakeStt, FakeTts, FakeVad
from server import create_app


def _make_client(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    fake_llm = FakeLlm(reply="hi there")
    app = create_app(
        stt=FakeStt("hello"),
        tts=FakeTts(),
        llm=fake_llm,
        vad_factory=lambda: FakeVad(start_at=1, end_at=3),
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

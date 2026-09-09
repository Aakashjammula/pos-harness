import time

import numpy as np

from asr_test import config
from asr_test.agent import Agent
from fakes import FakeAudioSink, FakeLlm, FakeStt, FakeTts, FakeVad


def _build_agent(**overrides):
    kwargs = dict(
        vad=FakeVad(start_at=1, end_at=3),
        stt=FakeStt("hello"),
        tts=FakeTts(),
        llm=FakeLlm("hi there"),
        audio_sink=FakeAudioSink(),
    )
    kwargs.update(overrides)
    return Agent(**kwargs)


def test_feed_audio_queues_a_frame():
    agent = _build_agent()
    frame = np.zeros(config.FRAME, dtype=np.float32)
    agent.feed_audio(frame)
    assert agent.mic_q.qsize() == 1


def test_start_returns_three_running_threads_and_shutdown_stops_them():
    agent = _build_agent()
    threads = agent.start()
    assert len(threads) == 3
    assert all(t.is_alive() for t in threads)
    agent.shutdown(threads)
    for t in threads:
        t.join(timeout=1.0)
        assert not t.is_alive()


def test_end_to_end_frame_to_response(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    fake_stt = FakeStt("hello")
    fake_llm = FakeLlm("hi there")
    fake_sink = FakeAudioSink()
    agent = _build_agent(stt=fake_stt, llm=fake_llm, audio_sink=fake_sink)

    threads = agent.start()
    try:
        frame = np.zeros(config.FRAME, dtype=np.float32)
        for _ in range(3):
            agent.feed_audio(frame)

        deadline = time.time() + 2.0
        while not fake_sink.pushed and time.time() < deadline:
            time.sleep(0.02)

        assert len(fake_stt.calls) == 1
        assert len(fake_llm.calls) == 1
        assert fake_sink.pushed  # TTS output reached the sink
    finally:
        agent.shutdown(threads)


def test_agent_tracks_conversation_history_across_turns(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    fake_llm = FakeLlm(reply="hi there")
    agent = _build_agent(
        vad=FakeVad(start_at=1, end_at=3), stt=FakeStt("hello"), llm=fake_llm,
    )
    threads = agent.start()
    try:
        frame = np.zeros(config.FRAME, dtype=np.float32)

        for _ in range(3):
            agent.feed_audio(frame)
        deadline = time.time() + 2.0
        while len(fake_llm.calls) < 1 and time.time() < deadline:
            time.sleep(0.02)
        assert fake_llm.calls[0] == [{"role": "user", "content": "hello"}]

        for _ in range(3):
            agent.feed_audio(frame)
        deadline = time.time() + 2.0
        while len(fake_llm.calls) < 2 and time.time() < deadline:
            time.sleep(0.02)
        assert fake_llm.calls[1] == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there "},
            {"role": "user", "content": "hello"},
        ]
    finally:
        agent.shutdown(threads)

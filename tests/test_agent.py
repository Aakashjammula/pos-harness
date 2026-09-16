import time

import numpy as np
from fakes import FakeAudioSink, FakeLlm, FakeStt, FakeTts, FakeVad

from pos import config
from pos.agent import Agent


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


def test_on_event_fires_user_text_and_bot_text(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    events: list[tuple[str, dict]] = []
    agent = _build_agent(
        vad=FakeVad(start_at=1, end_at=3), stt=FakeStt("hello"), llm=FakeLlm("hi there"),
        on_event=lambda name, data: events.append((name, data)),
    )
    threads = agent.start()
    try:
        frame = np.zeros(config.FRAME, dtype=np.float32)
        for _ in range(3):
            agent.feed_audio(frame)
        deadline = time.time() + 2.0
        while len(events) < 2 and time.time() < deadline:
            time.sleep(0.02)

        assert ("user_text", {"text": "hello"}) in events
        bot_events = [data for name, data in events if name == "bot_text"]
        assert len(bot_events) == 1
        assert bot_events[0]["text"] == "hi there "
        assert bot_events[0]["latency"]["ttft"] >= 0
        assert bot_events[0]["latency"]["total"] >= bot_events[0]["latency"]["ttft"]
    finally:
        agent.shutdown(threads)


def test_on_event_fires_interrupted_on_barge_in():
    events: list[tuple[str, dict]] = []
    agent = _build_agent(on_event=lambda name, data: events.append((name, data)))
    agent.interrupt()
    assert ("interrupted", {}) in events


def test_bot_text_event_includes_usage_when_llm_reports_it(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    events: list[tuple[str, dict]] = []
    fake_usage = {
        "provider": "local", "model": "lfm2.5-230m",
        "input_tokens": 12, "output_tokens": 4, "total_tokens": 16,
        "cost_usd": 0.0, "tool_calls": [], "context_window": 131072,
    }
    agent = _build_agent(
        vad=FakeVad(start_at=1, end_at=3), stt=FakeStt("hello"),
        llm=FakeLlm("hi there", fake_usage=fake_usage),
        on_event=lambda name, data: events.append((name, data)),
    )
    threads = agent.start()
    try:
        frame = np.zeros(config.FRAME, dtype=np.float32)
        for _ in range(3):
            agent.feed_audio(frame)
        deadline = time.time() + 2.0
        while not any(name == "bot_text" for name, _ in events) and time.time() < deadline:
            time.sleep(0.02)

        bot_events = [data for name, data in events if name == "bot_text"]
        assert len(bot_events) == 1
        assert bot_events[0]["text"] == "hi there "
        assert bot_events[0]["usage"] == fake_usage
        assert bot_events[0]["latency"]["ttft"] >= 0
    finally:
        agent.shutdown(threads)


def test_text_only_agent_never_pushes_to_tts_queue(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    fake_llm = FakeLlm(reply="hi there")
    agent = _build_agent(llm=fake_llm, text_only=True)

    agent.on_text_message("hello")

    assert agent.tts_q.qsize() == 0
    assert agent.conversation == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there "},
    ]


def test_on_text_message_fires_user_text_then_bot_text():
    events: list[tuple[str, dict]] = []
    agent = _build_agent(
        llm=FakeLlm("hi there"), text_only=True,
        on_event=lambda name, data: events.append((name, data)),
    )

    agent.on_text_message("hello")

    assert events[0] == ("user_text", {"text": "hello"})
    assert len(events) == 2
    bot_name, bot_data = events[1]
    assert bot_name == "bot_text"
    assert bot_data["text"] == "hi there "
    assert bot_data["latency"]["ttft"] >= 0


def test_voice_mode_agent_still_pushes_to_tts_queue(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    fake_llm = FakeLlm(reply="hi there")
    agent = _build_agent(llm=fake_llm)  # text_only defaults to False

    agent.respond("hello", agent.new_turn(), stt_t=0.0)

    assert agent.tts_q.qsize() == 1


def test_conversation_can_be_seeded_to_resume_a_prior_session():
    seed = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    agent = _build_agent(conversation=seed)

    assert agent.conversation == seed


def test_seeded_conversation_is_copied_not_aliased():
    seed = [{"role": "user", "content": "hello"}]
    agent = _build_agent(conversation=seed)

    agent.conversation.append({"role": "assistant", "content": "hi"})

    assert seed == [{"role": "user", "content": "hello"}]


def test_no_conversation_seed_defaults_to_empty():
    agent = _build_agent()

    assert agent.conversation == []


def test_feed_audio_drops_frames_while_muted():
    agent = _build_agent()
    agent.muted.set()
    agent.feed_audio(np.zeros(config.FRAME, dtype=np.float32))
    assert agent.mic_q.qsize() == 0

    agent.muted.clear()
    agent.feed_audio(np.zeros(config.FRAME, dtype=np.float32))
    assert agent.mic_q.qsize() == 1


# --- push-to-talk ---


class _PlayingAudioSink(FakeAudioSink):
    """FakeAudioSink always reports playing=False (see its own
    docstring) -- ptt_start()'s barge-in-equivalent branch needs a sink
    that can report True, so this test-local subclass adds a settable
    flag instead of changing the shared fake's contract."""

    def __init__(self):
        super().__init__()
        self.playing_flag = False

    @property
    def playing(self) -> bool:
        return self.playing_flag


def test_feed_ptt_frame_buffers_without_touching_mic_q():
    agent = _build_agent()
    frame = np.ones(config.FRAME, dtype=np.float32)

    agent.feed_ptt_frame(frame)

    assert agent.mic_q.qsize() == 0  # push-to-talk bypasses VAD's mic_q entirely
    assert agent._ptt_buf == [frame]


def test_feed_ptt_frame_drops_frames_while_muted():
    agent = _build_agent()
    agent.muted.set()

    agent.feed_ptt_frame(np.ones(config.FRAME, dtype=np.float32))

    assert agent._ptt_buf == []


def test_ptt_start_resets_the_buffer():
    agent = _build_agent()
    agent.feed_ptt_frame(np.ones(config.FRAME, dtype=np.float32))

    agent.ptt_start()

    assert agent._ptt_buf == []


def test_ptt_start_interrupts_playback_in_progress():
    sink = _PlayingAudioSink()
    sink.playing_flag = True
    events: list[tuple[str, dict]] = []
    agent = _build_agent(audio_sink=sink, on_event=lambda name, data: events.append((name, data)))

    agent.ptt_start()

    assert ("interrupted", {}) in events


def test_ptt_start_does_not_interrupt_when_nothing_is_playing():
    events: list[tuple[str, dict]] = []
    agent = _build_agent(on_event=lambda name, data: events.append((name, data)))  # FakeAudioSink.playing == False

    agent.ptt_start()

    assert events == []


def test_ptt_stop_pushes_the_concatenated_segment_to_seg_q():
    agent = _build_agent()
    frame = np.ones(config.FRAME, dtype=np.float32)
    agent.feed_ptt_frame(frame)
    agent.feed_ptt_frame(frame)

    agent.ptt_stop()

    assert agent.seg_q.qsize() == 1
    turn, seg = agent.seg_q.get_nowait()
    assert turn == agent.current_turn()
    assert seg.shape == (config.FRAME * 2,)
    assert agent._ptt_buf == []


def test_ptt_stop_with_no_buffered_audio_is_a_noop():
    agent = _build_agent()

    agent.ptt_stop()

    assert agent.seg_q.qsize() == 0


def test_ptt_end_to_end_produces_a_response(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    fake_stt = FakeStt("hello")
    fake_llm = FakeLlm("hi there")
    fake_sink = FakeAudioSink()
    agent = _build_agent(stt=fake_stt, llm=fake_llm, audio_sink=fake_sink)

    threads = agent.start()
    try:
        agent.ptt_start()
        frame = np.zeros(config.FRAME, dtype=np.float32)
        for _ in range(3):
            agent.feed_ptt_frame(frame)
        agent.ptt_stop()

        deadline = time.time() + 2.0
        while not fake_sink.pushed and time.time() < deadline:
            time.sleep(0.02)

        assert len(fake_stt.calls) == 1
        assert len(fake_llm.calls) == 1
        assert fake_sink.pushed
    finally:
        agent.shutdown(threads)

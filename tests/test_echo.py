import sys
import types

import numpy as np
import pytest

from asr_test.audio.echo import EchoControl


def test_headphones_mode_always_passes_mic_through():
    echo = EchoControl("headphones", mic_rate=16000, frame_size=512)
    assert echo.barge_in is True
    frame = np.ones(512, dtype=np.float32)
    assert np.array_equal(echo.process(frame, is_playing=True), frame)
    assert np.array_equal(echo.process(frame, is_playing=False), frame)


def test_duck_mode_drops_frames_only_while_playing():
    echo = EchoControl("duck", mic_rate=16000, frame_size=512)
    assert echo.barge_in is False
    frame = np.ones(512, dtype=np.float32)
    assert echo.process(frame, is_playing=True) is None
    assert np.array_equal(echo.process(frame, is_playing=False), frame)


def test_aec_mode_falls_back_to_duck_when_voiceclean_unavailable(monkeypatch):
    # voiceclean isn't installed in this environment, so the real `import
    # voiceclean` inside EchoControl.__init__ raises naturally — this is
    # exactly the fallback path a user without the optional package hits.
    monkeypatch.delitem(sys.modules, "voiceclean", raising=False)
    echo = EchoControl("aec", mic_rate=16000, frame_size=512)
    assert echo.mode == "duck"
    assert echo.barge_in is False


class _FakeAecResult:
    def __init__(self, audio: bytes):
        self.audio = audio
        self.is_speech = True
        self.speech_prob = 0.9


class _FakeVoiceClean:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.fed_references: list[bytes] = []
        self.processed: list[bytes] = []

    def feed_reference(self, pcm_bytes: bytes) -> None:
        self.fed_references.append(pcm_bytes)

    def process(self, pcm_bytes: bytes) -> _FakeAecResult:
        self.processed.append(pcm_bytes)
        return _FakeAecResult(audio=pcm_bytes)  # identity: echo mic back unchanged


@pytest.fixture
def fake_voiceclean(monkeypatch):
    module = types.ModuleType("voiceclean")
    module.VoiceClean = _FakeVoiceClean
    monkeypatch.setitem(sys.modules, "voiceclean", module)
    return module


def test_aec_mode_calls_the_real_voiceclean_api_shape(fake_voiceclean):
    """Regression test for the bug this fixes: EchoControl used to call a
    voiceclean.AEC(...)/process(mic, ref) shape that doesn't exist in the
    real package (confirmed against voiceclean's own docs), so "aec" mode
    silently fell back to "duck" even with voiceclean installed. The real
    API is VoiceClean(sample_rate=...), feed_reference(pcm_bytes) for the
    bot's own audio, then process(mic_pcm_bytes) -> result.audio."""
    echo = EchoControl("aec", mic_rate=16000, frame_size=512)
    assert echo.mode == "aec"
    assert echo.barge_in is True
    assert isinstance(echo.aec, _FakeVoiceClean)

    ref_audio = np.full(512, 0.1, dtype=np.float32)
    echo.note_playback(ref_audio, source_rate=16000)
    assert len(echo.aec.fed_references) == 1

    mic_frame = np.full(512, 0.2, dtype=np.float32)
    out = echo.process(mic_frame, is_playing=True)

    assert len(echo.aec.processed) == 1
    assert out is not None
    assert np.allclose(out, mic_frame, atol=1e-3)


def test_aec_mode_skips_processing_when_not_playing(fake_voiceclean):
    echo = EchoControl("aec", mic_rate=16000, frame_size=512)
    mic_frame = np.full(512, 0.2, dtype=np.float32)

    out = echo.process(mic_frame, is_playing=False)

    assert np.array_equal(out, mic_frame)
    assert echo.aec.processed == []

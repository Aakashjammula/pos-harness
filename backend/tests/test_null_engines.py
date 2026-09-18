import numpy as np

from pos.null_engines import NullTts, NullVad


def test_null_tts_returns_empty_audio():
    tts = NullTts()

    result = tts("anything")

    assert isinstance(result, np.ndarray)
    assert result.size == 0


def test_null_tts_has_a_sample_rate():
    tts = NullTts()

    assert isinstance(tts.sample_rate, int)
    assert tts.sample_rate > 0


def test_null_vad_never_fires_a_speech_boundary():
    vad = NullVad()
    frame = np.zeros(512, dtype=np.float32)

    for _ in range(10):
        assert vad(frame) is None

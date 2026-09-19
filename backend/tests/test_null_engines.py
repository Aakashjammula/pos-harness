import numpy as np

from pos.null_engines import NullVad


def test_null_vad_never_fires_a_speech_boundary():
    vad = NullVad()
    frame = np.zeros(512, dtype=np.float32)

    for _ in range(10):
        assert vad(frame) is None

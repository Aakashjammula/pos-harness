import numpy as np
import pytest

from asr_test.utils import float32_to_pcm16, pcm16_to_float32


def test_pcm16_to_float32_roundtrip_silence():
    data = (np.zeros(10, dtype=np.int16)).tobytes()
    out = pcm16_to_float32(data)
    assert out.dtype == np.float32
    assert np.allclose(out, 0.0)


def test_pcm16_to_float32_full_scale():
    data = np.array([32767, -32768], dtype=np.int16).tobytes()
    out = pcm16_to_float32(data)
    assert out[0] == pytest.approx(1.0, abs=1e-3)
    assert out[1] == pytest.approx(-1.0, abs=1e-3)


def test_float32_to_pcm16_full_scale():
    audio = np.array([1.0, -1.0, 0.0], dtype=np.float32)
    data = float32_to_pcm16(audio)
    values = np.frombuffer(data, dtype=np.int16)
    assert values[0] == 32767
    assert values[1] == -32767
    assert values[2] == 0


def test_roundtrip_is_close():
    original = np.array([0.5, -0.25, 0.1], dtype=np.float32)
    restored = pcm16_to_float32(float32_to_pcm16(original))
    assert np.allclose(original, restored, atol=1e-4)

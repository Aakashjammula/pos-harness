from __future__ import annotations

import numpy as np


def resample_linear(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or x.size == 0:
        return x
    n_out = int(round(x.size * dst / src))
    return np.interp(
        np.linspace(0.0, x.size - 1, n_out, dtype=np.float64),
        np.arange(x.size, dtype=np.float64),
        x,
    ).astype(np.float32)


def pcm16_to_float32(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()

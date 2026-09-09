from __future__ import annotations

import numpy as np
from silero_vad import VADIterator, load_silero_vad

from ..interfaces.vad import VadBase


class SileroVad(VadBase):
    def __init__(
        self,
        sample_rate: int = 16000,
        min_silence_ms: int = 900,
        speech_pad_ms: int = 300,
    ):
        self._model = load_silero_vad(onnx=True)
        self._iterator = VADIterator(
            self._model,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )

    def __call__(self, frame: np.ndarray) -> dict | None:
        return self._iterator(frame, return_seconds=False)

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class TtsBase(ABC):
    """Text-to-speech: one text chunk in, mono float32 PCM out.

    `sample_rate` is the rate of the audio this engine produces — callers
    must read it from the instance rather than assume a fixed value, since
    it varies by engine.
    """

    sample_rate: int

    @abstractmethod
    def __call__(self, text: str) -> np.ndarray: ...

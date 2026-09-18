from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SttBase(ABC):
    """Speech-to-text: one finished utterance in, transcript text out."""

    @abstractmethod
    def __call__(self, audio: np.ndarray) -> str: ...

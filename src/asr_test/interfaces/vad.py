from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class VadBase(ABC):
    """Streaming speech-boundary detector.

    Implementations are stateful: call once per fixed-size audio frame,
    in order. Returns an event dict containing "start" or "end" when a
    speech boundary is crossed for that frame, else None.
    """

    @abstractmethod
    def __call__(self, frame: np.ndarray) -> dict | None: ...

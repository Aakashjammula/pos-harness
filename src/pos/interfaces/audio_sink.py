from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AudioSinkBase(ABC):
    """Receives synthesized audio for playback/delivery.

    Implementations own their own real-time pacing: push() queues
    audio, and the implementation is responsible for draining it at the
    correct rate (a real output device does this via its own callback;
    a network sink needs an explicit timer thread to reproduce the same
    behavior — see WebSocketAudioSink).
    """

    underruns: int

    @abstractmethod
    def push(self, audio: np.ndarray) -> None: ...

    @abstractmethod
    def flush(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def playing(self) -> bool: ...

    @property
    @abstractmethod
    def elapsed_ms(self) -> float: ...

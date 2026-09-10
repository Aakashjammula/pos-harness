from __future__ import annotations

import numpy as np

from ..interfaces.audio_sink import AudioSinkBase


class NullAudioSink(AudioSinkBase):
    """No-op sink for text-mode sessions, which never synthesize audio
    at all (Agent(text_only=True) never calls push()) — used instead of
    WebSocketAudioSink so its background pacing thread never starts and
    streams silence over a connection that has no audio to deliver."""

    underruns = 0

    def push(self, audio: np.ndarray) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass

    @property
    def playing(self) -> bool:
        return False

    @property
    def elapsed_ms(self) -> float:
        return 0.0

from __future__ import annotations

import threading
from collections.abc import Iterator

import numpy as np

from asr_test.interfaces import AudioSinkBase, LlmBase, SttBase, TtsBase, VadBase


class FakeVad(VadBase):
    """Fires 'start' on the Nth call and 'end' on the Mth call, else
    None — deterministic speech-boundary events for tests, no real
    audio analysis."""

    def __init__(self, start_at: int = 1, end_at: int = 3):
        self._n = 0
        self._start_at = start_at
        self._end_at = end_at

    def __call__(self, frame: np.ndarray) -> dict | None:
        self._n += 1
        if self._n == self._start_at:
            return {"start": 0}
        if self._n == self._end_at:
            return {"end": 0}
        return None


class FakeStt(SttBase):
    def __init__(self, text: str = "hello"):
        self.text = text
        self.calls: list[np.ndarray] = []

    def __call__(self, audio: np.ndarray) -> str:
        self.calls.append(audio)
        return self.text


class FakeLlm(LlmBase):
    def __init__(self, reply: str = "hi there"):
        self.reply = reply
        self.last_ttft: float | None = 0.01
        self.last_total: float | None = 0.02
        self.calls: list[str] = []

    def stream(self, user_text: str, cancel: threading.Event) -> Iterator[str]:
        self.calls.append(user_text)
        for word in self.reply.split():
            yield word + " "


class FakeTts(TtsBase):
    sample_rate = 16000

    def __call__(self, text: str) -> np.ndarray:
        return np.zeros(160, dtype=np.float32)


class FakeAudioSink(AudioSinkBase):
    def __init__(self):
        self.pushed: list[np.ndarray] = []
        self.underruns = 0
        self._playing = False

    def push(self, audio: np.ndarray) -> None:
        self.pushed.append(audio)
        self._playing = True

    def flush(self) -> None:
        self.pushed.clear()
        self._playing = False

    def close(self) -> None:
        pass

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def elapsed_ms(self) -> float:
        return 0.0

from __future__ import annotations

import threading
from collections.abc import Iterator

import numpy as np

from asr_test.interfaces import AudioSinkBase, LlmBase, SttBase, TtsBase, VadBase


class FakeVad(VadBase):
    """Fires 'start' at position start_at and 'end' at position end_at
    within a repeating cycle of length end_at — deterministic
    speech-boundary events for tests (no real audio analysis), cyclic
    so a test can drive multiple turns through the same Agent."""

    def __init__(self, start_at: int = 1, end_at: int = 3):
        self._n = 0
        self._start_at = start_at
        self._end_at = end_at

    def __call__(self, frame: np.ndarray) -> dict | None:
        self._n += 1
        pos = (self._n - 1) % self._end_at + 1
        if pos == self._start_at:
            return {"start": 0}
        if pos == self._end_at:
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
        self.calls: list[list[dict]] = []

    def stream(self, messages: list[dict], cancel: threading.Event) -> Iterator[str]:
        self.calls.append(messages)
        for word in self.reply.split():
            yield word + " "


class FakeTts(TtsBase):
    sample_rate = 16000
    instances_created = 0  # class-level counter — tests assert engine-cache reuse against this
    created_voices: list[str | None] = []

    def __init__(self, voice: str | None = None):
        self.voice = voice
        FakeTts.instances_created += 1
        FakeTts.created_voices.append(voice)

    def __call__(self, text: str) -> np.ndarray:
        return np.zeros(160, dtype=np.float32)

    @classmethod
    def list_voices(cls) -> list[str]:
        return ["voice-a", "voice-b"]


class FakeAudioSink(AudioSinkBase):
    """playing always reports False: unlike the real sinks, this fake
    has no timer/callback draining its buffer, so a naive "push() sets
    playing=True" would never clear again — that stuck Agent.vad_thread
    in its barge-in branch forever after one push, silently swallowing
    every later frame (found via a real two-turn test failure). No
    current test needs playing=True, so it's left unsimulated rather
    than building a real virtual clock for it."""

    def __init__(self):
        self.pushed: list[np.ndarray] = []
        self.underruns = 0

    def push(self, audio: np.ndarray) -> None:
        self.pushed.append(audio)

    def flush(self) -> None:
        self.pushed.clear()

    def close(self) -> None:
        pass

    @property
    def playing(self) -> bool:
        return False

    @property
    def elapsed_ms(self) -> float:
        return 0.0

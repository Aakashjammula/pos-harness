"""No-op stand-ins for text-mode sessions (see server.py's ws_endpoint),
which never feed audio in or synthesize audio out at all. Deliberately
NOT placed inside the tts/ or vad/ packages -- those packages' own
__init__.py eagerly import the real (heavy) engines, and importing
anything from a package always runs its __init__.py first. A module
that needs "a null engine, nothing else" would otherwise drag in
KokoroTts/SupertonicTts/SileroVad's own imports just by asking for
these -- exactly the cost text mode exists to avoid, and exactly what
this project's existing lazy-import convention (see create_app()'s own
deferred `from asr_test.tts import ...`) is careful to prevent for
tests that inject fake engines instead."""

from __future__ import annotations

import numpy as np

from .interfaces.tts import TtsBase
from .interfaces.vad import VadBase


class NullTts(TtsBase):
    sample_rate = 16000

    def __call__(self, text: str) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)


class NullVad(VadBase):
    def __call__(self, frame: np.ndarray) -> dict | None:
        return None

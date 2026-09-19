"""A no-op VAD for push-to-talk sessions (see server.py's ws_endpoint), where the
client says explicitly when speech starts and stops, so voice-activity
detection is never consulted. Kept out of the vad/ package: that package's
__init__ eagerly imports the real (heavy) SileroVad, and importing anything from
a package runs its __init__ first."""

from __future__ import annotations

import numpy as np

from .interfaces.vad import VadBase


class NullVad(VadBase):
    def __call__(self, frame: np.ndarray) -> dict | None:
        return None

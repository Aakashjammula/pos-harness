from __future__ import annotations

import threading
from collections import deque

import numpy as np

from ..utils import resample_linear


class EchoControl:
    def __init__(self, mode: str, mic_rate: int, frame_size: int):
        self.mode = mode
        self.mic_rate = mic_rate
        self.frame_size = frame_size
        self.barge_in = mode in ("headphones", "aec")
        self._ref: deque[np.ndarray] = deque(maxlen=64)
        self._lock = threading.Lock()
        self.aec = None

        if mode == "aec":
            try:
                import voiceclean

                self.aec = voiceclean.AEC(sample_rate=mic_rate, frame_size=frame_size)
                print("  echo: SpeexDSP AEC active")
            except Exception as e:
                print(f"  echo: AEC unavailable ({e}) — using duck")
                self.mode = "duck"
                self.barge_in = False
        else:
            print(f"  echo: {mode}")

    def note_playback(self, audio: np.ndarray, source_rate: int):
        if self.mode != "aec":
            return
        with self._lock:
            self._ref.append(resample_linear(audio, source_rate, self.mic_rate))

    def process(self, mic: np.ndarray, is_playing: bool) -> np.ndarray | None:
        if self.mode == "duck":
            return None if is_playing else mic
        if self.mode == "aec" and self.aec is not None and is_playing:
            with self._lock:
                ref = self._ref.popleft() if self._ref else None
            if ref is not None:
                if ref.size < mic.size:
                    ref = np.concatenate(
                        [ref, np.zeros(mic.size - ref.size, dtype=np.float32)]
                    )
                try:
                    return np.asarray(
                        self.aec.process(mic, ref[: mic.size]), dtype=np.float32
                    )
                except Exception as e:
                    print(f"   [aec failed: {e}]")
        return mic

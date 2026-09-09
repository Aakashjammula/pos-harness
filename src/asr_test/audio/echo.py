from __future__ import annotations

import threading

import numpy as np

from ..utils import float32_to_pcm16, pcm16_to_float32, resample_linear


class EchoControl:
    def __init__(self, mode: str, mic_rate: int, frame_size: int):
        self.mode = mode
        self.mic_rate = mic_rate
        self.frame_size = frame_size
        self.barge_in = mode in ("headphones", "aec")
        self._lock = threading.Lock()
        self.aec = None

        if mode == "aec":
            try:
                import voiceclean

                # voiceclean's real API (confirmed against its own docs,
                # not the mismatched voiceclean.AEC(...)/process(mic, ref)
                # shape this used to call — that class doesn't exist in the
                # real package, so "aec" mode silently fell back to "duck"
                # unconditionally, even with voiceclean installed): a
                # VoiceClean instance is fed reference (bot) audio
                # separately via feed_reference(), then process(mic_bytes)
                # returns a result whose .audio is the cleaned PCM.
                self.aec = voiceclean.VoiceClean(sample_rate=mic_rate)
                print("  echo: voiceclean AEC active")
            except Exception as e:
                print(f"  echo: AEC unavailable ({e}) — using duck")
                self.mode = "duck"
                self.barge_in = False
        else:
            print(f"  echo: {mode}")

    def note_playback(self, audio: np.ndarray, source_rate: int):
        if self.mode != "aec" or self.aec is None:
            return
        resampled = resample_linear(audio, source_rate, self.mic_rate)
        with self._lock:
            self.aec.feed_reference(float32_to_pcm16(resampled))

    def process(self, mic: np.ndarray, is_playing: bool) -> np.ndarray | None:
        if self.mode == "duck":
            return None if is_playing else mic
        if self.mode == "aec" and self.aec is not None and is_playing:
            try:
                with self._lock:
                    result = self.aec.process(float32_to_pcm16(mic))
                return pcm16_to_float32(result.audio)
            except Exception as e:
                print(f"   [aec failed: {e}]")
        return mic

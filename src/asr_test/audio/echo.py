from __future__ import annotations

import threading

import numpy as np

from ..utils import float32_to_pcm16, pcm16_to_float32, resample_linear

# voiceclean's AEC is correlation-based echo *suppression* (attenuates a
# chunk once it correlates with the buffered reference above this
# threshold), not classic adaptive-filter cancellation — confirmed against
# the real installed package's voiceclean.aec.AEC signature, not just its
# docs. Its own docs recommend lowering the 0.15 default to ~0.10 for
# "noisy environments"/challenging echo (their own example: PSTN telephony
# calls where borderline echo was slipping through) — a speaker/non-headphone
# room is a closer match to that than voiceclean's quieter default use case,
# so this project ships that same lowered default rather than requiring
# users to discover it. Not independently verified with real audio here
# (still needs your own live test); raise back toward 0.15 if it ends up
# suppressing your actual voice, lower further (their docs mention
# 0.08-0.10) if residual echo still triggers VAD.
_AEC_CORRELATION_THRESHOLD = 0.10


class EchoControl:
    def __init__(self, mode: str, mic_rate: int, frame_size: int):
        self.mode = mode
        self.mic_rate = mic_rate
        self.frame_size = frame_size
        self.barge_in = mode in ("headphones", "aec")
        self._lock = threading.Lock()
        self.aec = None
        # voiceclean buffers internally on its own frame size and emits
        # variable-length (sometimes empty) chunks per process() call —
        # confirmed live: feeding 512-sample frames produced 0 or 640
        # sample outputs, never 512. Accumulated here and re-chunked to
        # exactly frame_size per call, same pattern as WebSocketAudioSink's
        # drain logic elsewhere in this codebase.
        self._aec_out_buf = np.zeros(0, dtype=np.float32)

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
                self.aec = voiceclean.VoiceClean(
                    sample_rate=mic_rate, correlation_threshold=_AEC_CORRELATION_THRESHOLD
                )
                print(f"  echo: voiceclean AEC active (correlation_threshold={_AEC_CORRELATION_THRESHOLD})")
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
        if self.mode == "aec" and self.aec is not None:
            if not is_playing:
                # Drop any cleaned audio buffered from the previous burst —
                # otherwise the next time the bot talks, the first frame(s)
                # returned would be stale audio from before, not this
                # moment's actual mic input.
                self._aec_out_buf = np.zeros(0, dtype=np.float32)
                return mic
            try:
                with self._lock:
                    result = self.aec.process(float32_to_pcm16(mic))
                self._aec_out_buf = np.concatenate(
                    [self._aec_out_buf, pcm16_to_float32(result.audio)]
                )
            except Exception as e:
                print(f"   [aec failed: {e}]")
                return mic
            if self._aec_out_buf.size < mic.size:
                return None  # voiceclean hasn't buffered enough cleaned audio yet this tick
            frame_out, self._aec_out_buf = (
                self._aec_out_buf[: mic.size],
                self._aec_out_buf[mic.size :],
            )
            return frame_out
        return mic

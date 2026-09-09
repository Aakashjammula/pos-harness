from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd

# Short enough to be inaudible as a ramp (~1-4ms depending on rate), long
# enough to avoid the click/pop a hard jump to/from zero produces. Underruns
# (TTS falling behind playback) aren't eliminated by this — see kokoro/HF's
# own reference client, which takes the same approach: don't try to hide a
# stall in time, just make its edges sound clean instead of a click.
_FADE_FRAMES = 64


class AudioOutput:
    def __init__(self, rate: int, blocksize: int = 1024, on_played=None):
        self.rate = rate
        self.on_played = on_played
        self._buf: deque[np.ndarray] = deque()
        self._lock = threading.Lock()
        self._playing = threading.Event()
        self._started_at: float | None = None
        self._starved = False  # previous callback ended in padding — fade the next real block in
        self.underruns = 0

        self.stream = sd.OutputStream(
            samplerate=rate,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            callback=self._callback,
        )
        self.stream.start()

    def _callback(self, outdata, frames, time_info, status):
        need = frames
        chunks: list[np.ndarray] = []

        with self._lock:
            while need > 0 and self._buf:
                head = self._buf[0]
                if head.size <= need:
                    chunks.append(head)
                    need -= head.size
                    self._buf.popleft()
                else:
                    chunks.append(head[:need])
                    self._buf[0] = head[need:]
                    need = 0
            empty = not self._buf

        block = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

        if block.size < frames:
            if block.size:
                fade_len = min(_FADE_FRAMES, block.size)
                block[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                if self._playing.is_set():
                    self.underruns += 1
            block = np.concatenate(
                [block, np.zeros(frames - block.size, dtype=np.float32)]
            )
            self._starved = True
        elif self._starved:
            fade_len = min(_FADE_FRAMES, block.size)
            block[:fade_len] *= np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            self._starved = False

        outdata[:, 0] = block

        if self.on_played is not None and chunks:
            self.on_played(np.concatenate(chunks))

        if empty and not chunks and self._playing.is_set():
            self._playing.clear()
            self._started_at = None

    def push(self, audio: np.ndarray):
        if audio.size == 0:
            return
        with self._lock:
            self._buf.append(audio.astype(np.float32))
        if not self._playing.is_set():
            self._playing.set()
            self._started_at = time.perf_counter()

    def flush(self):
        with self._lock:
            self._buf.clear()
        self._playing.clear()
        self._started_at = None

    @property
    def playing(self) -> bool:
        return self._playing.is_set()

    @property
    def elapsed_ms(self) -> float:
        if self._started_at is None:
            return 0.0
        return (time.perf_counter() - self._started_at) * 1000.0

    def close(self):
        self.flush()
        self.stream.stop()
        self.stream.close()

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque

import numpy as np

from ..interfaces.audio_sink import AudioSinkBase
from ..utils import float32_to_pcm16

_FADE_FRAMES = 64


class WebSocketAudioSink(AudioSinkBase):
    """Same push/flush/playing/elapsed_ms contract as LocalAudioSink,
    but paced by a background thread instead of a PortAudio callback —
    there's no real output device pulling frames at a fixed rate, so
    this drives that pacing itself and sends each drained block over
    the websocket."""

    def __init__(self, websocket, loop: asyncio.AbstractEventLoop, rate: int, blocksize: int = 1024):
        self.websocket = websocket
        self.loop = loop
        self.rate = rate
        self.blocksize = blocksize
        self._buf: deque[np.ndarray] = deque()
        self._lock = threading.Lock()
        self._playing_flag = threading.Event()
        self._started_at: float | None = None
        self._starved = False
        self.underruns = 0
        self._stop = threading.Event()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _drain_once(self) -> np.ndarray:
        """Pull exactly one block's worth of audio, applying the same
        fade-in/out and underrun accounting as LocalAudioSink's
        callback. Split out from _run() so tests can call it directly
        without waiting on real wall-clock pacing."""
        need = self.blocksize
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

        if block.size < self.blocksize:
            if block.size:
                fade_len = min(_FADE_FRAMES, block.size)
                block[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                if self._playing_flag.is_set():
                    self.underruns += 1
            block = np.concatenate([block, np.zeros(self.blocksize - block.size, dtype=np.float32)])
            self._starved = True
        elif self._starved:
            fade_len = min(_FADE_FRAMES, block.size)
            block[:fade_len] *= np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            self._starved = False

        if empty and not chunks and self._playing_flag.is_set():
            self._playing_flag.clear()
            self._started_at = None

        return block

    async def _safe_send(self, data: bytes) -> None:
        try:
            await self.websocket.send_bytes(data)
        except Exception:
            pass  # socket already closing/closed — nothing to deliver to

    def _schedule_send(self, data: bytes) -> bool:
        """Schedule one block for sending. Returns False (and does not
        raise) if the event loop is already closed/closing — e.g. the
        client disconnected and the server is tearing down this
        connection while this thread is mid-tick. run_coroutine_threadsafe()
        itself (not just the coroutine it schedules) raises synchronously
        in that case — _safe_send's own try/except only covers errors
        *after* successful scheduling, not this."""
        coro = self._safe_send(data)
        try:
            asyncio.run_coroutine_threadsafe(coro, self.loop)
            return True
        except RuntimeError:
            coro.close()  # never got scheduled — close it explicitly or asyncio warns "never awaited"
            return False

    def _run(self):
        interval = self.blocksize / self.rate
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            next_tick += interval
            block = self._drain_once()
            if not self._schedule_send(float32_to_pcm16(block)):
                break
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)

    def push(self, audio: np.ndarray) -> None:
        if audio.size == 0:
            return
        with self._lock:
            self._buf.append(audio.astype(np.float32))
        if not self._playing_flag.is_set():
            self._playing_flag.set()
            self._started_at = time.perf_counter()

    def flush(self) -> None:
        with self._lock:
            self._buf.clear()
        self._playing_flag.clear()
        self._started_at = None

    def close(self) -> None:
        self.flush()
        self._stop.set()
        self._thread.join(timeout=1.0)

    @property
    def playing(self) -> bool:
        return self._playing_flag.is_set()

    @property
    def elapsed_ms(self) -> float:
        if self._started_at is None:
            return 0.0
        return (time.perf_counter() - self._started_at) * 1000.0

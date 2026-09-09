# WebSocket Transport (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the VAD→STT→LLM→TTS pipeline transport-agnostic (no direct `sounddevice` ownership inside `Agent`), then add a FastAPI websocket server supporting multiple concurrent sessions, plus a browser HTML client and a Python CLI client — while local mode (`main.py`) keeps working unchanged.

**Architecture:** Split `Agent` into pipeline logic (unchanged) and a generic `feed_audio()` entry point + injectable `AudioSinkBase` for output, so the same `Agent` class runs identically whether driven by a real `sd.InputStream` (local mode) or a websocket receive loop (server mode). Heavy STT/TTS model instances are constructed once at server startup and shared read-only across all concurrent sessions (they hold no per-conversation state); only VAD (per-utterance iterator state) and conversation history are per-session.

**Tech Stack:** Python 3.14, FastAPI + uvicorn (server), `websockets` (CLI client), pytest (new — this repo has no test suite today), numpy, existing onnx-asr/onnxruntime/sounddevice/openai stack.

**Spec:** `docs/superpowers/specs/2026-09-09-transport-provider-harness-design.md` (Phase 1 section)

## Global Constraints

- Python >=3.14 (per `pyproject.toml`).
- Wire format is raw PCM16 mono, matching `config.MIC_RATE` (16000) for
  client→server audio and the TTS engine's own `sample_rate` for
  server→client audio — no codec/compression.
- No backwards-compatibility aliases: `AudioOutput` is renamed to
  `LocalAudioSink` everywhere, not re-exported under the old name.
- No hardcoded credentials — this phase adds no cloud providers, but any
  future ones must read keys from environment variables only (org data
  policy); nothing in this plan violates that today.
- Server-mode sessions default `config.ECHO_MODE` to `"duck"`, not
  `"headphones"` — the server can't verify a remote client has real
  headphone isolation.
- Every pure-logic change gets a pytest unit/integration test using
  fakes (no real audio hardware, no real network calls, no real ONNX
  model loads in tests). Hardware-touching code (local mic/speaker,
  browser JS) is verified by a manual smoke test instead, matching this
  repo's existing testing philosophy (README documents manual
  verification for hardware paths already).

---

### Task 1: PCM16/float32 conversion helpers + pytest setup

This repo has zero tests today. This task adds pytest as a dev
dependency and writes the first tests against pure functions with no
I/O — the smallest possible TDD loop to establish the pattern the rest
of this plan follows. These two functions are also the wire-format
conversion every later networked task needs (websocket sink, server,
CLI client all move PCM16 bytes across a socket and need float32
internally).

**Files:**
- Modify: `pyproject.toml` (add pytest dev dependency)
- Create: `conftest.py` (repo root, empty — see note below)
- Create: `tests/test_utils.py`
- Modify: `src/asr_test/utils.py`

**Interfaces:**
- Produces: `pcm16_to_float32(data: bytes) -> np.ndarray` (mono, dtype
  `float32`, range roughly [-1.0, 1.0]); `float32_to_pcm16(audio:
  np.ndarray) -> bytes` (dtype `int16` little-endian bytes). Every later
  task that crosses the websocket boundary (Tasks 4, 6, 7) imports these
  two functions from `asr_test.utils`.

- [x] **Step 1: Add pytest as a dev dependency**

Run: `uv add --dev pytest`

Expected: `pyproject.toml` gains a `[dependency-groups]` `dev = ["pytest>=..."]`
entry (or similar, depending on installed uv version's exact TOML
shape) and `uv.lock` updates.

- [x] **Step 2: Add a root `conftest.py` so top-level scripts are importable from tests**

```python
# conftest.py
# Empty on purpose. pytest inserts this file's directory (the repo
# root) into sys.path when it exists, which is what lets tests import
# top-level scripts like server.py (added in Task 6) with a plain
# `from server import ...` instead of packaging them.
```

- [x] **Step 3: Write the failing tests**

```python
# tests/test_utils.py
import numpy as np

from asr_test.utils import float32_to_pcm16, pcm16_to_float32


def test_pcm16_to_float32_roundtrip_silence():
    data = (np.zeros(10, dtype=np.int16)).tobytes()
    out = pcm16_to_float32(data)
    assert out.dtype == np.float32
    assert np.allclose(out, 0.0)


def test_pcm16_to_float32_full_scale():
    data = np.array([32767, -32768], dtype=np.int16).tobytes()
    out = pcm16_to_float32(data)
    assert out[0] == pytest.approx(1.0, abs=1e-3)
    assert out[1] == pytest.approx(-1.0, abs=1e-3)


def test_float32_to_pcm16_full_scale():
    audio = np.array([1.0, -1.0, 0.0], dtype=np.float32)
    data = float32_to_pcm16(audio)
    values = np.frombuffer(data, dtype=np.int16)
    assert values[0] == 32767
    assert values[1] == -32768
    assert values[2] == 0


def test_roundtrip_is_close():
    original = np.array([0.5, -0.25, 0.1], dtype=np.float32)
    restored = pcm16_to_float32(float32_to_pcm16(original))
    assert np.allclose(original, restored, atol=1e-4)
```

Add `import pytest` at the top of the file alongside the `numpy` import.

- [x] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_utils.py -v`
Expected: FAIL with `ImportError: cannot import name 'pcm16_to_float32'`
(the functions don't exist yet).

- [x] **Step 5: Implement the functions**

```python
# src/asr_test/utils.py — add alongside the existing resample_linear
def pcm16_to_float32(data: bytes) -> np.ndarray:
    return (np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0)


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()
```

- [x] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_utils.py -v`
Expected: 4 passed.

- [x] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock conftest.py tests/test_utils.py src/asr_test/utils.py
git commit -m "test: add pytest and PCM16/float32 conversion helpers"
```

---

### Task 2: `AudioSinkBase` interface + rename `AudioOutput` → `LocalAudioSink`

Establishes the injectable output interface every transport (local,
websocket) will implement. Pure rename + ABC conformance for the
existing local implementation — no behavior change, verified with tests
that stub out `sounddevice.OutputStream` so no real audio device is
touched.

**Files:**
- Create: `src/asr_test/interfaces/audio_sink.py`
- Modify: `src/asr_test/interfaces/__init__.py`
- Modify: `src/asr_test/audio/output.py` (rename class)
- Modify: `src/asr_test/agent.py` (update the one import/usage site)
- Test: `tests/test_local_audio_sink.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `AudioSinkBase` (ABC with `push(audio: np.ndarray) -> None`,
  `flush() -> None`, `close() -> None`, `playing: bool` property,
  `elapsed_ms: float` property, `underruns: int` attribute) and
  `LocalAudioSink(rate: int, blocksize: int = 1024, on_played=None)`
  implementing it — same constructor signature `AudioOutput` had. Task 3
  wires this into `Agent.__init__` as the default; Task 4 adds the
  second implementation.

- [x] **Step 1: Write the interface**

```python
# src/asr_test/interfaces/audio_sink.py
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AudioSinkBase(ABC):
    """Receives synthesized audio for playback/delivery.

    Implementations own their own real-time pacing: push() queues
    audio, and the implementation is responsible for draining it at the
    correct rate (a real output device does this via its own callback;
    a network sink needs an explicit timer thread to reproduce the same
    behavior — see WebSocketAudioSink).
    """

    underruns: int

    @abstractmethod
    def push(self, audio: np.ndarray) -> None: ...

    @abstractmethod
    def flush(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def playing(self) -> bool: ...

    @property
    @abstractmethod
    def elapsed_ms(self) -> float: ...
```

- [x] **Step 2: Export it from the interfaces package**

```python
# src/asr_test/interfaces/__init__.py
from .audio_sink import AudioSinkBase
from .llm import LlmBase
from .stt import SttBase
from .tts import TtsBase
from .vad import VadBase

__all__ = ["AudioSinkBase", "LlmBase", "SttBase", "TtsBase", "VadBase"]
```

- [x] **Step 3: Write the failing tests (stubbing `sounddevice.OutputStream`)**

```python
# tests/test_local_audio_sink.py
import numpy as np
import pytest

from asr_test.audio.output import LocalAudioSink
from asr_test.interfaces.audio_sink import AudioSinkBase


class _FakeStream:
    def __init__(self, **kwargs):
        self.callback = kwargs["callback"]
        self.blocksize = kwargs["blocksize"]
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


@pytest.fixture
def sink(monkeypatch):
    monkeypatch.setattr("asr_test.audio.output.sd.OutputStream", _FakeStream)
    s = LocalAudioSink(rate=16000, blocksize=256)
    yield s
    s.close()


def test_local_audio_sink_is_audio_sink_base(sink):
    assert isinstance(sink, AudioSinkBase)


def test_push_sets_playing(sink):
    assert not sink.playing
    sink.push(np.ones(256, dtype=np.float32))
    assert sink.playing


def test_callback_drains_pushed_audio(sink):
    sink.push(np.ones(256, dtype=np.float32))
    outdata = np.zeros((256, 1), dtype=np.float32)
    sink.stream.callback(outdata, 256, None, None)
    assert np.allclose(outdata[:, 0], 1.0)


def test_flush_clears_buffer_and_stops_playing(sink):
    sink.push(np.ones(256, dtype=np.float32))
    sink.flush()
    assert not sink.playing


def test_underrun_counted_when_buffer_empties_mid_playback(sink):
    sink.push(np.ones(100, dtype=np.float32))  # shorter than one callback block
    outdata = np.zeros((256, 1), dtype=np.float32)
    sink.stream.callback(outdata, 256, None, None)
    assert sink.underruns == 1
```

- [x] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_local_audio_sink.py -v`
Expected: FAIL — `ImportError: cannot import name 'LocalAudioSink'`
(class is still named `AudioOutput`).

- [x] **Step 5: Rename the class and make it implement the ABC**

```python
# src/asr_test/audio/output.py — change only the class line and imports,
# body is unchanged from today's AudioOutput
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd

from ..interfaces.audio_sink import AudioSinkBase

_FADE_FRAMES = 64


class LocalAudioSink(AudioSinkBase):
    def __init__(self, rate: int, blocksize: int = 1024, on_played=None):
        # ... identical body to today's AudioOutput.__init__ ...
```

(Keep every other method — `_callback`, `push`, `flush`, `playing`,
`elapsed_ms`, `close` — byte-for-byte identical to today's `AudioOutput`;
only the class name and base class change.)

- [x] **Step 6: Update the one call site in `agent.py`**

```python
# src/asr_test/agent.py
from .audio.output import LocalAudioSink   # was: from .audio.output import AudioOutput
```

```python
# and in Agent.__init__, the AudioOutput(...) construction becomes:
self.audio_out = LocalAudioSink(
    self.tts.sample_rate,
    blocksize=config.OUT_BLOCK,
    on_played=lambda a: self.echo.note_playback(a, self.tts.sample_rate),
)
```

- [x] **Step 7: Verify no `AudioOutput` references remain**

Run: `grep -rn "AudioOutput" src/ main.py`
Expected: no output (rename is complete everywhere).

- [x] **Step 8: Run tests to verify they pass, and confirm nothing else broke**

Run: `uv run pytest tests/ -v`
Expected: all tests pass (Task 1's + this task's).

Run: `uv run python -m py_compile main.py src/asr_test/agent.py src/asr_test/audio/output.py`
Expected: no errors.

- [x] **Step 9: Commit**

```bash
git add src/asr_test/interfaces/audio_sink.py src/asr_test/interfaces/__init__.py src/asr_test/audio/output.py src/asr_test/agent.py tests/test_local_audio_sink.py
git commit -m "refactor: extract AudioSinkBase, rename AudioOutput to LocalAudioSink"
```

---

### Task 3: `Agent.feed_audio()` / `start()` / `shutdown()` — transport-agnostic pipeline

The core of Phase 1: `Agent` stops owning a `sounddevice.InputStream`
directly. `feed_audio()` becomes the one entry point any transport uses
to push a frame in; `start()`/`shutdown()` let a caller manage the
background threads without also owning an input device. `run()` becomes
a thin wrapper combining `start()` with a real `sd.InputStream`, so
local-mode behavior is provably unchanged.

Also adds `tests/fakes.py` — lightweight fake `VadBase`/`SttBase`/
`TtsBase`/`LlmBase`/`AudioSinkBase` implementations reused by this task
and Tasks 5 and 6, so none of those tests need real models, real audio
hardware, or real network calls.

**Files:**
- Create: `tests/fakes.py`
- Test: `tests/test_agent.py`
- Modify: `src/asr_test/agent.py`

**Interfaces:**
- Consumes: `LocalAudioSink` (Task 2), `AudioSinkBase` (Task 2).
- Produces: `Agent.__init__(..., audio_sink: AudioSinkBase | None = None)`
  (new optional param); `Agent.feed_audio(frame: np.ndarray) -> None`;
  `Agent.start() -> list[threading.Thread]`; `Agent.shutdown(threads:
  list[threading.Thread]) -> None`. `Agent.run()` keeps its existing
  no-argument signature and behavior. `tests/fakes.py` exports `FakeVad`,
  `FakeStt`, `FakeLlm`, `FakeTts`, `FakeAudioSink` — Task 5 modifies
  `FakeLlm`'s `stream()` signature (documented there); Task 6 reuses all
  five as-is.

- [x] **Step 1: Write the fakes**

```python
# tests/fakes.py
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
```

Note: `FakeLlm.stream(user_text, cancel)` intentionally matches today's
`LlmBase` contract (a plain string, owned history elsewhere) — Task 5
changes this signature to `stream(messages, cancel)` and updates every
place that constructs a `FakeLlm` call assertion. This task's own test
below does not assert on `fake_llm.calls`' exact shape, specifically so
Task 5's interface change doesn't require touching it again.

- [x] **Step 2: Write the failing test**

```python
# tests/test_agent.py
import threading
import time

import numpy as np
import pytest

from asr_test import config
from asr_test.agent import Agent
from fakes import FakeAudioSink, FakeLlm, FakeStt, FakeTts, FakeVad


def _build_agent(**overrides):
    kwargs = dict(
        vad=FakeVad(start_at=1, end_at=3),
        stt=FakeStt("hello"),
        tts=FakeTts(),
        llm=FakeLlm("hi there"),
        audio_sink=FakeAudioSink(),
    )
    kwargs.update(overrides)
    return Agent(**kwargs)


def test_feed_audio_queues_a_frame():
    agent = _build_agent()
    frame = np.zeros(config.FRAME, dtype=np.float32)
    agent.feed_audio(frame)
    assert agent.mic_q.qsize() == 1


def test_start_returns_three_running_threads_and_shutdown_stops_them():
    agent = _build_agent()
    threads = agent.start()
    assert len(threads) == 3
    assert all(t.is_alive() for t in threads)
    agent.shutdown(threads)
    for t in threads:
        t.join(timeout=1.0)
        assert not t.is_alive()


def test_end_to_end_frame_to_response(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    fake_stt = FakeStt("hello")
    fake_llm = FakeLlm("hi there")
    fake_sink = FakeAudioSink()
    agent = _build_agent(stt=fake_stt, llm=fake_llm, audio_sink=fake_sink)

    threads = agent.start()
    try:
        frame = np.zeros(config.FRAME, dtype=np.float32)
        for _ in range(3):
            agent.feed_audio(frame)

        deadline = time.time() + 2.0
        while not fake_sink.pushed and time.time() < deadline:
            time.sleep(0.02)

        assert len(fake_stt.calls) == 1
        assert len(fake_llm.calls) == 1
        assert fake_sink.pushed  # TTS output reached the sink
    finally:
        agent.shutdown(threads)
```

- [x] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_agent.py -v`
Expected: FAIL — `TypeError: Agent.__init__() got an unexpected keyword
argument 'audio_sink'` (doesn't exist yet), and `feed_audio`/`start`/
`shutdown` don't exist yet either.

- [x] **Step 4: Implement `audio_sink` param, `feed_audio`, `start`, `shutdown`**

```python
# src/asr_test/agent.py — changes to Agent.__init__ signature and body
from .interfaces import AudioSinkBase, LlmBase, SttBase, TtsBase, VadBase
# (add AudioSinkBase to the existing interfaces import)

class Agent:
    def __init__(
        self,
        vad: VadBase | None = None,
        stt: SttBase | None = None,
        tts: TtsBase | None = None,
        llm: LlmBase | None = None,
        trigger_word: str | None = None,
        audio_sink: AudioSinkBase | None = None,
    ):
        ...
        self.audio_out = audio_sink or LocalAudioSink(
            self.tts.sample_rate,
            blocksize=config.OUT_BLOCK,
            on_played=lambda a: self.echo.note_playback(a, self.tts.sample_rate),
        )
        ...
```

```python
    def feed_audio(self, frame: np.ndarray) -> None:
        """Transport-agnostic entry point for one mono audio frame.
        Replaces the sounddevice-specific mic_callback signature — any
        transport (local InputStream, websocket receive loop) calls
        this the same way."""
        self.mic_q.put(frame)

    def start(self) -> list[threading.Thread]:
        """Spawn the vad/worker/tts threads without opening any input
        device. run() (local mode) wraps this; server mode calls it
        directly per connection."""
        threads = [
            threading.Thread(target=self.vad_thread, daemon=True),
            threading.Thread(target=self.worker_thread, daemon=True),
            threading.Thread(target=self.tts_thread, daemon=True),
        ]
        for t in threads:
            t.start()
        return threads

    def shutdown(self, threads: list[threading.Thread]) -> None:
        self.stop.set()
        self.cancel.set()
        for t in threads:
            t.join(timeout=1.0)
        self.audio_out.close()
```

- [x] **Step 5: Rewrite `run()` and remove `mic_callback()` as a thin wrapper over the new methods**

```python
    def run(self):
        threads = self.start()

        print("Listening — speak any time, including over the bot. Ctrl+C to stop.\n")

        def _on_frame(indata, frames, time_info, status):
            if status:
                print("Audio status:", status)
            self.feed_audio(indata.flatten().astype(np.float32))

        mic = sd.InputStream(
            samplerate=config.MIC_RATE,
            channels=1,
            dtype="float32",
            blocksize=config.FRAME,
            callback=_on_frame,
        )

        with mic:
            try:
                while True:
                    time.sleep(0.2)
            except KeyboardInterrupt:
                print("\nStopping...")
                self.shutdown(threads)
                self.report()
```

Remove the old standalone `mic_callback` method entirely (its body is
now the `_on_frame` closure above) — no alias, per this plan's
no-backwards-compat-shims constraint.

- [x] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/ -v`
Expected: all tests pass (Tasks 1, 2, and this task's).

- [ ] **Step 7: Manual smoke test — confirm local mode (`main.py`) still behaves identically**

Run: `uv run main.py` (LM Studio running, as usual), speak a sentence,
confirm you get a spoken response and Ctrl+C still prints the session
summary. This is the hardware-touching path pytest can't cover — see
Global Constraints.

- [x] **Step 8: Commit**

```bash
git add tests/fakes.py tests/test_agent.py src/asr_test/agent.py
git commit -m "refactor: make Agent transport-agnostic via feed_audio/start/shutdown"
```

---

### Task 4: `WebSocketAudioSink`

The second `AudioSinkBase` implementation. No real output device pulls
frames on a schedule over a websocket, so this drives its own pacing
with a background thread, reproducing `LocalAudioSink`'s exact
push/flush/playing/elapsed_ms/underrun semantics so `agent.py`'s
barge-in logic needs no changes to work under this transport.

**Files:**
- Create: `src/asr_test/audio/ws_sink.py`
- Test: `tests/test_ws_sink.py`

**Interfaces:**
- Consumes: `AudioSinkBase` (Task 2), `float32_to_pcm16`/`pcm16_to_float32`
  (Task 1).
- Produces: `WebSocketAudioSink(websocket, loop: asyncio.AbstractEventLoop,
  rate: int, blocksize: int = 1024)` implementing `AudioSinkBase`, where
  `websocket` is any object with an `async def send_bytes(data: bytes)`
  method (this is FastAPI/Starlette's `WebSocket` interface — Task 6
  passes a real one; this task's tests pass a fake). Task 6 constructs
  this per-connection.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_ws_sink.py
import asyncio
import threading
import time

import numpy as np
import pytest

from asr_test.audio.ws_sink import WebSocketAudioSink
from asr_test.interfaces.audio_sink import AudioSinkBase
from asr_test.utils import pcm16_to_float32


class _FakeWebSocket:
    def __init__(self):
        self.sent: list[bytes] = []

    async def send_bytes(self, data: bytes) -> None:
        self.sent.append(data)


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    t = threading.Thread(target=lp.run_forever, daemon=True)
    t.start()
    yield lp
    lp.call_soon_threadsafe(lp.stop)
    t.join(timeout=1.0)


def test_is_audio_sink_base(loop):
    sink = WebSocketAudioSink(_FakeWebSocket(), loop, rate=8000, blocksize=8)
    assert isinstance(sink, AudioSinkBase)
    sink.close()


def test_drain_once_returns_pushed_audio(loop):
    sink = WebSocketAudioSink(_FakeWebSocket(), loop, rate=8000, blocksize=4)
    sink.push(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
    block = sink._drain_once()
    assert np.allclose(block, 1.0)
    sink.close()


def test_drain_once_pads_and_counts_underrun_when_playing(loop):
    sink = WebSocketAudioSink(_FakeWebSocket(), loop, rate=8000, blocksize=8)
    sink.push(np.ones(3, dtype=np.float32))
    sink._drain_once()
    assert sink.underruns == 1
    sink.close()


def test_flush_stops_playing(loop):
    sink = WebSocketAudioSink(_FakeWebSocket(), loop, rate=8000, blocksize=8)
    sink.push(np.ones(8, dtype=np.float32))
    assert sink.playing
    sink.flush()
    assert not sink.playing
    sink.close()


def test_background_thread_sends_audio_over_websocket(loop):
    ws = _FakeWebSocket()
    sink = WebSocketAudioSink(ws, loop, rate=8000, blocksize=8)
    sink.push(np.ones(800, dtype=np.float32))
    time.sleep(0.2)
    sink.close()
    assert len(ws.sent) > 0
    assert pcm16_to_float32(ws.sent[0]).size == 8
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ws_sink.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'asr_test.audio.ws_sink'`.

- [x] **Step 3: Implement `WebSocketAudioSink`**

```python
# src/asr_test/audio/ws_sink.py
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

    def _run(self):
        interval = self.blocksize / self.rate
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            next_tick += interval
            block = self._drain_once()
            asyncio.run_coroutine_threadsafe(
                self.websocket.send_bytes(float32_to_pcm16(block)), self.loop
            )
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
```

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ws_sink.py -v`
Expected: 5 passed.

- [x] **Step 5: Commit**

```bash
git add src/asr_test/audio/ws_sink.py tests/test_ws_sink.py
git commit -m "feat: add WebSocketAudioSink"
```

---

### Task 5: Stateless `LlmBase.stream()` + `Agent`-owned conversation history

Changes `LlmBase.stream()` from `(user_text: str, cancel)` — which owns
conversation history internally — to `(messages: list[dict], cancel)`,
stateless. This is required before Task 6: sharing one `LlmBase`
instance across concurrent server sessions is only safe if it holds no
per-conversation state. `Agent` becomes the owner of conversation
history instead (via `self.conversation`), which is also the shape
Phase 3's LangChain harness will expect.

**Files:**
- Modify: `src/asr_test/interfaces/llm.py`
- Modify: `src/asr_test/llm/openai_compatible.py`
- Modify: `src/asr_test/agent.py` (`respond()` and `__init__`)
- Modify: `src/asr_test/config.py` (add `HISTORY_TURNS`)
- Modify: `tests/fakes.py` (`FakeLlm.stream()` signature)
- Test: `tests/test_openai_compatible_llm.py`
- Test: extend `tests/test_agent.py`

**Interfaces:**
- Produces: `LlmBase.stream(messages: list[dict], cancel:
  threading.Event) -> Iterator[str]`. `messages` is prior turns only
  (`{"role": "user"|"assistant", "content": str}`, oldest first, no
  system message) — the engine prepends its own system prompt.
  `Agent.conversation: list[dict]` holds the running history; Phase 2/3
  work (not in this plan) can read/reset it.

- [x] **Step 1: Update `FakeLlm` to the new signature**

```python
# tests/fakes.py — replace FakeLlm.stream
class FakeLlm(LlmBase):
    def __init__(self, reply: str = "hi there"):
        self.reply = reply
        self.last_ttft: float | None = 0.01
        self.last_total: float | None = 0.02
        self.calls: list[list[dict]] = []

    def stream(self, messages: list[dict], cancel: threading.Event) -> Iterator[str]:
        self.calls.append(messages)
        for word in self.reply.split():
            yield word + " "
```

- [x] **Step 2: Write the failing tests**

```python
# tests/test_openai_compatible_llm.py
import threading
from unittest.mock import MagicMock

from asr_test.llm.openai_compatible import OpenAiCompatibleLlm


def _fake_chunk(content):
    chunk = MagicMock()
    chunk.choices = [MagicMock(delta=MagicMock(content=content))]
    return chunk


def test_stream_is_stateless_and_prepends_system_prompt(monkeypatch):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = [
        _fake_chunk("hel"), _fake_chunk("lo"), _fake_chunk(None),
    ]
    monkeypatch.setattr("asr_test.llm.openai_compatible.OpenAI", lambda **kw: mock_client)

    llm = OpenAiCompatibleLlm(system_prompt="sys", warmup=False)
    cancel = threading.Event()
    result = list(llm.stream([{"role": "user", "content": "hi"}], cancel))

    assert result == ["hel", "lo"]
    sent = mock_client.chat.completions.create.call_args.kwargs["messages"]
    assert sent[0] == {"role": "system", "content": "sys"}
    assert sent[1] == {"role": "user", "content": "hi"}
    assert not hasattr(llm, "history")


def test_stream_stops_on_cancel(monkeypatch):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = [
        _fake_chunk("a"), _fake_chunk("b"), _fake_chunk("c"),
    ]
    monkeypatch.setattr("asr_test.llm.openai_compatible.OpenAI", lambda **kw: mock_client)

    llm = OpenAiCompatibleLlm(warmup=False)
    cancel = threading.Event()
    pieces = []
    for i, piece in enumerate(llm.stream([{"role": "user", "content": "hi"}], cancel)):
        pieces.append(piece)
        if i == 0:
            cancel.set()

    assert pieces == ["a"]
```

```python
# tests/test_agent.py — add this test, reusing _build_agent from Task 3
def test_agent_tracks_conversation_history_across_turns(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    fake_llm = FakeLlm(reply="hi there")
    agent = _build_agent(
        vad=FakeVad(start_at=1, end_at=3), stt=FakeStt("hello"), llm=fake_llm,
    )
    threads = agent.start()
    try:
        frame = np.zeros(config.FRAME, dtype=np.float32)

        for _ in range(3):
            agent.feed_audio(frame)
        deadline = time.time() + 2.0
        while len(fake_llm.calls) < 1 and time.time() < deadline:
            time.sleep(0.02)
        assert fake_llm.calls[0] == [{"role": "user", "content": "hello"}]

        for _ in range(3):
            agent.feed_audio(frame)
        deadline = time.time() + 2.0
        while len(fake_llm.calls) < 2 and time.time() < deadline:
            time.sleep(0.02)
        assert fake_llm.calls[1] == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there "},
            {"role": "user", "content": "hello"},
        ]
    finally:
        agent.shutdown(threads)
```

- [x] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_openai_compatible_llm.py tests/test_agent.py -v`
Expected: FAIL — `TypeError` on `llm.stream(...)` call signature
mismatch, and `Agent` has no `conversation` attribute yet.

- [x] **Step 4: Update the interface**

```python
# src/asr_test/interfaces/llm.py
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator


class LlmBase(ABC):
    """Chat LLM: streams a response to one turn as text pieces.
    Stateless — callers pass the full prior-turns message list each
    call; implementations own no conversation history. `last_ttft` /
    `last_total` reflect the most recently completed `stream()` call.
    """

    last_ttft: float | None
    last_total: float | None

    @abstractmethod
    def stream(self, messages: list[dict], cancel: threading.Event) -> Iterator[str]: ...
```

- [x] **Step 5: Update `OpenAiCompatibleLlm`**

```python
# src/asr_test/llm/openai_compatible.py
from __future__ import annotations

import threading
import time
from collections.abc import Iterator

from openai import OpenAI

from ..interfaces.llm import LlmBase


class OpenAiCompatibleLlm(LlmBase):
    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        model: str = "lfm2.5-230m",
        system_prompt: str = (
            "You are a concise voice assistant. Answer in one or two short sentences. "
            "Plain text only — no markdown, lists, or emoji. Your words are spoken aloud."
        ),
        max_tokens: int = 120,
        timeout: float = 30,
        warmup: bool = True,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.timeout = timeout

        self.last_ttft: float | None = None
        self.last_total: float | None = None

        if warmup:
            t0 = time.perf_counter()
            try:
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=1,
                    timeout=self.timeout,
                )
                print(f"  llm warm-up: {time.perf_counter() - t0:.2f}s")
            except Exception as e:
                print(f"  llm warm-up failed ({e}) — is LM Studio running?")

    def stream(self, messages: list[dict], cancel: threading.Event) -> Iterator[str]:
        full_messages = [{"role": "system", "content": self.system_prompt}] + messages

        t_start = time.perf_counter()
        self.last_ttft = None
        self.last_total = None

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            max_tokens=self.max_tokens,
            temperature=0.7,
            stream=True,
            timeout=self.timeout,
        )
        for chunk in completion:
            if cancel.is_set():
                break
            if not chunk.choices:
                continue
            piece = chunk.choices[0].delta.content
            if piece:
                if self.last_ttft is None:
                    self.last_ttft = time.perf_counter() - t_start
                yield piece

        self.last_total = time.perf_counter() - t_start
```

Note what's removed vs. today: `history_turns` constructor param,
`self.history`, and the `full: list[str]` accumulator + history-append
logic at the end of `stream()` — all gone. `Agent` takes over that job
next.

- [x] **Step 6: Add `HISTORY_TURNS` to config**

```python
# src/asr_test/config.py — add near the other tunables
HISTORY_TURNS = 3   # how many prior user/assistant turn-pairs to include
                     # as LLM context; was previously owned by
                     # OpenAiCompatibleLlm itself, now Agent's job since
                     # the LLM engine is stateless and shared across
                     # sessions in server mode.
```

- [x] **Step 7: Update `Agent.__init__` and `respond()`**

```python
# src/asr_test/agent.py — in __init__, after existing state init:
        self.conversation: list[dict] = []
```

```python
    def respond(self, text: str, turn: int, stt_t: float):
        buf = ""
        first = True
        spoken: list[str] = []
        full_response: list[str] = []
        chunk_no = 0
        turn_start = self.turn_start or time.perf_counter()

        def limit() -> int:
            return config.FIRST_CHUNK_CHARS if first else config.MAX_CHUNK_CHARS

        def enqueue(chunk: str) -> bool:
            nonlocal first, chunk_no
            chunk = chunk.strip()
            if not chunk:
                return True
            if self.cancel.is_set() or turn != self.current_turn():
                return False
            chunk_no += 1
            first = False
            self.tts_q.put((turn, chunk_no, chunk, stt_t, turn_start))
            spoken.append(chunk)
            return True

        def drain(buf: str) -> str | None:
            while True:
                lim = limit()
                m = config.SENTENCE_END.search(buf)
                if m and m.start() <= lim:
                    piece, buf = buf[: m.start()], buf[m.end():]
                elif len(buf) >= lim:
                    cut = buf.rfind(" ", 0, lim)
                    if cut <= 0:
                        break
                    piece, buf = buf[:cut], buf[cut + 1:]
                else:
                    break
                if not enqueue(piece):
                    return None
            return buf

        if config.REALTIME_LOG:
            print(f"      [LLM] triggered      \"{text}\"")

        messages = self.conversation[-config.HISTORY_TURNS * 2:] + [
            {"role": "user", "content": text}
        ]

        try:
            for piece in self.llm.stream(messages, self.cancel):
                if self.cancel.is_set() or turn != self.current_turn():
                    return
                full_response.append(piece)
                buf += piece
                buf = drain(buf)
                if buf is None:
                    return
        except Exception as e:
            print(f"   [llm failed: {e}]")
            return

        if buf.strip():
            enqueue(buf)

        if self.llm.last_ttft is not None:
            self.m_ttft.append(self.llm.last_ttft)
        if self.llm.last_total is not None:
            self.m_llm.append(self.llm.last_total)
            if config.VERBOSE_TIMING:
                print(f"      llm: ttft {self.llm.last_ttft or 0:.2f}s / "
                      f"total {self.llm.last_total:.2f}s")

        if full_response:
            self.conversation.append({"role": "user", "content": text})
            self.conversation.append({"role": "assistant", "content": "".join(full_response)})

        if spoken:
            print(f"BOT:  {' '.join(spoken)}")
```

(Only the `messages` construction, the `full_response` accumulation,
and the final history-append block are new — `limit`/`enqueue`/`drain`
and the chunking logic are unchanged from today.)

- [x] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/ -v`
Expected: all tests pass (Tasks 1-4's plus this task's).

- [ ] **Step 9: Manual smoke test**

Run: `uv run main.py`, have a two-turn conversation, confirm the second
answer shows awareness of the first (proving `self.conversation` is
actually being threaded through) and Ctrl+C still prints the summary
cleanly.

- [x] **Step 10: Commit**

```bash
git add src/asr_test/interfaces/llm.py src/asr_test/llm/openai_compatible.py src/asr_test/agent.py src/asr_test/config.py tests/fakes.py tests/test_openai_compatible_llm.py tests/test_agent.py
git commit -m "refactor: make LlmBase stateless, move conversation history to Agent"
```

---

### Task 6: FastAPI server with multi-session `/ws` endpoint

Ties Tasks 2-5 together: a FastAPI app that constructs STT/TTS/LLM
engines once, and for each websocket connection builds a fresh `Agent`
(with its own `SileroVad` and its own `WebSocketAudioSink`) sharing
those engine instances. Proves the "shared heavy models, isolated
per-session state" resource story from the spec with an actual
concurrency test.

**Files:**
- Create: `server.py` (repo root, alongside `main.py`)
- Modify: `pyproject.toml` (add `fastapi`, `uvicorn[standard]`)
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `Agent` (`start`/`shutdown`/`feed_audio`/`audio_sink` param
  from Task 3), `WebSocketAudioSink` (Task 4), `pcm16_to_float32`/
  `float32_to_pcm16` (Task 1), `SttBase`/`TtsBase`/`LlmBase`/`VadBase`
  (existing interfaces).
- Produces: `create_app(stt: SttBase, tts: TtsBase, llm: LlmBase,
  vad_factory: Callable[[], VadBase] = <default SileroVad factory>) ->
  FastAPI`. Task 7's clients connect to the `/ws` endpoint this exposes.

- [x] **Step 1: Add FastAPI/uvicorn dependencies**

Run: `uv add fastapi "uvicorn[standard]"`

- [x] **Step 2: Write the failing tests**

```python
# tests/test_server.py
import numpy as np
from starlette.testclient import TestClient

from asr_test import config
from asr_test.utils import float32_to_pcm16
from fakes import FakeLlm, FakeStt, FakeTts, FakeVad
from server import create_app


def _make_client(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    fake_llm = FakeLlm(reply="hi there")
    app = create_app(
        stt=FakeStt("hello"),
        tts=FakeTts(),
        llm=fake_llm,
        vad_factory=lambda: FakeVad(start_at=1, end_at=3),
    )
    return TestClient(app), fake_llm


def test_ws_endpoint_sends_ready_event_then_audio_reply(monkeypatch):
    client, _ = _make_client(monkeypatch)
    with client.websocket_connect("/ws") as ws:
        ready = ws.receive_json()
        assert ready["event"] == "ready"

        frame = np.zeros(config.FRAME, dtype=np.float32)
        for _ in range(3):
            ws.send_bytes(float32_to_pcm16(frame))

        reply = ws.receive_bytes()
        assert len(reply) > 0


def test_two_concurrent_sessions_have_independent_history(monkeypatch):
    client, fake_llm = _make_client(monkeypatch)
    frame = np.zeros(config.FRAME, dtype=np.float32)

    with client.websocket_connect("/ws") as ws_a, client.websocket_connect("/ws") as ws_b:
        ws_a.receive_json()
        ws_b.receive_json()
        for ws in (ws_a, ws_b):
            for _ in range(3):
                ws.send_bytes(float32_to_pcm16(frame))
            ws.receive_bytes()

    assert len(fake_llm.calls) == 2
    assert fake_llm.calls[0] == fake_llm.calls[1] == [{"role": "user", "content": "hello"}]
```

- [x] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_server.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'server'`.

- [x] **Step 4: Implement `server.py`**

```python
# server.py
"""
FastAPI websocket server — Phase 1 of the transport plan (see
docs/superpowers/specs/2026-09-09-transport-provider-harness-design.md).

Serves one endpoint, /ws: binary frames carry raw PCM16 mono audio in
both directions (16kHz in, matching the TTS engine's own sample_rate
out); one JSON "ready" event is sent right after connecting so a client
knows the output sample rate. Each connection gets its own Agent (own
VAD state, own conversation history) but shares the same STT/TTS/LLM
engine instances across all connections — those hold no per-session
state (see Task 5), so sharing them is what keeps memory flat regardless
of concurrent session count.

Usage:
    uv run server.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from asr_test import config
from asr_test.agent import Agent
from asr_test.audio.ws_sink import WebSocketAudioSink
from asr_test.interfaces import LlmBase, SttBase, TtsBase, VadBase
from asr_test.utils import pcm16_to_float32
from asr_test.vad import SileroVad


def _default_vad_factory() -> VadBase:
    return SileroVad(
        sample_rate=config.MIC_RATE,
        min_silence_ms=config.MIN_SILENCE_MS,
        speech_pad_ms=config.SPEECH_PAD_MS,
    )


def create_app(
    stt: SttBase,
    tts: TtsBase,
    llm: LlmBase,
    vad_factory: Callable[[], VadBase] = _default_vad_factory,
) -> FastAPI:
    app = FastAPI()

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        loop = asyncio.get_running_loop()

        await websocket.send_json({
            "event": "ready",
            "input_sample_rate": config.MIC_RATE,
            "output_sample_rate": tts.sample_rate,
        })

        sink = WebSocketAudioSink(websocket, loop, rate=tts.sample_rate, blocksize=config.OUT_BLOCK)
        agent = Agent(vad=vad_factory(), stt=stt, tts=tts, llm=llm, audio_sink=sink)
        threads = agent.start()
        try:
            while True:
                data = await websocket.receive_bytes()
                agent.feed_audio(pcm16_to_float32(data))
        except WebSocketDisconnect:
            pass
        finally:
            agent.shutdown(threads)

    return app


if __name__ == "__main__":
    import uvicorn

    from asr_test.llm import OpenAiCompatibleLlm
    from asr_test.stt import OnnxAsrEngine
    from asr_test.tts import KokoroTts

    config.ECHO_MODE = "duck"  # server can't verify a remote client has real headphone isolation

    app = create_app(stt=OnnxAsrEngine(), tts=KokoroTts(), llm=OpenAiCompatibleLlm())
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

- [x] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_server.py -v`
Expected: 2 passed.

Run: `uv run pytest tests/ -v`
Expected: all tests across the whole suite pass.

- [ ] **Step 6: Manual smoke test**

Run: `uv run server.py` (this loads the real Parakeet/Kokoro/LM-Studio
stack, same as `main.py` does) — confirm it starts and listens on
`:8000` without error. Full end-to-end audio verification happens in
Task 7 once a client exists to connect with.

- [x] **Step 7: Commit**

```bash
git add server.py pyproject.toml uv.lock tests/test_server.py
git commit -m "feat: add FastAPI multi-session websocket server"
```

---

### Task 7: Browser HTML client + Python CLI client

Two thin relay clients against the same `/ws` endpoint — neither
contains pipeline logic, so neither gets a pytest suite (there's no
pure logic to unit-test: it's mic capture, socket I/O, and playback, all
of which need real hardware/a real browser). Verified by the manual
smoke tests below instead, consistent with how this repo already treats
hardware-touching code.

**Files:**
- Create: `static/index.html`
- Create: `ws_client.py` (repo root)
- Modify: `server.py` (serve the static file)
- Modify: `pyproject.toml` (add `websockets`)

- [x] **Step 1: Add the `websockets` dependency for the CLI client**

Run: `uv add websockets`

- [x] **Step 2: Serve the HTML client from the FastAPI app**

```python
# server.py — add near the top-level imports
from pathlib import Path

from fastapi.responses import FileResponse
```

```python
# server.py — add inside create_app(), alongside the /ws route
    @app.get("/")
    async def index():
        return FileResponse(Path(__file__).parent / "static" / "index.html")
```

- [x] **Step 3: Write the HTML client**

```html
<!-- static/index.html -->
<!doctype html>
<html>
<head><title>asr-test</title></head>
<body>
  <button id="start">Start</button>
  <button id="stop" disabled>Stop</button>
  <pre id="log"></pre>
  <script>
    const log = (msg) => { document.getElementById("log").textContent += msg + "\n"; };
    let ws, audioCtx, source, processor, playCtx, outputSampleRate = 16000;

    function floatToPcm16(input) {
      const out = new Int16Array(input.length);
      for (let i = 0; i < input.length; i++) {
        const s = Math.max(-1, Math.min(1, input[i]));
        out[i] = s < 0 ? s * 32768 : s * 32767;
      }
      return out.buffer;
    }

    function pcm16ToFloat(buf) {
      const view = new Int16Array(buf);
      const out = new Float32Array(view.length);
      for (let i = 0; i < view.length; i++) out[i] = view[i] / 32768;
      return out;
    }

    async function start() {
      ws = new WebSocket(`ws://${location.host}/ws`);
      ws.binaryType = "arraybuffer";

      ws.onmessage = (event) => {
        if (typeof event.data === "string") {
          const msg = JSON.parse(event.data);
          if (msg.event === "ready") {
            outputSampleRate = msg.output_sample_rate;
            playCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: outputSampleRate });
          }
          log(JSON.stringify(msg));
          return;
        }
        const floatData = pcm16ToFloat(event.data);
        const buffer = playCtx.createBuffer(1, floatData.length, outputSampleRate);
        buffer.copyToChannel(floatData, 0);
        const src = playCtx.createBufferSource();
        src.buffer = buffer;
        src.connect(playCtx.destination);
        src.start();
      };

      audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      source = audioCtx.createMediaStreamSource(stream);
      processor = audioCtx.createScriptProcessor(512, 1, 1);
      processor.onaudioprocess = (e) => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(floatToPcm16(e.inputBuffer.getChannelData(0)));
        }
      };
      source.connect(processor);
      processor.connect(audioCtx.destination);

      document.getElementById("start").disabled = true;
      document.getElementById("stop").disabled = false;
    }

    function stop() {
      processor && processor.disconnect();
      source && source.disconnect();
      ws && ws.close();
      document.getElementById("start").disabled = false;
      document.getElementById("stop").disabled = true;
    }

    document.getElementById("start").onclick = start;
    document.getElementById("stop").onclick = stop;
  </script>
</body>
</html>
```

- [x] **Step 4: Write the CLI client**

```python
# ws_client.py
"""
Python CLI relay client for server.py — captures mic audio locally via
sounddevice and streams it to the FastAPI /ws endpoint over a
websocket; plays back whatever audio the server sends in return. No
pipeline logic lives here — this is the CLI equivalent of
static/index.html, both are dumb relays to the same server.

Usage:
    uv run server.py            # in one terminal
    uv run ws_client.py         # in another
"""

from __future__ import annotations

import argparse
import asyncio
import queue

import numpy as np
import sounddevice as sd
import websockets

from asr_test import config
from asr_test.utils import float32_to_pcm16, pcm16_to_float32


async def run(url: str):
    async with websockets.connect(url, max_size=None) as ws:
        ready = await ws.recv()
        print(f"server: {ready}")

        mic_q: queue.Queue[bytes] = queue.Queue()

        def on_mic(indata, frames, time_info, status):
            if status:
                print("mic status:", status)
            mic_q.put(float32_to_pcm16(indata.flatten().astype(np.float32)))

        out_stream = sd.OutputStream(samplerate=config.MIC_RATE, channels=1, dtype="float32")
        out_stream.start()

        async def sender():
            loop = asyncio.get_running_loop()
            while True:
                data = await loop.run_in_executor(None, mic_q.get)
                await ws.send(data)

        async def receiver():
            async for message in ws:
                if isinstance(message, bytes):
                    out_stream.write(pcm16_to_float32(message).reshape(-1, 1))
                else:
                    print(f"server: {message}")

        with sd.InputStream(
            samplerate=config.MIC_RATE, channels=1, dtype="float32",
            blocksize=config.FRAME, callback=on_mic,
        ):
            await asyncio.gather(sender(), receiver())


def main():
    parser = argparse.ArgumentParser(description="CLI client for server.py")
    parser.add_argument("--url", default="ws://localhost:8000/ws")
    args = parser.parse_args()
    try:
        asyncio.run(run(args.url))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Manual smoke test — HTML client**

Run: `uv run server.py`, open `http://localhost:8000/` in a browser,
click Start, grant mic permission, speak, confirm you hear a spoken
response and the log shows the `ready` event. Click Stop, confirm the
websocket closes cleanly (check the server log for no traceback).

- [ ] **Step 6: Manual smoke test — CLI client**

Run: `uv run server.py` in one terminal, `uv run ws_client.py` in
another, speak, confirm the same round-trip works. Then run a second
`uv run ws_client.py` (or the HTML client) concurrently and confirm both
sessions get independent responses (proving multi-session works
end-to-end with real hardware, not just the faked test in Task 6).

- [x] **Step 7: Commit**

```bash
git add static/index.html ws_client.py server.py pyproject.toml uv.lock
git commit -m "feat: add browser HTML client and Python CLI client"
```

---

### Task 8: README updates

Documents all three run modes, the new dependencies, and the
`ECHO_MODE`/`HISTORY_TURNS` config knobs this plan introduces — keeping
the README's existing standard of documenting real, verified behavior
rather than aspirational claims.

**Files:**
- Modify: `README.md`

- [x] **Step 1: Add a "Running: local vs. server" section**

Add a new section after "## Usage" explaining the three modes:

```markdown
## Running: local vs. server

Three ways to run this, all sharing the same VAD/STT/LLM/TTS pipeline:

1. **Local** (`uv run main.py`) — everything in one process, direct
   `sounddevice` mic/speaker access. No network involved. This is the
   original mode and still the simplest for single-user local use.
2. **FastAPI server + browser client** (`uv run server.py`, then open
   `http://localhost:8000/`) — the pipeline runs server-side; the
   browser captures your mic and plays responses via a websocket at
   `/ws`. Supports multiple simultaneous browser tabs/users, each with
   independent conversation history.
3. **FastAPI server + CLI client** (`uv run server.py`, then in another
   terminal `uv run ws_client.py`) — same server, a Python relay client
   instead of a browser. Useful for scripting/headless use, or testing
   the server without a browser.

Modes 2 and 3 share 100% of the server-side code — the only difference
is which relay client captures/plays your audio.

Server mode defaults `config.ECHO_MODE` to `"duck"` regardless of the
module-level default, since the server has no way to verify a remote
client has real headphone isolation (`"headphones"` mode assumes that
and would mistake the bot's own voice for your speech otherwise).
```

- [x] **Step 2: Document the new dependencies in the existing dependency-related prose**

Add a short note wherever `pyproject.toml`/`uv sync` is discussed: `fastapi`,
`uvicorn[standard]`, and `websockets` are new dependencies added for
server mode (Task 6/7) — `pytest` is a dev-only dependency (Task 1),
not needed to run the app, only to run `uv run pytest`.

- [x] **Step 3: Add `HISTORY_TURNS` to the Configuration section's bullet list**

```markdown
- **`HISTORY_TURNS`** (default `3`) — how many prior user/assistant turn
  pairs `Agent` includes as LLM context. Previously owned by the LLM
  engine itself; moved to `Agent` so the LLM engine can be a single
  shared, stateless instance across concurrent server sessions.
```

- [x] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document local/server run modes and new config knobs"
```

---

## Known simplification vs. the spec (self-review finding)

The spec's WebSocket protocol table also lists `turn_start`/`bot_text`/
`interrupted` server→client JSON events (for UI captions/status) and a
client→server `config` JSON handshake (per-connection `trigger_word`/TTS
choice sent right after connecting). This plan implements only the
`ready` event — the minimum needed for a client to know the output
sample rate and for the core audio round-trip to work end-to-end.

Deferred deliberately, not dropped silently: adding the other events
means threading an `on_event` callback through `Agent` (called from
`worker_thread`/`respond`/`interrupt`) and changing both clients (Task
7) to send an initial config message before audio — real scope, not a
one-line addition, and not required for a working multi-session
round-trip. Once this plan lands and is verified working end-to-end,
that's a small, clean follow-up task (same file set: `agent.py`,
`server.py`, `static/index.html`, `ws_client.py`) rather than something
worth blocking this plan on.

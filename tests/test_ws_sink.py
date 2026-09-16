import asyncio
import threading
import time

import numpy as np
import pytest

from pos.audio.ws_sink import WebSocketAudioSink
from pos.interfaces.audio_sink import AudioSinkBase
from pos.utils import pcm16_to_float32


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


def _stop_pacing_thread(sink):
    """The background thread starts draining immediately in __init__ —
    stop it before probing _drain_once() directly, so the test isn't
    racing the pacing thread's own concurrent consumption of the same
    buffer."""
    sink._stop.set()
    sink._thread.join(timeout=1.0)


def test_drain_once_returns_pushed_audio(loop):
    sink = WebSocketAudioSink(_FakeWebSocket(), loop, rate=8000, blocksize=4)
    _stop_pacing_thread(sink)
    sink.push(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
    sink._drain_once()  # first block fades in from the initial silent state — see test below
    sink.push(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
    block = sink._drain_once()
    assert np.allclose(block, 1.0)


def test_drain_once_pads_and_counts_underrun_when_playing(loop):
    sink = WebSocketAudioSink(_FakeWebSocket(), loop, rate=8000, blocksize=8)
    _stop_pacing_thread(sink)
    sink.push(np.ones(3, dtype=np.float32))
    sink._drain_once()
    assert sink.underruns == 1


def test_flush_stops_playing(loop):
    sink = WebSocketAudioSink(_FakeWebSocket(), loop, rate=8000, blocksize=8)
    sink.push(np.ones(8, dtype=np.float32))
    assert sink.playing
    sink.flush()
    assert not sink.playing
    sink.close()


def test_schedule_send_returns_false_without_raising_when_loop_closed(loop):
    ws = _FakeWebSocket()
    sink = WebSocketAudioSink(ws, loop, rate=8000, blocksize=8)
    _stop_pacing_thread(sink)

    closed_loop = asyncio.new_event_loop()
    closed_loop.close()
    sink.loop = closed_loop

    assert sink._schedule_send(b"\x00\x00") is False


def test_run_stops_cleanly_instead_of_crashing_when_loop_closes_mid_tick(loop):
    ws = _FakeWebSocket()
    sink = WebSocketAudioSink(ws, loop, rate=8000, blocksize=8)
    sink.push(np.ones(800, dtype=np.float32))
    time.sleep(0.1)  # let it send at least one real block

    closed_loop = asyncio.new_event_loop()
    closed_loop.close()
    sink.loop = closed_loop  # simulate the server's loop tearing down mid-session

    time.sleep(0.2)  # _run()'s next tick should see the closed loop and stop itself
    assert not sink._thread.is_alive()
    sink._stop.set()  # already stopped itself, but keep close() a no-op either way


def test_background_thread_sends_audio_over_websocket(loop):
    ws = _FakeWebSocket()
    sink = WebSocketAudioSink(ws, loop, rate=8000, blocksize=8)
    sink.push(np.ones(800, dtype=np.float32))
    time.sleep(0.2)
    sink.close()
    assert len(ws.sent) > 0
    assert pcm16_to_float32(ws.sent[0]).size == 8

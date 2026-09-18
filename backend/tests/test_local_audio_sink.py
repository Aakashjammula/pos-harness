import numpy as np
import pytest

from pos.audio.output import LocalAudioSink
from pos.interfaces.audio_sink import AudioSinkBase


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
    monkeypatch.setattr("pos.audio.output.sd.OutputStream", _FakeStream)
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

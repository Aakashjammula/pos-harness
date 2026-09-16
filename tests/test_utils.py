import threading
import time

import numpy as np
import pytest

from pos.utils import (
    float32_to_pcm16,
    list_input_devices,
    pcm16_to_float32,
    resolve_input_device,
    start_mute_toggle_listener,
)

_FAKE_DEVICES = [
    {"name": "Speakers (Realtek)", "max_input_channels": 0},
    {"name": "Microphone Array (Intel)", "max_input_channels": 4},
    {"name": "Headset Mic (USB)", "max_input_channels": 1},
]


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
    assert values[1] == -32767
    assert values[2] == 0


def test_roundtrip_is_close():
    original = np.array([0.5, -0.25, 0.1], dtype=np.float32)
    restored = pcm16_to_float32(float32_to_pcm16(original))
    assert np.allclose(original, restored, atol=1e-4)


def test_list_input_devices_excludes_output_only_devices(monkeypatch):
    monkeypatch.setattr("sounddevice.query_devices", lambda: _FAKE_DEVICES)
    result = list_input_devices()
    assert len(result) == 2
    assert "Speakers" not in "".join(result)
    assert "Microphone Array (Intel)" in result[0]
    assert "Headset Mic (USB)" in result[1]


def test_resolve_input_device_by_index(monkeypatch):
    monkeypatch.setattr("sounddevice.query_devices", lambda: _FAKE_DEVICES)
    assert resolve_input_device("2") == 2


def test_resolve_input_device_by_name_substring(monkeypatch):
    monkeypatch.setattr("sounddevice.query_devices", lambda: _FAKE_DEVICES)
    assert resolve_input_device("headset") == 2
    assert resolve_input_device("intel") == 1


def test_resolve_input_device_none_means_default():
    assert resolve_input_device(None) is None


def test_resolve_input_device_raises_on_no_match(monkeypatch):
    monkeypatch.setattr("sounddevice.query_devices", lambda: _FAKE_DEVICES)
    with pytest.raises(ValueError, match="no input device matching"):
        resolve_input_device("nonexistent-device")


def test_mute_toggle_listener_toggles_on_each_input_line(monkeypatch):
    class _FakeInput:
        def __init__(self, n):
            self.n = n

        def __call__(self):
            if self.n <= 0:
                raise EOFError
            self.n -= 1
            return ""

    monkeypatch.setattr("builtins.input", _FakeInput(2))

    muted = threading.Event()
    start_mute_toggle_listener(muted)

    deadline = time.time() + 2.0
    while not muted.is_set() and time.time() < deadline:
        time.sleep(0.02)
    assert muted.is_set()

    deadline = time.time() + 2.0
    while muted.is_set() and time.time() < deadline:
        time.sleep(0.02)
    assert not muted.is_set()

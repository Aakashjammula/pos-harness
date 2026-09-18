from __future__ import annotations

import threading

import numpy as np


def resample_linear(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or x.size == 0:
        return x
    n_out = int(round(x.size * dst / src))
    return np.interp(
        np.linspace(0.0, x.size - 1, n_out, dtype=np.float64),
        np.arange(x.size, dtype=np.float64),
        x,
    ).astype(np.float32)


def pcm16_to_float32(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def list_input_devices() -> list[str]:
    """Formatted list of input-capable audio devices, for a --list-mics
    CLI flag — each line is "<index>: <name> (in: <channels>)"."""
    import sounddevice as sd

    return [
        f"{i}: {d['name']} (in: {d['max_input_channels']})"
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]


def resolve_input_device(spec: str | None) -> int | None:
    """Resolve a --mic value to a sounddevice device index: a bare
    integer is used as-is, otherwise it's matched as a case-insensitive
    substring of the device name (first match wins). None means "use
    the system default input device" (sounddevice's own default)."""
    if spec is None:
        return None
    if spec.isdigit():
        return int(spec)

    import sounddevice as sd

    spec_lower = spec.lower()
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and spec_lower in d["name"].lower():
            return i
    raise ValueError(f"no input device matching {spec!r} — try --list-mics")


def start_mute_toggle_listener(muted: threading.Event, label: str = "mic") -> None:
    """Background daemon thread: pressing Enter in the terminal toggles
    `muted`. No extra dependency (curses etc.) — just a blocking input()
    loop on its own thread, since the mic/audio streams already run on
    their own threads independent of stdin."""

    def _loop():
        while True:
            try:
                input()
            except EOFError:
                return
            if muted.is_set():
                muted.clear()
                print(f"      [{label}] unmuted")
            else:
                muted.set()
                print(f"      [{label}] muted — press Enter again to unmute")

    threading.Thread(target=_loop, daemon=True).start()

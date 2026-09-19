"""
Python CLI relay client for server.py — captures mic audio locally via
sounddevice and streams it to the FastAPI /ws endpoint over a
websocket; plays back whatever audio the server sends in return. No
pipeline logic lives here — this is the CLI equivalent of the
web frontend, both are dumb relays to the same server.

Usage:
    uv run server.py            # in one terminal
    uv run ws_client.py         # in another

    # Config, forwarded to the server as query params (see server.py):
    uv run ws_client.py --tts supertonic --voice M1 --llm-model gemma-3-1b-it
    uv run ws_client.py --trigger-word "computer"
    uv run ws_client.py --vad-threshold 0.35 --vad-min-silence-ms 800

    # Pick an input device (index or a substring of its name):
    uv run ws_client.py --list-mics
    uv run ws_client.py --mic "USB"

    # Mute: once running, press Enter in this terminal to toggle
    # muting the mic (frames are simply not sent while muted) — press
    # Enter again to unmute.
"""

from __future__ import annotations

import argparse
import asyncio
import queue
import threading
import urllib.parse

import numpy as np
import sounddevice as sd
import websockets

from pos import config
from pos.utils import (
    float32_to_pcm16,
    list_input_devices,
    pcm16_to_float32,
    resolve_input_device,
    start_mute_toggle_listener,
)


async def run(url: str, device: int | None):
    async with websockets.connect(url, max_size=None) as ws:
        ready = await ws.recv()
        print(f"server: {ready}")

        muted = threading.Event()
        start_mute_toggle_listener(muted)

        mic_q: queue.Queue[bytes] = queue.Queue()

        def on_mic(indata, frames, time_info, status):
            if status:
                print("mic status:", status)
            if muted.is_set():
                return
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
            device=device,
            samplerate=config.MIC_RATE, channels=1, dtype="float32",
            blocksize=config.FRAME, callback=on_mic,
        ):
            await asyncio.gather(sender(), receiver())


def main():
    parser = argparse.ArgumentParser(description="CLI client for server.py")
    parser.add_argument("--url", default="ws://localhost:8000/ws")
    parser.add_argument("--tts", default=None, metavar="ENGINE", help="TTS engine (e.g. kokoro, supertonic)")
    parser.add_argument("--voice", default=None, help="Voice name, engine-specific")
    parser.add_argument("--llm-model", default=None, metavar="MODEL", help="LLM model name (as loaded in LM Studio)")
    parser.add_argument(
        "--trigger-word", default=None, metavar="PHRASE",
        help="Only respond to speech whose transcript leads with this phrase; "
             "everything else is ignored. Off by default.",
    )
    parser.add_argument(
        "--mic", default=None, metavar="DEVICE",
        help="Input device to use: an index or a substring of its name "
             "(see --list-mics). Default: system default input device.",
    )
    parser.add_argument(
        "--list-mics", action="store_true",
        help="Print available input devices and exit.",
    )
    parser.add_argument(
        "--vad-threshold", type=float, default=None, metavar="0-1",
        help="Speech probability threshold (server default: 0.5). Lower = "
             "more sensitive (catches quieter speech, more false positives).",
    )
    parser.add_argument(
        "--vad-min-silence-ms", type=int, default=None,
        help="How long a pause must last before a turn is considered "
             "finished (server default: 1200).",
    )
    parser.add_argument(
        "--vad-speech-pad-ms", type=int, default=None,
        help="Padding kept on each side of detected speech (server default: 300).",
    )
    args = parser.parse_args()

    if args.list_mics:
        for line in list_input_devices():
            print(line)
        return

    device = resolve_input_device(args.mic)

    params = {}
    if args.tts:
        params["tts"] = args.tts
    if args.voice:
        params["voice"] = args.voice
    if args.llm_model:
        params["llm_model"] = args.llm_model
    if args.trigger_word:
        params["trigger_word"] = args.trigger_word
    if args.vad_threshold is not None:
        params["vad_threshold"] = args.vad_threshold
    if args.vad_min_silence_ms is not None:
        params["vad_min_silence_ms"] = args.vad_min_silence_ms
    if args.vad_speech_pad_ms is not None:
        params["vad_speech_pad_ms"] = args.vad_speech_pad_ms
    url = args.url + ("?" + urllib.parse.urlencode(params) if params else "")

    try:
        asyncio.run(run(url, device))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

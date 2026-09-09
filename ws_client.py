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

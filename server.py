"""
FastAPI websocket server — Phase 1 of the transport plan (see
docs/superpowers/specs/2026-09-09-transport-provider-harness-design.md).

Serves one endpoint, /ws: binary frames carry raw PCM16 mono audio in
both directions (16kHz in, matching the TTS engine's own sample_rate
out); one JSON "ready" event is sent right after connecting so a client
knows the output sample rate. Each connection gets its own Agent (own
VAD state, own conversation history) but shares the same STT/TTS/LLM
engine instances across all connections — those hold no per-session
state (see agent.py's LlmBase.stream()), so sharing them is what keeps
memory flat regardless of concurrent session count.

Usage:
    uv run server.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

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

    @app.get("/")
    async def index():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

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

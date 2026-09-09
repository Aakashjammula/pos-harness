"""
FastAPI websocket server — see docs/superpowers/specs/
2026-09-09-transport-provider-harness-design.md (transport) and
2026-09-10-web-ui-provider-selection-design.md (model/provider
selection, this file's /options + query-param handling).

Two endpoints:
  GET  /options   what the UI can offer before connecting (TTS engines/
                  voices, LLM models currently loaded in LM Studio)
  WS   /ws        binary frames carry raw PCM16 mono audio in both
                  directions; one JSON "ready" event on connect, then
                  "user_text"/"bot_text"/"interrupted" events for live
                  captions. Config is chosen via query params, e.g.
                  /ws?tts=kokoro&voice=af_bella&llm_model=lfm2.5-230m&trigger_word=computer
                  — all optional, falling back to each engine's own
                  default.

Each connection gets its own Agent (own VAD state, own conversation
history), but STT and same-(engine,voice)/same-model TTS/LLM instances
are shared across every session that requests them — those hold no
per-session state, so sharing is what keeps memory flat regardless of
concurrent session count (see agent.py's LlmBase.stream()).

Usage:
    uv run server.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

import requests
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


def _default_llm_models(base_url: str, fallback: str) -> list[str]:
    """LM Studio's own loaded-model list, for the /options dropdown —
    advisory only. Falls back to just the configured default rather than
    failing the whole endpoint if LM Studio isn't reachable right now."""
    try:
        resp = requests.get(f"{base_url}/models", timeout=3)
        resp.raise_for_status()
        ids = [m["id"] for m in resp.json().get("data", [])]
        return ids or [fallback]
    except Exception:
        return [fallback]


def create_app(
    stt: SttBase,
    tts_engines: dict[str, type[TtsBase]] | None = None,
    llm_factory: Callable[[str], LlmBase] | None = None,
    vad_factory: Callable[[], VadBase] = _default_vad_factory,
    default_tts_engine: str = "kokoro",
    default_llm_model: str = "lfm2.5-230m",
    llm_base_url: str = "http://localhost:1234/v1",
) -> FastAPI:
    if tts_engines is None:
        from asr_test.tts import KokoroTts, SupertonicTts

        tts_engines = {"kokoro": KokoroTts, "supertonic": SupertonicTts}

    if llm_factory is None:
        from asr_test.llm import OpenAiCompatibleLlm

        llm_factory = lambda model: OpenAiCompatibleLlm(model=model)  # noqa: E731

    app = FastAPI()

    # Eagerly warm the default (engine, voice)/model combo at startup —
    # before uvicorn ever accepts a connection — so the *first* client
    # to connect with default settings doesn't pay TTS/LLM construction
    # and warm-up cost (several seconds, see each engine's own warmup
    # timing) as part of their own connection setup. Any other combo a
    # client explicitly asks for still loads lazily on first request.
    _tts_cache: dict[tuple[str, str], TtsBase] = {
        (default_tts_engine, ""): tts_engines[default_tts_engine]()
    }
    _llm_cache: dict[str, LlmBase] = {default_llm_model: llm_factory(default_llm_model)}
    _cache_lock = asyncio.Lock()

    async def get_tts(engine: str, voice: str | None) -> TtsBase:
        cls = tts_engines.get(engine, tts_engines[default_tts_engine])
        key = (engine, voice or "")
        async with _cache_lock:
            if key not in _tts_cache:
                loop = asyncio.get_running_loop()
                kwargs = {"voice": voice} if voice else {}
                _tts_cache[key] = await loop.run_in_executor(None, lambda: cls(**kwargs))
        return _tts_cache[key]

    async def get_llm(model: str) -> LlmBase:
        async with _cache_lock:
            if model not in _llm_cache:
                loop = asyncio.get_running_loop()
                _llm_cache[model] = await loop.run_in_executor(None, llm_factory, model)
        return _llm_cache[model]

    @app.get("/")
    async def index():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/options")
    async def options():
        loop = asyncio.get_running_loop()
        llm_models = await loop.run_in_executor(
            None, _default_llm_models, llm_base_url, default_llm_model
        )
        return {
            "tts": {name: cls.list_voices() for name, cls in tts_engines.items()},
            "llm_models": llm_models,
            "defaults": {"tts_engine": default_tts_engine, "llm_model": default_llm_model},
        }

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        loop = asyncio.get_running_loop()

        params = websocket.query_params
        tts_engine = params.get("tts", default_tts_engine)
        voice = params.get("voice") or None
        llm_model = params.get("llm_model", default_llm_model)
        trigger_word = params.get("trigger_word") or None

        tts = await get_tts(tts_engine, voice)
        llm = await get_llm(llm_model)

        await websocket.send_json({
            "event": "ready",
            "input_sample_rate": config.MIC_RATE,
            "output_sample_rate": tts.sample_rate,
            "tts_engine": tts_engine,
            "llm_model": llm_model,
        })

        def emit(name: str, data: dict) -> None:
            async def _send():
                try:
                    await websocket.send_json({"event": name, **data})
                except Exception:
                    pass  # socket already closing/closed — nothing to deliver to

            asyncio.run_coroutine_threadsafe(_send(), loop)

        sink = WebSocketAudioSink(websocket, loop, rate=tts.sample_rate, blocksize=config.OUT_BLOCK)
        agent = Agent(
            vad=vad_factory(), stt=stt, tts=tts, llm=llm,
            trigger_word=trigger_word, audio_sink=sink, on_event=emit,
        )
        threads = agent.start()
        try:
            while True:
                data = await websocket.receive_bytes()
                agent.feed_audio(pcm16_to_float32(data))
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"  session error: {e}")
        finally:
            print("  session closed — shutting down this connection's Agent")
            # Runs off the event loop: a turn in flight (blocking STT/LLM
            # call) can't be interrupted, so shutdown() may block for a
            # while waiting on it — must not stall other sessions' I/O.
            await loop.run_in_executor(None, agent.shutdown, threads)

    return app


if __name__ == "__main__":
    import uvicorn

    from asr_test.stt import OnnxAsrEngine

    config.ECHO_MODE = "duck"  # server can't verify a remote client has real headphone isolation

    app = create_app(stt=OnnxAsrEngine())
    uvicorn.run(app, host="0.0.0.0", port=8000)

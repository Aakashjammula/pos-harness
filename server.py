"""
FastAPI websocket server — see docs/superpowers/specs/
2026-09-09-transport-provider-harness-design.md (transport) and
2026-09-10-web-ui-provider-selection-design.md (model/provider
selection, this file's /options + query-param handling).

Four endpoints:
  GET  /options          what the UI can offer before connecting (TTS
                         engines/voices, LLM models currently loaded in
                         LM Studio)
  GET  /sessions         list of past sessions (id, mode, turn count, ...),
                         newest first — read-only, see storage.py
  GET  /sessions/{id}    one session's stored turns, 404 if unknown
  WS   /ws               binary frames carry raw PCM16 mono audio in both
                         directions (voice mode only — see mode below); one
                         JSON "ready" event on connect (includes a
                         session_id), then "user_text"/"bot_text"/"interrupted"
                         events for live captions. Config is chosen via
                         query params, e.g.
                         /ws?tts=kokoro&voice=af_bella&llm_model=lfm2.5-230m&trigger_word=computer&vad_threshold=0.5&vad_min_silence_ms=1200&vad_speech_pad_ms=300&mode=voice
                         — all optional, falling back to each engine's own
                         default. Headphones are assumed unconditionally (no
                         echo suppression, full barge-in) — every client is
                         expected to have real mic/speaker isolation; there's
                         no safer fallback mode. An invalid vad_* value gets
                         an "error" event and the socket is closed rather
                         than silently falling back.

                         `mode` is "voice" (default) or "text" — text mode
                         skips mic/VAD/STT/TTS entirely: the client sends
                         {"text": "..."} JSON messages instead of PCM audio,
                         and never receives binary audio frames back. An
                         invalid mode gets the same error+close treatment as
                         an invalid vad_* value.

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
import uuid
from collections.abc import Callable
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from asr_test import config
from asr_test.agent import Agent
from asr_test.audio.null_sink import NullAudioSink
from asr_test.audio.ws_sink import WebSocketAudioSink
from asr_test.interfaces import LlmBase, SttBase, TtsBase, VadBase
from asr_test.storage import SessionStore
from asr_test.utils import pcm16_to_float32
from asr_test.vad import SileroVad


def _default_vad_factory(
    threshold: float = 0.5,
    min_silence_ms: int = config.MIN_SILENCE_MS,
    speech_pad_ms: int = config.SPEECH_PAD_MS,
) -> VadBase:
    return SileroVad(
        sample_rate=config.MIC_RATE,
        threshold=threshold,
        min_silence_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
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
    vad_factory: Callable[..., VadBase] = _default_vad_factory,
    default_tts_engine: str = "kokoro",
    default_llm_model: str = "lfm2.5-230m",
    llm_base_url: str = "http://localhost:1234/v1",
    session_store: SessionStore | None = None,
) -> FastAPI:
    if tts_engines is None:
        from asr_test.tts import KokoroTts, SupertonicTts

        tts_engines = {"kokoro": KokoroTts, "supertonic": SupertonicTts}

    if llm_factory is None:
        from asr_test.llm import LangChainLlm

        llm_factory = lambda model: LangChainLlm(model=model)  # noqa: E731

    app = FastAPI()
    store = session_store or SessionStore(config.SESSIONS_DB_PATH)

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

    @app.get("/sessions")
    async def list_sessions():
        return store.list_sessions()

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        result = store.get_session(session_id)
        if result is None:
            raise HTTPException(status_code=404, detail="session not found")
        return result

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        loop = asyncio.get_running_loop()

        params = websocket.query_params
        tts_engine = params.get("tts", default_tts_engine)
        voice = params.get("voice") or None
        llm_model = params.get("llm_model", default_llm_model)
        trigger_word = params.get("trigger_word") or None

        mode = params.get("mode", "voice")
        if mode not in ("voice", "text"):
            await websocket.send_json({
                "event": "error",
                "message": f"mode must be 'voice' or 'text', got {mode!r}",
            })
            await websocket.close(code=1008)
            return

        try:
            vad_threshold = float(params.get("vad_threshold", 0.5))
            vad_min_silence_ms = int(params.get("vad_min_silence_ms", config.MIN_SILENCE_MS))
            vad_speech_pad_ms = int(params.get("vad_speech_pad_ms", config.SPEECH_PAD_MS))
        except ValueError as e:
            await websocket.send_json({"event": "error", "message": f"invalid vad_* value: {e}"})
            await websocket.close(code=1008)
            return
        if not (0.0 <= vad_threshold <= 1.0):
            await websocket.send_json({
                "event": "error",
                "message": f"vad_threshold must be in [0, 1], got {vad_threshold}",
            })
            await websocket.close(code=1008)
            return
        if vad_min_silence_ms < 0 or vad_speech_pad_ms < 0:
            await websocket.send_json({
                "event": "error",
                "message": "vad_min_silence_ms/vad_speech_pad_ms must be non-negative",
            })
            await websocket.close(code=1008)
            return

        tts = await get_tts(tts_engine, voice)
        llm = await get_llm(llm_model)

        session_id = uuid.uuid4().hex
        try:
            store.create_session(session_id, mode=mode, tts_engine=tts_engine, llm_model=llm_model)
        except Exception as e:
            print(f"  session store error (create_session): {e}")

        await websocket.send_json({
            "event": "ready",
            "session_id": session_id,
            "input_sample_rate": config.MIC_RATE,
            "output_sample_rate": tts.sample_rate,
            "tts_engine": tts_engine,
            "llm_model": llm_model,
        })

        def emit(name: str, data: dict) -> None:
            if name == "user_text":
                try:
                    store.add_turn(session_id, "user", data["text"])
                except Exception as e:
                    print(f"  session store error (add_turn user): {e}")
            elif name == "bot_text":
                try:
                    store.add_turn(session_id, "assistant", data["text"], data.get("usage"))
                except Exception as e:
                    print(f"  session store error (add_turn assistant): {e}")

            async def _send():
                try:
                    await websocket.send_json({"event": name, **data})
                except Exception:
                    pass  # socket already closing/closed — nothing to deliver to

            asyncio.run_coroutine_threadsafe(_send(), loop)

        vad = vad_factory(
            threshold=vad_threshold,
            min_silence_ms=vad_min_silence_ms,
            speech_pad_ms=vad_speech_pad_ms,
        )
        sink = (
            NullAudioSink()
            if mode == "text"
            else WebSocketAudioSink(websocket, loop, rate=tts.sample_rate, blocksize=config.OUT_BLOCK)
        )
        agent = Agent(
            vad=vad, stt=stt, tts=tts, llm=llm,
            trigger_word=trigger_word, audio_sink=sink, on_event=emit,
            text_only=(mode == "text"),
        )
        threads = agent.start()
        try:
            if mode == "text":
                while True:
                    msg = await websocket.receive_json()
                    text = msg.get("text", "")
                    if text.strip():
                        await loop.run_in_executor(None, agent.on_text_message, text)
            else:
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

    app = create_app(stt=OnnxAsrEngine())
    uvicorn.run(app, host="0.0.0.0", port=8000)

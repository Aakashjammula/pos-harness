"""
FastAPI server: text chat over server-sent events, voice over a websocket.
See docs/superpowers/specs/2026-09-09-transport-provider-harness-design.md
(transport) and 2026-09-10-web-ui-provider-selection-design.md (model/provider
selection, this file's /options handling).

Endpoints (plus /auth/*, /credentials/* and /tools -- see pos.auth.routes):
  GET  /options          what the UI can offer before connecting (TTS
                         engines/voices, the active provider, and which tools
                         would be bound server-wide)
  GET  /sessions         list of past sessions (id, mode, turn count, ...),
                         newest first — read-only, see storage.py
  GET  /sessions/{id}    one session's stored turns, 404 if unknown
  POST /chat/stream      text chat. Body {message, session_id?, provider?,
                         llm_model?}; the reply streams back as server-sent
                         events (`session`, `token`, `ping`, `done`, `title`,
                         `error`) from the same LlmBase.stream() voice uses
                         (see pos.turn.run_turn). Problems found before
                         streaming starts are plain HTTP statuses (401, 404,
                         409 no LLM configured, 422 empty message).
  WS   /ws               voice only: binary frames carry raw PCM16 mono audio
                         in both directions; one JSON "ready" event on connect
                         (includes a session_id), then
                         "user_text"/"bot_text"/"interrupted" events for live
                         captions. Config is chosen via query params, e.g.
                         /ws?tts=kokoro&voice=af_bella&llm_model=<id>&vad_threshold=0.5
                         — all optional, falling back to each engine's own
                         default. Headphones are assumed unconditionally (no
                         echo suppression, full barge-in). An invalid vad_*
                         value gets an "error" event and the socket is closed
                         rather than silently falling back. `mode=text` is
                         refused with a pointer to /chat/stream.

                         `voice_input_mode` is "vad" (default), "wake_word",
                         or "push_to_talk". "wake_word" requires trigger_word
                         to also be set (rejected otherwise) — same underlying
                         continuous VAD as "vad", just gated on the transcript
                         leading with the trigger phrase. "push_to_talk" skips
                         VAD entirely: the client sends {"event":"ptt_start"}
                         before streaming frames and {"event":"ptt_stop"}
                         after the last one, and any trigger_word sent
                         alongside it is ignored. Switching between voice and
                         text mid-conversation reuses the same session_id
                         (resume_session_id here, session_id on /chat/stream).

Both transports take a `provider` (optional) selecting which stored, per-user
encrypted credential to load -- see pos.auth.routes.PROVIDER_FIELDS and
pos.auth.store.UserStore. A user with stored credentials or tool choices never
uses the shared per-model LLM cache -- they get their own uncached LangChainLlm
with their own tools bound, since two sessions sharing one model name must never
share one session's key.

Every /ws connection, /chat/stream call and /sessions* request requires a signed-in
user (see pos.auth.deps) -- there is no anonymous mode.

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
import json
import os
import threading
import traceback
import uuid
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager

import requests
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from pos import config
from pos.agent import Agent
from pos.audio.ws_sink import WebSocketAudioSink
from pos.auth.deps import Auth
from pos.auth.routes import build_auth_router
from pos.auth.store import UserStore
from pos.db import create_pool, init_schema
from pos.http_security import OriginCheckMiddleware, SecurityHeadersMiddleware
from pos.interfaces import LlmBase, SttBase, TtsBase, VadBase
from pos.llm.providers import is_configured, resolve_provider
from pos.null_engines import NullVad
from pos.storage import SessionStore
from pos.tools import all_tools, resolve_enabled, tool_status
from pos.turn import run_turn
from pos.utils import pcm16_to_float32
from pos.vad import SileroVad


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


def _default_llm_models(base_url: str | None, fallback: str | None) -> list[str]:
    """LM Studio's own loaded-model list, for the /options dropdown —
    advisory only. Falls back to just the configured default rather than
    failing the whole endpoint if LM Studio isn't reachable right now.
    With no base_url configured there is nothing to ask, so no request is
    made at all."""
    fallback_list = [fallback] if fallback else []
    if not base_url:
        return fallback_list
    try:
        resp = requests.get(f"{base_url}/models", timeout=3, allow_redirects=False)
        resp.raise_for_status()
        ids = [m["id"] for m in resp.json().get("data", [])]
        return ids or fallback_list
    except Exception:
        return fallback_list


class ChatBody(BaseModel):
    message: str
    session_id: str | None = None   # continue this session; None starts a new one
    provider: str | None = None     # which stored credential to use (default: local)
    llm_model: str | None = None    # None: the provider's own default


def _sse(event: str, data: dict) -> str:
    """One server-sent event. Like Anthropic's stream, the name is on the
    `event:` line and the payload is JSON on `data:`."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


class _LazyStt(SttBase):
    """Defers building the real STT engine until load() -- which
    create_app's background warm-up calls -- so that constructing the app
    (and opening the port) doesn't wait on a multi-hundred-MB model."""

    def __init__(self, factory: Callable[[], SttBase]):
        self._factory = factory
        self._engine: SttBase | None = None

    def load(self) -> None:
        self._engine = self._factory()

    def __call__(self, audio):
        if self._engine is None:
            raise RuntimeError("STT model not loaded yet")
        return self._engine(audio)


def create_app(
    stt: SttBase,
    tts_engines: dict[str, type[TtsBase]] | None = None,
    llm_factory: Callable[[str | None], LlmBase] | None = None,
    llm_env_factory: Callable[[str | None, Mapping[str, str], list[str]], LlmBase] | None = None,
    vad_factory: Callable[..., VadBase] = _default_vad_factory,
    default_tts_engine: str = "kokoro",
    default_llm_model: str | None = None,   # no model is assumed: each provider uses its own default
    llm_base_url: str | None = None,
    session_store: SessionStore | None = None,
    user_store: UserStore | None = None,
    warm_in_background: bool = False,
) -> FastAPI:
    """warm_in_background=False (default) loads STT + the default TTS
    before returning, so a caller that only wants a ready app gets one.
    True returns at once and loads them on a startup thread instead: auth,
    /options and text-mode chat work immediately, and only a voice-mode
    connection waits (it is told so, rather than hanging)."""
    if tts_engines is None:
        from pos.tts import KokoroTts, SupertonicTts

        tts_engines = {"kokoro": KokoroTts, "supertonic": SupertonicTts}

    uses_real_llm = llm_factory is None
    if llm_factory is None or llm_env_factory is None:
        from pos.llm import LangChainLlm

        if llm_factory is None:
            llm_factory = lambda model: LangChainLlm(model=model)  # noqa: E731
        if llm_env_factory is None:
            # Used only for a connection with stored per-user credentials or
            # tool choices -- deliberately bypasses the shared _llm_cache below,
            # since two sessions sharing a model name must never share one
            # session's personal key. No warm-up: for a cloud provider it is a
            # paid call that once delayed a connect by 89s, and it only ever
            # helped a local model that was still loading.
            llm_env_factory = lambda model, env, enabled: LangChainLlm(  # noqa: E731
                model=model, env=env, enabled_tools=enabled, warmup=False
            )

    if session_store is None or user_store is None:
        pool = create_pool()
        init_schema(pool)
        session_store = session_store or SessionStore(pool)
        user_store = user_store or UserStore(pool)
    store = session_store
    users = user_store
    auth = Auth(users)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if warm_in_background:
            threading.Thread(target=_warm_in_background, name="model-warmup", daemon=True).start()
        yield

    docs = config.ENABLE_API_DOCS   # Swagger UI and the schema are development aids: off unless asked for
    app = FastAPI(
        lifespan=lifespan,
        docs_url="/docs" if docs else None,
        redoc_url="/redoc" if docs else None,
        openapi_url="/openapi.json" if docs else None,
    )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, exc: RequestValidationError):
        # FastAPI's default body echoes every rejected value back (`input`), which would
        # send a mistyped password straight into logs and proxies. Say where and why only.
        return JSONResponse(
            status_code=422,
            content={"detail": [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]},
        )

    app.add_middleware(OriginCheckMiddleware, allowed_origins=config.CORS_ORIGINS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(SecurityHeadersMiddleware, hsts=config.COOKIE_SECURE)
    app.include_router(build_auth_router(users, auth))

    # Warm the default STT model and TTS engine/voice ahead of the first
    # client, so nobody pays their construction cost (seconds to minutes on
    # a cold cache) as part of their own connection setup. Any other
    # (engine, voice) combo a client explicitly asks for loads lazily.
    #
    # The LLM is deliberately NOT warmed: no endpoint is assumed. It is
    # built on the first connection that has one (a user's saved credential
    # or a server-level env var) and never contacted before that.
    _tts_cache: dict[tuple[str, str], TtsBase] = {}
    _llm_cache: dict[str | None, LlmBase] = {}
    _models_ready = threading.Event()
    _warm_errors: list[str] = []

    def _warm_models() -> None:
        load = getattr(stt, "load", None)   # a lazy STT wrapper loads here, not at construction
        if load is not None:
            load()
        _tts_cache[(default_tts_engine, "")] = tts_engines[default_tts_engine]()
        _models_ready.set()

    def _warm_in_background() -> None:
        try:
            _warm_models()
        except Exception as e:
            _warm_errors.append(str(e))
            traceback.print_exc()

    if not warm_in_background:
        _warm_models()
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

    async def get_llm(model: str | None) -> LlmBase:
        async with _cache_lock:
            if model not in _llm_cache:
                loop = asyncio.get_running_loop()
                _llm_cache[model] = await loop.run_in_executor(None, llm_factory, model)
        return _llm_cache[model]

    def _llm_configured(env: Mapping[str, str]) -> bool:
        """True once some provider has real settings (a local server URL or
        a provider API key). Nothing is assumed, so with none of them no
        request may be sent anywhere. Fakes injected via llm_factory (tests,
        embedding) bring their own backend, so they are always configured."""
        return True if not uses_real_llm else is_configured(env)

    class _NoLlmConfigured(Exception):
        pass

    async def _resolve_user_llm(user_id: str, provider: str, model: str | None) -> LlmBase:
        """The LLM for one user's connection, with their credentials and tool
        choices applied. Anything user-specific gets a private instance;
        otherwise the shared, warmed one for that model is reused."""
        loop = asyncio.get_running_loop()
        stored = users.get_credential(user_id, provider)
        overrides = dict(stored or {})
        for spec in all_tools():   # each key-requiring tool's own saved key
            if spec.credential_provider:
                overrides.update(users.get_credential(user_id, spec.credential_provider) or {})
        if not _llm_configured({**os.environ, **overrides}):
            raise _NoLlmConfigured
        tool_settings = users.get_tool_settings(user_id)
        if overrides or tool_settings:
            merged = {**os.environ, **overrides}
            enabled = resolve_enabled(tool_settings, merged)
            return await loop.run_in_executor(None, llm_env_factory, model, merged, enabled)
        return await get_llm(model)

    @app.get("/options")
    async def options():
        loop = asyncio.get_running_loop()
        llm_models = await loop.run_in_executor(
            None, _default_llm_models, llm_base_url, default_llm_model
        )
        llm_configured = _llm_configured(os.environ)
        provider = {"name": "", "model": ""}   # nothing configured server-wide: don't invent one
        if llm_configured:
            try:
                resolved = resolve_provider()
                provider = {"name": resolved.name, "model": resolved.model}
            except RuntimeError:
                pass   # e.g. Azure key without an endpoint -- surfaces when a session connects
        return {
            "tts": {name: cls.list_voices() for name, cls in tts_engines.items()},
            "llm_models": llm_models,
            "defaults": {"tts_engine": default_tts_engine, "llm_model": default_llm_model or ""},
            "provider": provider,
            "llm_configured": llm_configured,
            "tools": tool_status(),
        }

    @app.get("/sessions")
    async def list_sessions(user_id: str = Depends(auth.require_user_id)):
        return store.list_sessions(user_id)

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str, user_id: str = Depends(auth.require_user_id)):
        result = store.get_session(session_id, user_id)
        if result is None:
            raise HTTPException(status_code=404, detail="session not found")
        return result

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str, user_id: str = Depends(auth.require_user_id)):
        if not store.delete_session(session_id, user_id):
            raise HTTPException(status_code=404, detail="session not found")
        return {"deleted": True}

    @app.post("/chat/stream")
    async def chat_stream(body: ChatBody, user_id: str = Depends(auth.require_user_id)):
        """Text chat as a server-sent event stream, over the same LlmBase.stream()
        that voice mode uses (see pos.turn.run_turn).

        Problems found before streaming starts are plain HTTP statuses (401, 404,
        409, 422). Once the 200 is sent a status can't change, so a failure
        mid-stream is an `error` event instead. Events: `session` {id}, `token`
        {text}, `ping` {} (keepalive), `done` {text, usage?, latency?}, `title`
        {title}, `error` {message}."""
        text = body.message.strip()
        if not text:
            raise HTTPException(status_code=422, detail="message is empty")
        history: list[dict] = []
        if body.session_id:
            existing = store.get_session(body.session_id, user_id)
            if existing is None:
                raise HTTPException(status_code=404, detail="session not found")
            history = [{"role": t["role"], "content": t["text"]} for t in existing["turns"]]
            if existing["session"]["mode"] != "text":
                # Same rule as resuming over /ws: the stored mode is whichever
                # the session was last used in.
                store.set_mode(body.session_id, "text")
        try:
            llm = await _resolve_user_llm(user_id, body.provider or "local", body.llm_model or default_llm_model)
        except _NoLlmConfigured:
            raise HTTPException(
                status_code=409, detail="no LLM configured -- add a provider (e.g. your local server URL) in Settings"
            ) from None

        loop = asyncio.get_running_loop()
        used_model = getattr(getattr(llm, "provider", None), "model", None) or body.llm_model or ""
        new_session = body.session_id is None
        session_id = body.session_id or uuid.uuid4().hex
        if new_session:
            store.create_session(session_id, user_id, mode="text", tts_engine=None, llm_model=used_model)
        store.add_turn(session_id, "user", text)
        messages = history[-config.HISTORY_TURNS * 2 :] + [{"role": "user", "content": text}]

        async def events():
            cancel = threading.Event()
            queue: asyncio.Queue = asyncio.Queue()

            def push(item) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, item)

            def on_piece(piece: str) -> bool:
                push(("token", {"text": piece}))
                return not cancel.is_set()

            def work() -> None:
                try:
                    push(("stats", run_turn(llm, messages, cancel, on_piece)))
                except Exception as e:  # noqa: BLE001 -- reported to the client as an error event
                    print(f"  chat stream error: {e}")
                    push(("error", {"message": str(e)[:300] or type(e).__name__}))
                finally:
                    push(None)

            worker = loop.run_in_executor(None, work)
            try:
                yield _sse("session", {"id": session_id})
                while True:
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=15)
                    except TimeoutError:
                        yield _sse("ping", {})   # keeps proxies from closing a stream that looks idle
                        continue
                    if item is None:
                        break
                    kind, payload = item
                    if kind != "stats":
                        yield _sse(kind, payload)
                        continue
                    if not payload.text:
                        continue
                    done: dict = {"text": payload.text}
                    if payload.usage:
                        done["usage"] = payload.usage
                    if payload.ttft is not None:
                        done["latency"] = {"ttft": round(payload.ttft, 3), "total": round(payload.total, 3)}
                    store.add_turn(session_id, "assistant", payload.text, payload.usage or None)
                    yield _sse("done", done)
                    generate_title = getattr(llm, "generate_title", None)
                    if new_session and generate_title is not None:
                        title = await loop.run_in_executor(None, generate_title, text, payload.text)
                        if title:
                            store.set_title(session_id, title)
                            yield _sse("title", {"title": title})
            finally:
                cancel.set()   # client gone (or finished): stop generating tokens nobody will read
                await worker

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        loop = asyncio.get_running_loop()

        user_id = auth.user_id_from_request(websocket)
        if user_id is None:
            await websocket.send_json({"event": "error", "message": "not authenticated"})
            await websocket.close(code=1008)
            return

        params = websocket.query_params
        tts_engine = params.get("tts", default_tts_engine)
        voice = params.get("voice") or None
        llm_model = params.get("llm_model") or default_llm_model   # None -> the provider's own default
        trigger_word = params.get("trigger_word") or None

        mode = params.get("mode", "voice")
        if mode != "voice":
            # Text chat moved to POST /chat/stream (server-sent events); this
            # socket is only for voice, which needs bidirectional binary audio.
            await websocket.send_json({
                "event": "error",
                "message": (
                    "text chat moved to POST /chat/stream"
                    if mode == "text"
                    else f"mode must be 'voice', got {mode!r}"
                ),
            })
            await websocket.close(code=1008)
            return

        voice_input_mode = params.get("voice_input_mode", "vad")
        if voice_input_mode not in ("vad", "wake_word", "push_to_talk"):
            await websocket.send_json({
                "event": "error",
                "message": f"voice_input_mode must be 'vad', 'wake_word', or 'push_to_talk', got {voice_input_mode!r}",
            })
            await websocket.close(code=1008)
            return
        if voice_input_mode == "wake_word" and not trigger_word:
            await websocket.send_json({
                "event": "error",
                "message": "trigger_word is required when voice_input_mode is 'wake_word'",
            })
            await websocket.close(code=1008)
            return
        # push_to_talk has already explicitly signaled speech by holding
        # the control -- a wake phrase would be redundant, so any
        # trigger_word sent alongside it is ignored rather than honored.
        if voice_input_mode != "wake_word":
            trigger_word = None

        # Resuming a prior session: seed the new Agent's conversation from
        # its stored turns and keep writing further turns under the same
        # session_id, instead of starting a fresh one.
        resume_session_id = params.get("resume_session_id") or None
        resumed_conversation = None
        if resume_session_id:
            existing = store.get_session(resume_session_id, user_id)
            if existing is None:
                await websocket.send_json({
                    "event": "error",
                    "message": f"no session found with id {resume_session_id!r}",
                })
                await websocket.close(code=1008)
                return
            resumed_conversation = [
                {"role": turn["role"], "content": turn["text"]} for turn in existing["turns"]
            ]

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

        if not _models_ready.is_set():
            # Voice needs the STT and TTS models; text chat (POST /chat/stream)
            # needs neither, so it works while these are still loading.
            await websocket.send_json({
                "event": "error",
                "message": (
                    f"speech models failed to load: {_warm_errors[0]}"
                    if _warm_errors
                    else "the server is still loading its speech models -- try again in a minute"
                ),
            })
            await websocket.close(code=1013)
            return
        tts = await get_tts(tts_engine, voice)
        try:
            llm = await _resolve_user_llm(user_id, params.get("provider", "local"), llm_model)
        except _NoLlmConfigured:
            await websocket.send_json({
                "event": "error",
                "message": "no LLM configured -- add a provider (e.g. your local server URL) in Settings",
            })
            await websocket.close(code=1008)
            return
        # What actually answers: the provider's resolved model when the client
        # named none (fakes without a .provider fall back to what was asked).
        used_model = getattr(getattr(llm, "provider", None), "model", None) or llm_model or ""

        session_id = resume_session_id or uuid.uuid4().hex
        stored_tts_engine = tts_engine
        if not resume_session_id:
            try:
                store.create_session(
                    session_id, user_id, mode=mode,
                    tts_engine=stored_tts_engine, llm_model=used_model,
                )
            except Exception as e:
                print(f"  session store error (create_session): {e}")
        elif existing["session"]["mode"] != mode:
            # A resumed session's stored mode reflects whichever mode it
            # was most recently used in -- see the mid-session voice/text
            # switch flow in the browser UI, which resumes the same
            # session_id under a different mode.
            try:
                store.set_mode(session_id, mode)
            except Exception as e:
                print(f"  session store error (set_mode): {e}")

        await websocket.send_json({
            "event": "ready",
            "session_id": session_id,
            "resumed": resume_session_id is not None,
            "input_sample_rate": config.MIC_RATE,
            "output_sample_rate": tts.sample_rate,
            "tts_engine": stored_tts_engine,
            "llm_model": used_model,
        })

        # Title generation (like ChatGPT's own "name the chat after the
        # first exchange"): fires once, after the first assistant reply
        # of a brand-new session -- never for a resumed one, since it
        # already has (or already had the chance to get) a title from
        # its original run, and re-titling from a continuation message
        # would describe the wrong exchange.
        last_user_text: str | None = None
        title_pending = resume_session_id is None

        async def _generate_and_store_title(user_text: str, bot_text: str) -> None:
            generate_title = getattr(llm, "generate_title", None)
            if generate_title is None:
                return
            try:
                title = await loop.run_in_executor(None, generate_title, user_text, bot_text)
            except Exception as e:
                print(f"  title generation failed: {e}")
                return
            if title:
                try:
                    store.set_title(session_id, title)
                except Exception as e:
                    print(f"  session store error (set_title): {e}")

        def emit(name: str, data: dict) -> None:
            nonlocal last_user_text, title_pending
            if name == "user_text":
                last_user_text = data["text"]
                try:
                    store.add_turn(session_id, "user", data["text"])
                except Exception as e:
                    print(f"  session store error (add_turn user): {e}")
            elif name == "bot_text":
                try:
                    store.add_turn(session_id, "assistant", data["text"], data.get("usage"))
                except Exception as e:
                    print(f"  session store error (add_turn assistant): {e}")
                if title_pending and last_user_text is not None:
                    title_pending = False
                    asyncio.run_coroutine_threadsafe(
                        _generate_and_store_title(last_user_text, data["text"]), loop
                    )

            async def _send():
                try:
                    await websocket.send_json({"event": name, **data})
                except Exception:
                    pass  # socket already closing/closed — nothing to deliver to

            asyncio.run_coroutine_threadsafe(_send(), loop)

        vad = (
            NullVad()
            if voice_input_mode == "push_to_talk"
            else vad_factory(
                threshold=vad_threshold,
                min_silence_ms=vad_min_silence_ms,
                speech_pad_ms=vad_speech_pad_ms,
            )
        )
        sink = WebSocketAudioSink(websocket, loop, rate=tts.sample_rate, blocksize=config.OUT_BLOCK)
        agent = Agent(
            vad=vad, stt=stt, tts=tts, llm=llm,
            trigger_word=trigger_word, audio_sink=sink, on_event=emit,
            conversation=resumed_conversation,
        )
        threads = agent.start()
        try:
            if voice_input_mode == "push_to_talk":
                # Unlike vad/wake_word (audio frames only), push-to-talk
                # also needs small JSON control messages on the same
                # connection -- receive_bytes()/receive_json() each raise
                # if the other shape arrives, so this reads the raw ASGI
                # message and dispatches on whichever key is present.
                while True:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        raise WebSocketDisconnect()
                    if message.get("bytes") is not None:
                        agent.feed_ptt_frame(pcm16_to_float32(message["bytes"]))
                    elif message.get("text") is not None:
                        control = json.loads(message["text"])
                        event = control.get("event")
                        if event == "ptt_start":
                            agent.ptt_start()
                        elif event == "ptt_stop":
                            agent.ptt_stop()
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


def run() -> None:
    import sys

    import uvicorn

    from pos.settings_check import check_settings
    from pos.stt import OnnxAsrEngine

    errors, warnings = check_settings(os.environ)
    for w in warnings:
        print(f"  WARNING: {w}")
    if errors:
        for e in errors:
            print(f"  CONFIG ERROR: {e}")
        sys.exit(1)

    llm_base_url = os.environ.get("LOCAL_BASE_URL") or None   # no default: see create_app()
    # Nothing heavy loads before the port opens: the STT model and default
    # TTS load on a background thread (see create_app's warm_in_background).
    app = create_app(stt=_LazyStt(OnnxAsrEngine), llm_base_url=llm_base_url, warm_in_background=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()

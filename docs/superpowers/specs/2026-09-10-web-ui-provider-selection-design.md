# Web UI redesign + model/provider selection

Date: 2026-09-10
Status: draft, awaiting review

## Context

`static/index.html` (from the Phase 1 transport plan) is functionally
complete but visually a placeholder — two unstyled buttons and a raw
JSON log on a blank page. Separately, `server.py` currently loads one
fixed `stt`/`tts`/`llm` instance for the whole process at startup
(`create_app(stt=OnnxAsrEngine(), tts=KokoroTts(), llm=OpenAiCompatibleLlm())`)
and shares it across every session — there is no way for a client to
choose anything.

This spec covers redesigning the HTML client properly and adding local
model/provider selection: which TTS engine + voice, and which LLM
model (from whatever's already loaded in LM Studio). Cloud providers
(OpenAI/Anthropic/etc.) are explicitly out of scope — those don't exist
in this codebase yet (Phase 2 of the earlier transport/provider spec)
and weren't requested here.

## Decisions from brainstorming

- Selectable: TTS engine (Kokoro/Supertonic) + voice, and LLM model
  (from LM Studio's `/v1/models`). STT stays fixed (only one
  implementation exists). No cloud providers.
- Live captions (You said / Bot replied) are in scope, driven by a new
  `Agent` event hook — not just audio + a status indicator.
- The trigger-word feature (`--trigger-word` CLI flag today) gets a UI
  text input too, using the same connect-time mechanism as model/provider
  selection.
- **Config is chosen at connect time via URL query params, not a
  first-message JSON handshake** (a handshake was sketched as a
  possibility in the original transport spec, but query params avoid
  the ordering requirement — client must send config before any audio —
  for no real benefit). Changing settings means disconnect + reconnect
  with new params; no live mid-session reconfiguration.

## Backend changes

### `GET /options` — what the UI can offer before connecting

Returns available choices so the frontend can populate dropdowns
without hardcoding them:

```json
{
  "tts": {
    "kokoro": ["af_bella", "af_sky", "..."],
    "supertonic": ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]
  },
  "llm_models": ["lfm2.5-230m", "google/gemma-3-270m", "..."]
}
```

- `tts.kokoro`: needs a new lightweight `KokoroTts.list_voices() ->
  list[str]` static/class method that downloads just `voices.json` (a
  small file, separate from the ~380MB `model.onnx`) and returns its
  keys — no need to construct a full `KokoroTts` instance (which loads
  the whole ONNX session) just to list voices.
- `tts.supertonic`: `SupertonicTts` already has a module-level
  `_VALID_VOICES` tuple — expose it as a public
  `SupertonicTts.VALID_VOICES` class attribute instead of leaving it
  private, since `server.py` now needs to read it from outside the
  module.
- `llm_models`: a `GET {base_url}/v1/models` call against LM Studio
  (using `requests`, already a dependency) — same endpoint verified
  reachable during Phase 1's manual smoke testing. Falls back to just
  `[DEFAULT_MODEL]` if LM Studio isn't reachable (don't fail the whole
  endpoint over it — this list is advisory for the dropdown, not a hard
  dependency).

### `/ws` query params

```
ws://host/ws?tts=kokoro&voice=af_bella&llm_model=lfm2.5-230m&trigger_word=computer
```

All optional — omitting them falls back to today's hardcoded defaults
(`KokoroTts`'s own `af_bella`, `OpenAiCompatibleLlm`'s own
`lfm2.5-230m`, no trigger word), so existing behavior for a client that
sends no params at all is unchanged.

### Server-side lazy, shared engine cache

Two module-level caches in `server.py`, guarded by one `asyncio.Lock`
(construction is a one-time cost per distinct combination — a lock
just prevents two simultaneous first-requesters from double-building
the same heavy instance):

```python
_tts_cache: dict[tuple[str, str], TtsBase] = {}
_llm_cache: dict[str, LlmBase] = {}
_cache_lock = asyncio.Lock()

async def get_tts(engine: str, voice: str | None) -> TtsBase:
    key = (engine, voice or "")
    async with _cache_lock:
        if key not in _tts_cache:
            cls = _TTS_ENGINES[engine]
            kwargs = {"voice": voice} if voice else {}
            _tts_cache[key] = await loop.run_in_executor(None, lambda: cls(**kwargs))
    return _tts_cache[key]
```

Constructing a `KokoroTts`/`SupertonicTts` costs real RAM (~1.5GB each,
per the README's own measurements) — this cache means a given
`(engine, voice)` combination is only ever loaded once, shared by every
session that requests it afterward, same resource story as Phase 1's
single shared default instance. Picking N distinct voices across a
session's lifetime costs N × ~1.5GB — an explicit tradeoff worth being
aware of on a personal/dev machine, not something this spec tries to
bound (no eviction policy — YAGNI for a local tool with a handful of
concurrent users).

`OpenAiCompatibleLlm` instances are cheap (just an HTTP client + a
model name string, no loaded weights) — cached the same way for
consistency, not because construction cost demands it.

Running construction via `run_in_executor` keeps one session's
first-time engine load from blocking other sessions' websocket
send/receive on the event loop (loading Kokoro takes ~1-2s per the
README's warm-up numbers).

### `Agent` event hook for captions

```python
# agent.py
def __init__(self, ..., on_event: Callable[[str, dict], None] | None = None):
    ...
    self._on_event = on_event or (lambda name, data: None)
```

Three call sites, all additive (no existing logic changes):

- `worker_thread`, right after STT succeeds and passes the
  length/trigger checks: `self._on_event("user_text", {"text": text})`
- `respond()`, where it already does `if full_response: self.conversation.append(...)`:
  also `self._on_event("bot_text", {"text": "".join(full_response)})`
- `interrupt()`: `self._on_event("interrupted", {})`

`server.py` wires this per-connection:

```python
def emit(name, data):
    asyncio.run_coroutine_threadsafe(websocket.send_json({"event": name, **data}), loop)

agent = Agent(..., on_event=emit)
```

Local mode (`main.py`) passes no `on_event` — the default no-op lambda
keeps it a pure no-op there, zero behavior change.

## Frontend

Redesigned `static/index.html` (visual execution follows proper design
guidance during implementation, not fully speced here — see below):

- **Config panel** (shown before connecting): TTS engine dropdown ->
  voice dropdown that repopulates based on engine choice, LLM model
  dropdown, trigger-word text input (optional), a Connect button.
  Populated from `GET /options` on page load.
- **Session view** (shown after connecting): a status indicator
  (idle / listening / bot speaking / error), a scrolling transcript
  pane showing `You: ...` / `Bot: ...` lines driven by the `user_text`/
  `bot_text` events, a Disconnect button.
- Audio capture/playback logic (Web Audio API, PCM16 framing) is
  unchanged from the existing implementation — only the config/status/
  transcript chrome around it is new.

## Testing

- `GET /options`: pytest with `TestClient`, monkeypatching
  `KokoroTts.list_voices`/the LM Studio `requests.get` call so no real
  network/model load happens in tests.
- Engine cache: pytest asserting two requests for the same `(engine,
  voice)` construct the underlying class only once (a `MagicMock`
  swapped in for the constructor, asserting `call_count == 1` after two
  `get_tts()` calls with the same key).
- `Agent` event hook: extend `tests/fakes.py`-based `test_agent.py` —
  assert `user_text`/`bot_text`/`interrupted` fire with the right
  payloads, using a list-appending fake `on_event` callable.
- Query-param wiring in `/ws`: extend `test_server.py`'s `TestClient`
  websocket tests to connect with `?tts=...&voice=...` and assert the
  right cached instance gets used (e.g. via a fake TTS class whose
  `__call__` tags output so the test can tell which instance ran).
- Visual UI: manual smoke test, same as Phase 1 (needs a real browser
  and a human).

## Risks / open questions

- Kokoro's actual voice list size is unknown until `list_voices()` is
  implemented and run against the real `voices.json` — could be a
  couple dozen entries, which is fine for a dropdown but worth knowing
  before assuming a particular UI treatment (plain `<select>` is fine
  either way).
- No eviction policy on the TTS cache means RAM only grows across a
  long-running server's lifetime as different voices get requested —
  acceptable for a personal/dev tool per this spec's own scope call,
  but would need revisiting before any real multi-tenant deployment.

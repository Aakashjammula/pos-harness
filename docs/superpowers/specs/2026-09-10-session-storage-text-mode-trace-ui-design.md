# Session persistence + text chat mode + token/trace UI

Date: 2026-09-10
Status: draft, awaiting review

## Context

This covers the remaining three sub-projects (B, C, D) from the
four-part plan started alongside sub-project A (LLM provider selection
+ usage/cost tracking — spec'd, planned, and implemented separately at
`docs/superpowers/specs/2026-09-10-llm-provider-cost-tracking-design.md`).
They are consolidated into one spec here (rather than three separate
documents) because they are small individually and tightly coupled:
C's turns are what B stores, and D displays both A's per-turn data and
B's stored history. Each still gets its own section below and can be
implemented/tested as an independent unit.

Today: `Agent` fires `on_event("user_text"/"bot_text"/"interrupted", ...)`
for live captions (`server.py` forwards these over the websocket;
`main.py` ignores them). Nothing is stored anywhere — a browser refresh
or process restart loses all history. The browser UI is voice-only (mic
required to connect at all). `bot_text`'s `usage` field (added by
sub-project A) is sent over the wire but nothing renders it.

## Decisions from brainstorming (this and the earlier session)

- Storage: SQLite (`sqlite3`, stdlib — no new dependency), one file at
  `config.SESSIONS_DB_PATH` (default `sessions.db` at the repo root).
- Storage goal: resume/history — a queryable, persistent record, with
  simple read endpoints so the UI can browse it. Not a write-only debug
  log.
- Text mode is a separate mode picked before connecting (not a text box
  that's always available alongside voice) — a text session skips
  mic/VAD/STT entirely.
- Text mode never plays audio (no TTS synthesis at all in that mode) —
  it's a normal text chat, not "type instead of speaking but still hear
  the reply."
- Trace UI shows tokens, cost, context size, and tool calls per turn
  (sub-project A already produces exactly this shape in `usage`).

## B — Session persistence

### Schema (`src/asr_test/storage.py`)

```sql
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,      -- ISO 8601 UTC
    mode TEXT NOT NULL,            -- "voice" | "text"
    tts_engine TEXT,               -- NULL for text-mode sessions
    llm_model TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,            -- "user" | "assistant"
    text TEXT NOT NULL,
    usage_json TEXT,               -- JSON-encoded usage dict, NULL if none
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
```

`SessionStore` wraps a single `sqlite3.Connection` opened with
`check_same_thread=False` (server.py's per-connection Agent threads
each call `add_turn` from their own worker thread) and a
`threading.Lock` around every write — SQLite serializes writers anyway,
but the lock avoids interleaved multi-statement transactions between
concurrent sessions. WAL journal mode (`PRAGMA journal_mode=WAL`) so
reads (`GET /sessions`) aren't blocked by an in-progress write.

```python
class SessionStore:
    def __init__(self, path: str = "sessions.db"): ...
    def create_session(self, session_id: str, mode: str, tts_engine: str | None, llm_model: str) -> None: ...
    def add_turn(self, session_id: str, role: str, text: str, usage: dict | None = None) -> None: ...
    def list_sessions(self, limit: int = 50) -> list[dict]: ...       # newest first
    def get_session(self, session_id: str) -> dict | None: ...         # {"session": {...}, "turns": [...]}
```

### Wiring into `server.py`

- `create_app(..., session_store: SessionStore | None = None)` —
  defaults to a real `SessionStore(config.SESSIONS_DB_PATH)` constructed
  once at app creation (like the existing TTS/LLM caches), so tests can
  inject an in-memory (`:memory:`) or fake store.
- `ws_endpoint`: generate `session_id = uuid.uuid4().hex` right after
  accepting the connection; call
  `session_store.create_session(session_id, mode, tts_engine, llm_model)`
  (`mode` from the new query param sub-project C adds — `"voice"` if
  absent). Add `"session_id": session_id` to the existing `ready` event
  payload.
- The existing `emit()` closure (which forwards `on_event` calls to the
  websocket) also writes to the store: on `"user_text"`, `add_turn(session_id, "user", data["text"])`;
  on `"bot_text"`, `add_turn(session_id, "assistant", data["text"], data.get("usage"))`.
  This keeps `Agent` itself completely unaware of storage — it already
  only ever calls `on_event`, and `server.py` owns what happens with
  that.
- Two new read-only endpoints:
  - `GET /sessions` → `session_store.list_sessions()`
  - `GET /sessions/{session_id}` → `session_store.get_session(session_id)`,
    404 if not found.

### Wiring into `main.py` (local CLI)

- Same store, same `config.SESSIONS_DB_PATH` file. `main.py` builds a
  `SessionStore` and passes `on_event=` into `Agent(...)` — a small
  closure that calls `add_turn` the same way `server.py`'s `emit()`
  does, using a single `session_id` generated once at startup (one
  local-mode run = one session, since there's no per-connection
  boundary in this mode). `create_session` is called once before
  `agent.run()`.

## C — Text chat mode

### `Agent` changes (`src/asr_test/agent.py`)

- New constructor param `text_only: bool = False`.
- `respond()`'s `enqueue()` (the inner function that currently does
  `self.tts_q.put(...)`) skips that `put` entirely when `self.text_only`
  is set — everything else in `respond()` (LLM streaming, `buf`/`drain`
  bookkeeping, `self.conversation` history, the `bot_text` event with
  `usage`) is unchanged. This is the only branch point; no new code path
  duplicates `respond()`.
- New method `on_text_message(self, text: str) -> None`: mirrors what
  `worker_thread()` does after STT succeeds — `self._on_event("user_text", {"text": text})`
  then `self.respond(text, self.new_turn(), stt_t=0.0)`. This is the
  entry point the websocket loop calls instead of `feed_audio()` for a
  text-mode session.
- `vad_thread`/`worker_thread`/`tts_thread` are still started (via the
  existing `Agent.start()`, unchanged) even for a text-only session —
  they simply sit idle (their queues never receive anything, since
  `feed_audio()` is never called and `enqueue()` never pushes to
  `tts_q`). Not worth special-casing `start()`/`shutdown()` to skip
  threads for what's 3 harmless idle polling loops.

### `server.py` changes

- `/ws` query param `mode` (`"voice"` default | `"text"`). Validated:
  any other value → the same `{"event": "error", ...}` + close(1008)
  pattern already used for invalid `vad_*` params.
- When `mode == "text"`: construct `Agent(..., text_only=True)`; the
  receive loop becomes:
  ```python
  while True:
      msg = await websocket.receive_json()
      text = msg.get("text", "")
      if text.strip():
          await loop.run_in_executor(None, agent.on_text_message, text)
  ```
  (`run_in_executor` because `on_text_message` -> `respond()` blocks on
  the synchronous LLM stream, same reasoning as `agent.shutdown` already
  running off the event loop elsewhere in this file.) When
  `mode == "voice"`: today's `receive_bytes()` + `feed_audio()` loop,
  unchanged.
- **Correction found during implementation:** `WebSocketAudioSink` is
  *not* safe to construct unconditionally — its background thread
  starts pacing and sending blocks (including silence, once the buffer
  is empty) immediately at construction, regardless of whether
  anything is ever `push()`ed (confirmed by reading `ws_sink.py`'s
  `_run()`: it ticks and calls `_schedule_send()` every
  `blocksize/rate` seconds unconditionally). A text-mode session would
  therefore stream continuous silent binary frames even though no TTS
  ever runs. Fix: a new `NullAudioSink(AudioSinkBase)`
  (`src/asr_test/audio/null_sink.py`) — every method a no-op,
  `playing` always `False` — constructed instead of `WebSocketAudioSink`
  when `mode == "text"`.

### Browser UI (`static/index.html`)

- A "Voice" / "Text" mode toggle (two buttons or a `<select>`) in the
  config panel, above Connect.
- Text mode, before connecting: no `getUserMedia`/`AudioContext`/mic
  permission prompt at all — `connect()` branches on the selected mode.
- Text mode, once connected: a text `<input>` + "Send" button appear
  (docked under the transcript). Enter key or the button sends
  `ws.send(JSON.stringify({text: value}))` and clears the input — it
  does **not** append a "you" line locally; the server's own
  `user_text` event (fired by `on_text_message()` before it calls
  `respond()`) is what puts it in the transcript, exactly the same as
  voice mode's STT-echoed line. One rendering path for both modes, no
  risk of a duplicated line if the two ever raced.
- Voice mode is entirely unchanged.

### Explicitly out of scope

- `main.py`/`ws_client.py` (local CLI clients) do not get a text-input
  path — a terminal already lets you type into `main.py`'s own stdin
  for nothing today, and building a parallel text REPL there doesn't
  add anything a browser text session doesn't already cover.
- No mid-session switching between voice and text — pick one at
  connect time, matching how model/provider selection already works.

## D — Token & trace UI

Pure client-side (`static/index.html`), built on data sub-project A
already sends and sub-project B already stores — no new backend logic
beyond the two read endpoints B adds.

### Per-turn stats

When a `bot_text` event carries `usage`, render one compact line under
that message (styled like the existing `.line.system` italic/muted
rows): `"142 in / 38 out · 180/4096 ctx · $0.0003 · tools: get_current_time"` —
tokens/cost/context omitted individually (not the whole line) if their
`usage` key is `null`/missing, and `tools:` segment omitted entirely
when `usage["tool_calls"]` is empty.

### Session totals

A small strip in the `Conversation` panel header (next to the existing
message count) accumulating from every `bot_text.usage` seen this
session: `"1,204 tokens · $0.0021"`. Purely additive client-side state
(a running sum kept in a JS variable, reset on each new `connect()`) —
no new server aggregation needed for the *live* session.

### History panel

A "History" button (in the topbar, next to the connection pill) opens
a panel listing past sessions from `GET /sessions` (id truncated for
display, created-at, mode, turn count — `list_sessions()`'s rows
already carry enough via a `turn_count` computed column, or the
endpoint can just include it via a `COUNT` join). Clicking a session
calls `GET /sessions/{id}` and renders its turns read-only, in the same
`.line` styling as the live transcript, each with its stats line from
"Per-turn stats" above when that turn stored `usage_json`. No resume
button — this is read-only browsing, matching B's own "resume/history"
framing (viewing, not live continuation).

## Error handling

- Storage: a `SessionStore` write failure (disk full, locked file)
  logs and does not crash the session — the same "best-effort telemetry,
  never load-bearing for the actual conversation" posture as the
  existing timing/metrics code. Wrapped in try/except at the `emit()`
  call site in `server.py`, not inside `SessionStore` itself (so a
  genuine bug in `SessionStore` isn't silently swallowed during
  development/testing — only the *call site* protects production
  behavior).
- `GET /sessions/{id}` for an unknown id → 404, not an empty 200.
- Text mode: an empty/whitespace-only `{"text": ""}` message is
  ignored (no turn started), mirroring how voice mode already drops
  short STT output (`len(text) < 2`).

## Testing

- `tests/test_storage.py` (new): `SessionStore` against an in-memory
  (`:memory:`) DB — create/list/get sessions, add turns with and
  without `usage`, `get_session` on an unknown id returns `None`.
- `tests/test_server.py` (extend): `ready` event includes `session_id`;
  a completed turn's `user_text`/`bot_text` land in the injected fake
  store; `GET /sessions` and `GET /sessions/{id}` (found + 404) against
  a fake/in-memory store; `mode=text` accepts JSON `{"text": ...}` and
  never produces a binary (audio) websocket message; `mode=invalid`
  gets the same error+close pattern as invalid `vad_*`.
- `tests/test_agent.py` (extend): `text_only=True` — `respond()` never
  pushes to `tts_q` even though `full_response`/`conversation`/`bot_text`
  behave identically to a normal turn; `on_text_message()` fires
  `user_text` then `bot_text` the same as the STT path does.
- No new UI (JS) automated tests — this project has none today (the
  existing browser UI has zero JS test coverage); manual smoke test
  covers D and C's client-side pieces, called out explicitly at
  hand-off rather than silently skipped.

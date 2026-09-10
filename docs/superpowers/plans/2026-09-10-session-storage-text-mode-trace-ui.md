# Session Persistence + Text Chat Mode + Trace UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist every session's turns (with usage) to SQLite, add a text-only chat mode alongside voice, and surface the tokens/cost/context/tool-call data (from the earlier LLM-provider sub-project) in the browser UI, plus a read-only history browser.

**Architecture:** A new `SessionStore` (plain `sqlite3`) is owned entirely by `server.py`/`main.py` — `Agent` stays storage-unaware, since it already calls `on_event()` for every turn and that's exactly where storage hooks in. Text mode is one new `Agent` flag (`text_only`) that makes `respond()` skip the TTS queue, plus a new small entry point (`on_text_message`) parallel to the existing STT-triggered path — no duplication of `respond()`'s LLM-streaming logic. The trace UI and history browser are additive, client-side-only changes to `static/index.html` built entirely on data the backend already produces or that this plan's storage task adds.

**Tech Stack:** Python 3.14, `sqlite3` (stdlib, no new dependency), pytest, vanilla JS (no new frontend dependency).

**Spec:** `docs/superpowers/specs/2026-09-10-session-storage-text-mode-trace-ui-design.md`

## Global Constraints

- Storage is SQLite via stdlib `sqlite3`, one file at `config.SESSIONS_DB_PATH` (default `"sessions.db"`). No new dependency.
- `Agent` itself never touches storage — `server.py`/`main.py` write to the store from their own `on_event` handling, same place they already forward events to the websocket/console.
- Text mode never synthesizes or plays audio — `respond()`'s TTS-queue push is skipped entirely when `Agent(text_only=True)`.
- Text mode is chosen at connect time via a `mode` query param (`"voice"` default | `"text"`); no mid-session switching.
- `main.py`/`ws_client.py` do not get a text-input path — out of scope per the spec.
- A storage write failure must never crash a session — wrapped in try/except at the call site in `server.py`/`main.py`, not swallowed inside `SessionStore` itself.
- The full existing suite (`uv run pytest tests/ -q`, 89 tests as of this plan) must stay green after every task.
- No new automated JS tests (this project has none today) — Task 8's UI work gets a manual smoke-test note instead.

---

### Task 1: `SessionStore` core (schema, create/list/get)

**Files:**
- Create: `src/asr_test/storage.py`
- Modify: `src/asr_test/config.py` (add `SESSIONS_DB_PATH`)
- Test: `tests/test_storage.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `SessionStore(path: str = "sessions.db")` with `create_session(session_id, mode, tts_engine, llm_model)`, `add_turn(session_id, role, text, usage=None)`, `list_sessions(limit=50) -> list[dict]`, `get_session(session_id) -> dict | None`. Task 2 (`server.py`) and Task 3 (`main.py`) both construct and call this.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_storage.py`:

```python
from asr_test.storage import SessionStore


def _store():
    return SessionStore(":memory:")


def test_create_and_list_sessions():
    store = _store()
    store.create_session("s1", mode="voice", tts_engine="kokoro", llm_model="lfm2.5-230m")
    store.create_session("s2", mode="text", tts_engine=None, llm_model="gpt-4o-mini")

    sessions = store.list_sessions()

    assert [s["id"] for s in sessions] == ["s2", "s1"]  # newest first
    assert sessions[0]["mode"] == "text"
    assert sessions[0]["tts_engine"] is None
    assert sessions[1]["tts_engine"] == "kokoro"
    assert "created_at" in sessions[0]


def test_list_sessions_respects_limit():
    store = _store()
    for i in range(5):
        store.create_session(f"s{i}", mode="voice", tts_engine="kokoro", llm_model="m")

    assert len(store.list_sessions(limit=2)) == 2


def test_add_turn_and_get_session():
    store = _store()
    store.create_session("s1", mode="voice", tts_engine="kokoro", llm_model="m")
    store.add_turn("s1", "user", "hello")
    store.add_turn("s1", "assistant", "hi there", usage={"input_tokens": 5, "output_tokens": 3})

    result = store.get_session("s1")

    assert result["session"]["id"] == "s1"
    assert len(result["turns"]) == 2
    assert result["turns"][0] == {"role": "user", "text": "hello", "usage": None}
    assert result["turns"][1] == {
        "role": "assistant", "text": "hi there",
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }


def test_get_session_returns_turn_count_in_list_sessions():
    store = _store()
    store.create_session("s1", mode="voice", tts_engine="kokoro", llm_model="m")
    store.add_turn("s1", "user", "hello")
    store.add_turn("s1", "assistant", "hi")

    sessions = store.list_sessions()

    assert sessions[0]["turn_count"] == 2


def test_get_session_unknown_id_returns_none():
    store = _store()

    assert store.get_session("does-not-exist") is None


def test_add_turn_without_usage_stores_none():
    store = _store()
    store.create_session("s1", mode="voice", tts_engine="kokoro", llm_model="m")
    store.add_turn("s1", "user", "hello")

    result = store.get_session("s1")

    assert result["turns"][0]["usage"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_storage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'asr_test.storage'`

- [ ] **Step 3: Write the implementation**

Add to `src/asr_test/config.py` (near the other top-level constants, e.g. after `HISTORY_TURNS`):

```python
SESSIONS_DB_PATH = "sessions.db"   # SQLite file for session/turn history — see storage.py
```

Create `src/asr_test/storage.py`:

```python
"""SQLite-backed session/turn history. Owned entirely by the transport
layers (server.py, main.py) -- Agent itself never imports this; it only
ever calls on_event(), and that's where storage hooks in (see
docs/superpowers/specs/2026-09-10-session-storage-text-mode-trace-ui-design.md).
A write failure here must never crash a live session -- callers wrap
add_turn()/create_session() in try/except, this module doesn't swallow
errors itself so a genuine bug surfaces during development."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone


class SessionStore:
    def __init__(self, path: str = "sessions.db"):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    tts_engine TEXT,
                    llm_model TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id),
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    usage_json TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id)"
            )
            self._conn.commit()

    def create_session(
        self, session_id: str, mode: str, tts_engine: str | None, llm_model: str
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (id, created_at, mode, tts_engine, llm_model) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, datetime.now(timezone.utc).isoformat(), mode, tts_engine, llm_model),
            )
            self._conn.commit()

    def add_turn(self, session_id: str, role: str, text: str, usage: dict | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO turns (session_id, role, text, usage_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    session_id, role, text,
                    json.dumps(usage) if usage is not None else None,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._conn.commit()

    def list_sessions(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT s.id, s.created_at, s.mode, s.tts_engine, s.llm_model,
                       COUNT(t.id) AS turn_count
                FROM sessions s
                LEFT JOIN turns t ON t.session_id = s.id
                GROUP BY s.id
                ORDER BY s.created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_session(self, session_id: str) -> dict | None:
        with self._lock:
            session_row = self._conn.execute(
                "SELECT id, created_at, mode, tts_engine, llm_model FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if session_row is None:
                return None
            turn_rows = self._conn.execute(
                "SELECT role, text, usage_json FROM turns WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return {
            "session": dict(session_row),
            "turns": [
                {
                    "role": row["role"],
                    "text": row["text"],
                    "usage": json.loads(row["usage_json"]) if row["usage_json"] else None,
                }
                for row in turn_rows
            ],
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_storage.py -v`
Expected: PASS (6 tests)

Then run: `uv run pytest tests/ -q`
Expected: PASS, all tests green

- [ ] **Step 5: Commit**

```bash
git add src/asr_test/storage.py src/asr_test/config.py tests/test_storage.py
git commit -m "feat: add SQLite SessionStore for session/turn history"
```

---

### Task 2: Wire `SessionStore` into `server.py`

**Files:**
- Modify: `server.py` (imports, `create_app` signature, `ws_endpoint`, two new routes)
- Test: `tests/test_server.py` (extend)

**Interfaces:**
- Consumes: `SessionStore` from Task 1.
- Produces: `create_app(..., session_store: SessionStore | None = None)`; `ready` event payload gains `"session_id"`; `GET /sessions` and `GET /sessions/{session_id}` routes. Task 5 (text mode) adds the `mode` query param this task reads (defaulting to `"voice"` until Task 5 lands, so this task must not assume `mode` already exists — read it with `.get("mode", "voice")` directly here).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_server.py` (near the top, alongside other imports):

```python
from asr_test.storage import SessionStore
```

Add these tests (place near the other `_make_client`-style tests):

```python
def test_ready_event_includes_session_id(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    store = SessionStore(":memory:")
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm(),
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=store,
    )
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ready = ws.receive_json()

    assert "session_id" in ready and ready["session_id"]


def test_completed_turn_is_persisted_to_session_store(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    store = SessionStore(":memory:")
    fake_usage = {"input_tokens": 5, "output_tokens": 3}
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm("hi there", fake_usage=fake_usage),
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=store,
    )
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ready = ws.receive_json()
        session_id = ready["session_id"]
        frame = np.zeros(config.FRAME, dtype=np.float32)
        for _ in range(3):
            ws.send_bytes(float32_to_pcm16(frame))
        ws.receive_bytes()  # wait for the TTS reply so the turn has completed

    result = store.get_session(session_id)
    assert result is not None
    assert result["turns"] == [
        {"role": "user", "text": "hello", "usage": None},
        {"role": "assistant", "text": "hi there ", "usage": fake_usage},
    ]


def test_get_sessions_lists_created_sessions(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    store = SessionStore(":memory:")
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm(),
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=store,
    )
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ready = ws.receive_json()

    resp = client.get("/sessions")

    assert resp.status_code == 200
    ids = [s["id"] for s in resp.json()]
    assert ready["session_id"] in ids


def test_get_session_by_id_returns_detail(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    store = SessionStore(":memory:")
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm(),
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=store,
    )
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ready = ws.receive_json()

    resp = client.get(f"/sessions/{ready['session_id']}")

    assert resp.status_code == 200
    assert resp.json()["session"]["id"] == ready["session_id"]


def test_get_session_by_unknown_id_returns_404(monkeypatch):
    store = SessionStore(":memory:")
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm(),
        default_tts_engine="kokoro",
        session_store=store,
    )
    client = TestClient(app)

    resp = client.get("/sessions/does-not-exist")

    assert resp.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_server.py -k session -v`
Expected: FAIL — `create_app() got an unexpected keyword argument 'session_store'`

- [ ] **Step 3: Write the implementation**

In `server.py`, add imports (alongside the existing `from asr_test...` imports):

```python
import uuid

from fastapi import HTTPException

from asr_test.storage import SessionStore
```

Change `create_app`'s signature — add one new parameter after `llm_base_url`:

```python
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
```

Right after the existing `app = FastAPI()` line, add:

```python
    store = session_store or SessionStore(config.SESSIONS_DB_PATH)
```

Add the two new routes anywhere alongside the existing `@app.get("/options")` route:

```python
    @app.get("/sessions")
    async def list_sessions():
        return store.list_sessions()

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        result = store.get_session(session_id)
        if result is None:
            raise HTTPException(status_code=404, detail="session not found")
        return result
```

In `ws_endpoint`, right after the existing `tts = await get_tts(...)` / `llm = await get_llm(...)` lines and before the `ready` event is sent, add:

```python
        session_id = uuid.uuid4().hex
        mode = params.get("mode", "voice")
        try:
            store.create_session(session_id, mode=mode, tts_engine=tts_engine, llm_model=llm_model)
        except Exception as e:
            print(f"  session store error (create_session): {e}")
```

Change the `ready` event's `send_json` call to include the session id:

```python
        await websocket.send_json({
            "event": "ready",
            "session_id": session_id,
            "input_sample_rate": config.MIC_RATE,
            "output_sample_rate": tts.sample_rate,
            "tts_engine": tts_engine,
            "llm_model": llm_model,
        })
```

Change the `emit()` closure to also persist, right before its `asyncio.run_coroutine_threadsafe(_send(), loop)` call:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_server.py -v`
Expected: PASS, all tests in this file (existing + 5 new)

Then run: `uv run pytest tests/ -q`
Expected: PASS, all tests green

- [ ] **Step 5: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "feat: persist sessions/turns to SessionStore, add /sessions read endpoints"
```

---

### Task 3: Wire `SessionStore` into `main.py` (local CLI)

**Files:**
- Modify: `main.py`

**Interfaces:**
- Consumes: `SessionStore` from Task 1.
- Produces: nothing consumed by later tasks — local-mode persistence is a leaf feature.

- [ ] **Step 1: Manual verification plan (no new automated test — `main.py` has no existing test file; it's an interactive CLI entrypoint verified by running it, matching how every other `main.py` flag in this project was verified)**

After Step 3's implementation, run:

```bash
uv run python -c "
from asr_test.storage import SessionStore
import uuid
store = SessionStore('sessions.db')
store.create_session(uuid.uuid4().hex, mode='voice', tts_engine='kokoro', llm_model='test-model')
print(store.list_sessions())
"
```

Expected: prints a list containing the just-created session — confirms `SessionStore` and the real `sessions.db` file work end-to-end before wiring it into `main.py`'s actual run loop.

- [ ] **Step 2: (no failing-test step for this task — see Step 1)**

- [ ] **Step 3: Write the implementation**

In `main.py`, add to the imports (alongside the existing `from asr_test...` imports):

```python
import uuid

from asr_test.storage import SessionStore
```

In `main()`, right before the existing `agent = Agent(vad=vad, tts=tts, trigger_word=args.trigger_word)` line, add:

```python
    session_store = SessionStore(config.SESSIONS_DB_PATH)
    session_id = uuid.uuid4().hex
    session_store.create_session(
        session_id, mode="voice", tts_engine=args.tts, llm_model="lfm2.5-230m"
    )

    def on_event(name: str, data: dict) -> None:
        try:
            if name == "user_text":
                session_store.add_turn(session_id, "user", data["text"])
            elif name == "bot_text":
                session_store.add_turn(session_id, "assistant", data["text"], data.get("usage"))
        except Exception as e:
            print(f"  session store error: {e}")
```

Change the `Agent(...)` construction to pass it through:

```python
    agent = Agent(vad=vad, tts=tts, trigger_word=args.trigger_word, on_event=on_event)
```

- [ ] **Step 4: Verify**

Run: `uv run python -m py_compile main.py` — Expected: no output (compiles clean)

Then run: `uv run pytest tests/ -q`
Expected: PASS, all tests green (this task touches no test-covered code path in a way that could regress `tests/test_agent.py`'s direct `Agent(...)` construction, since `on_event` there is passed explicitly per-test as before)

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "feat: persist local-mode sessions to SessionStore"
```

---

### Task 4: `Agent` gains `text_only` mode

**Files:**
- Modify: `src/asr_test/agent.py` (`__init__`, `respond()`'s `enqueue()`, new `on_text_message()`)
- Test: `tests/test_agent.py` (extend)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `Agent(..., text_only: bool = False)`; `Agent.on_text_message(text: str) -> None`. Task 5 (`server.py`) calls both.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent.py`:

```python
def test_text_only_agent_never_pushes_to_tts_queue(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    fake_llm = FakeLlm(reply="hi there")
    agent = _build_agent(llm=fake_llm, text_only=True)

    agent.on_text_message("hello")

    assert agent.tts_q.qsize() == 0
    assert agent.conversation == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there "},
    ]


def test_on_text_message_fires_user_text_then_bot_text():
    events: list[tuple[str, dict]] = []
    agent = _build_agent(
        llm=FakeLlm("hi there"), text_only=True,
        on_event=lambda name, data: events.append((name, data)),
    )

    agent.on_text_message("hello")

    assert events == [
        ("user_text", {"text": "hello"}),
        ("bot_text", {"text": "hi there "}),
    ]


def test_voice_mode_agent_still_pushes_to_tts_queue(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    fake_llm = FakeLlm(reply="hi there")
    agent = _build_agent(llm=fake_llm)  # text_only defaults to False

    agent.respond("hello", agent.new_turn(), stt_t=0.0)

    assert agent.tts_q.qsize() == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_agent.py -k "text_only or on_text_message" -v`
Expected: FAIL — `TypeError: Agent.__init__() got an unexpected keyword argument 'text_only'` (and `AttributeError: 'Agent' object has no attribute 'on_text_message'`)

- [ ] **Step 3: Write the implementation**

In `src/asr_test/agent.py`, add `text_only: bool = False` to `Agent.__init__`'s parameter list, right after `trigger_word: str | None = None,`:

```python
        text_only: bool = False,
```

Right after the existing `self.muted = threading.Event()` line at the end of `__init__`, add:

```python
        # Text-mode sessions never synthesize/play audio — see
        # respond()'s enqueue(), which checks this flag before pushing
        # to tts_q. Everything else about a turn (LLM streaming,
        # history, the bot_text event) is identical to voice mode.
        self.text_only = text_only
```

In `respond()`'s inner `enqueue()` function, change:

```python
            chunk_no += 1
            first = False
            self.tts_q.put((turn, chunk_no, chunk, stt_t, turn_start, ttft))
            spoken.append(chunk)
            return True
```

to:

```python
            chunk_no += 1
            first = False
            if not self.text_only:
                self.tts_q.put((turn, chunk_no, chunk, stt_t, turn_start, ttft))
            spoken.append(chunk)
            return True
```

Add a new method right after `interrupt()` (or any other convenient spot near the other public entry points like `feed_audio()`):

```python
    def on_text_message(self, text: str) -> None:
        """Transport-agnostic entry point for one text-mode chat turn —
        the text-mode equivalent of feed_audio() + worker_thread()'s
        STT hand-off, minus VAD/STT entirely. Runs respond() directly
        on the calling thread (the caller — server.py's websocket loop
        — is expected to run this off the event loop, same as any
        other blocking Agent call)."""
        print(f"USER: {text}")
        self._on_event("user_text", {"text": text})
        self.respond(text, self.new_turn(), stt_t=0.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_agent.py -v`
Expected: PASS, all tests in this file (existing + 3 new)

Then run: `uv run pytest tests/ -q`
Expected: PASS, all tests green

- [ ] **Step 5: Commit**

```bash
git add src/asr_test/agent.py tests/test_agent.py
git commit -m "feat: add Agent text_only mode and on_text_message() entry point"
```

---

### Task 5: `server.py` text-mode websocket handling

**Files:**
- Modify: `server.py` (module docstring, `ws_endpoint`'s param parsing and receive loop)
- Test: `tests/test_server.py` (extend)

**Interfaces:**
- Consumes: `Agent(text_only=...)` and `Agent.on_text_message()` from Task 4.
- Produces: `/ws?mode=text|voice` query param, validated the same way as `vad_*`; a `NullAudioSink` (new, `src/asr_test/audio/null_sink.py`) used instead of `WebSocketAudioSink` when `mode == "text"`. Task 6 (browser UI) is the client of this.

**Correction found while implementing this task (not anticipated in
the spec):** `WebSocketAudioSink` cannot be constructed unconditionally
for both modes as originally planned — its background thread starts
sending paced blocks (including silence once its buffer is empty)
immediately at construction, regardless of whether anything is ever
pushed to it. A text-mode session would otherwise stream continuous
silent binary frames despite never running TTS, which broke
`test_ws_text_mode_accepts_json_text_and_replies_with_bot_text` (the
first binary silence frame arrived before the JSON `user_text`/`bot_text`
events, confusing the test's `receive_json()` calls). Fix: added
`NullAudioSink(AudioSinkBase)` — every method a no-op — and construct
it instead of `WebSocketAudioSink` when `mode == "text"`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_server.py`:

```python
def test_ws_text_mode_accepts_json_text_and_replies_with_bot_text(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm("hi there"),
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=SessionStore(":memory:"),
    )
    client = TestClient(app)

    with client.websocket_connect("/ws?mode=text") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "hello there"})

        user_event = ws.receive_json()
        bot_event = ws.receive_json()

    assert user_event == {"event": "user_text", "text": "hello there"}
    assert bot_event["event"] == "bot_text"
    assert bot_event["text"] == "hi there "


def test_ws_text_mode_never_sends_binary_audio(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm("hi there"),
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=SessionStore(":memory:"),
    )
    client = TestClient(app)

    with client.websocket_connect("/ws?mode=text") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "hello there"})
        ws.receive_json()  # user_text
        ws.receive_json()  # bot_text
        # Starlette's TestClient raises if a third message never arrives within
        # its default wait, which is fine here — the point is no *binary* frame
        # was sent in between; if one had been, one of the two receive_json()
        # calls above would have raised (they only accept text frames).


def test_ws_rejects_invalid_mode(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm(),
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=SessionStore(":memory:"),
    )
    client = TestClient(app)

    with client.websocket_connect("/ws?mode=carrier-pigeon") as ws:
        msg = ws.receive_json()
        assert msg["event"] == "error"
        assert "mode" in msg["message"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_server.py -k "text_mode or invalid_mode" -v`
Expected: FAIL — `ws.send_json({"text": ...})` triggers a `WebSocketDisconnect` (server still runs the voice-mode `receive_bytes()` loop, chokes on receiving text instead of bytes) or FastAPI/Starlette raises on the mismatched frame type

- [ ] **Step 3: Write the implementation**

In `server.py`'s module docstring, extend the query-param example line to mention `mode`:

```
                  /ws?tts=kokoro&voice=af_bella&llm_model=lfm2.5-230m&trigger_word=computer&vad_threshold=0.5&vad_min_silence_ms=1200&vad_speech_pad_ms=300&mode=voice
```

and add a sentence near the existing "An invalid vad_* value..." note:

```
                  `mode` is "voice" (default) or "text" — text mode
                  skips mic/VAD/STT/TTS entirely: the client sends
                  {"text": "..."} JSON messages instead of PCM audio,
                  and never receives binary audio frames back. An
                  invalid mode gets the same error+close treatment as
                  an invalid vad_* value.
```

In `ws_endpoint`, right after the existing `trigger_word = params.get("trigger_word") or None` line, add:

```python
        mode = params.get("mode", "voice")
        if mode not in ("voice", "text"):
            await websocket.send_json({
                "event": "error",
                "message": f"mode must be 'voice' or 'text', got {mode!r}",
            })
            await websocket.close(code=1008)
            return
```

(Note: Task 2 already added a `mode = params.get("mode", "voice")` line for the session-store `create_session` call — remove that duplicate line now that this validated version exists above it, and use this `mode` variable at the `store.create_session(...)` call site instead.)

Change the `Agent(...)` construction to pass `text_only`:

```python
        agent = Agent(
            vad=vad, stt=stt, tts=tts, llm=llm,
            trigger_word=trigger_word, audio_sink=sink, on_event=emit,
            text_only=(mode == "text"),
        )
```

Change the receive loop (currently `while True: data = await websocket.receive_bytes(); agent.feed_audio(...)`) to branch on mode:

```python
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
            await loop.run_in_executor(None, agent.shutdown, threads)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_server.py -v`
Expected: PASS, all tests in this file

Then run: `uv run pytest tests/ -q`
Expected: PASS, all tests green

- [ ] **Step 5: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "feat: add text-mode websocket handling (mode=text query param)"
```

---

### Task 6: Browser UI — Voice/Text mode toggle and text input

**Files:**
- Modify: `static/index.html`

**Interfaces:**
- Consumes: `/ws?mode=text` from Task 5.
- Produces: nothing consumed by other tasks (leaf UI feature). Task 7 (trace UI) touches the same file but a disjoint region (the `bot_text` handler and the transcript header), so there's no ordering dependency, but this task is sequenced first since it's the larger structural change.

- [ ] **Step 1: Manual verification plan (no automated JS tests exist in this project — see Global Constraints)**

After Step 3's implementation:
1. `uv run server.py`, open `http://localhost:8000/`.
2. Select "Text" mode, click Connect — no microphone permission prompt should appear.
3. Type a message in the new text input, press Enter — it should appear in the transcript as "You", followed by the bot's reply as "Bot", with no audio playing.
4. Switch back to "Voice" mode (after disconncting), click Connect — the existing mic-based flow should work exactly as before (regression check).

- [ ] **Step 2: (no failing-test step — see Step 1)**

- [ ] **Step 3: Write the implementation**

In `static/index.html`, add a mode toggle as the first `field-group` inside `#configPanel` (right after the opening `<div class="settings-body" id="configPanel">` line, before the existing `<div class="field-group"><h3>Model</h3>`):

```html
            <div class="field-group">
              <h3>Session type</h3>
              <div class="field">
                <label for="modeSelect">Mode</label>
                <select id="modeSelect">
                  <option value="voice">Voice</option>
                  <option value="text">Text</option>
                </select>
              </div>
            </div>
```

Add a text-input row right after the closing `</div>` of `.transcript` (still inside `.transcript-panel`, i.e. modify:

```html
        <div class="transcript" id="transcript"></div>
      </div>
```

to:

```html
        <div class="transcript" id="transcript"></div>
        <div class="text-compose" id="textCompose" hidden>
          <input type="text" id="textInput" placeholder="Type a message…" disabled>
          <button class="primary" id="textSendBtn" disabled>Send</button>
        </div>
      </div>
```

Add the matching CSS, alongside the existing `.transcript` rules:

```css
  .text-compose {
    display: flex;
    gap: 8px;
    padding: 12px 20px;
    border-top: 1px solid var(--border);
  }
  .text-compose input[type="text"] { flex: 1; }
  .text-compose button.primary { flex: none; width: 84px; }
```

In the `<script>` block, add element references alongside the existing `const` block:

```javascript
  const modeSelect = el("modeSelect");
  const textCompose = el("textCompose"), textInput = el("textInput"), textSendBtn = el("textSendBtn");
```

Add a small helper to read the current mode, right after the `let options = null;` line:

```javascript
  function isTextMode() {
    return modeSelect.value === "text";
  }
```

Change `connect()`: wrap the existing microphone-acquisition block (everything from `setState("connecting", ...)` through building `audioConstraints`/`getUserMedia`/`populateMics()`) so it only runs in voice mode, and branch the query-param + websocket setup. Replace the whole `connect()` function body with:

```javascript
  async function connect() {
    connectBtn.disabled = true;
    const textMode = isTextMode();

    if (!textMode) {
      setState("connecting", "Opening the microphone…");
      const audioConstraints = micSelect.value ? { deviceId: { exact: micSelect.value } } : true;
      try {
        micStream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });
      } catch (e) {
        setState("error", "Microphone access was denied");
        connectBtn.disabled = false;
        return;
      }
      populateMics(); // refresh with real labels now that permission is granted
    } else {
      setState("connecting", "Connecting…");
    }

    const params = new URLSearchParams();
    params.set("mode", textMode ? "text" : "voice");
    params.set("tts", ttsEngineSel.value);
    if (ttsVoiceSel.value) params.set("voice", ttsVoiceSel.value);
    params.set("llm_model", llmModelSel.value);
    if (triggerInput.value.trim()) params.set("trigger_word", triggerInput.value.trim());
    if (vadThresholdInput.value.trim()) params.set("vad_threshold", vadThresholdInput.value.trim());
    if (vadMinSilenceInput.value.trim()) params.set("vad_min_silence_ms", vadMinSilenceInput.value.trim());
    if (vadSpeechPadInput.value.trim()) params.set("vad_speech_pad_ms", vadSpeechPadInput.value.trim());

    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}/ws?${params.toString()}`);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => { transcript.innerHTML = ""; turnCount = 0; lineCount.textContent = ""; };

    ws.onclose = () => { if (connected) disconnect("Connection closed"); };
    ws.onerror = () => { setState("error", "Connection error"); };

    ws.onmessage = (event) => {
      if (typeof event.data === "string") {
        const msg = JSON.parse(event.data);
        handleEvent(msg);
        return;
      }
      const floatData = pcm16ToFloat(event.data);
      noteIncomingAudio(floatData);
      if (playCtx && floatData.length) {
        const buffer = playCtx.createBuffer(1, floatData.length, outputSampleRate);
        buffer.copyToChannel(floatData, 0);
        const src = playCtx.createBufferSource();
        src.buffer = buffer;
        src.connect(playCtx.destination);
        const startAt = Math.max(playCtx.currentTime, nextPlayTime);
        src.start(startAt);
        nextPlayTime = startAt + buffer.duration;
      }
    };

    if (!textMode) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      source = audioCtx.createMediaStreamSource(micStream);
      processor = audioCtx.createScriptProcessor(512, 1, 1);
      processor.onaudioprocess = (e) => {
        if (!micMuted && ws && ws.readyState === WebSocket.OPEN) {
          ws.send(floatToPcm16(e.inputBuffer.getChannelData(0)));
        }
      };
      source.connect(processor);
      processor.connect(audioCtx.destination);
    }
  }
```

(This drops the inline comment that was on the playback-scheduling block purely because it's unchanged logic being moved — the comment still applies and can stay if preferred; functionally nothing about that block changes.)

Add a `sendText()` function and wire it up, right after `connect()`:

```javascript
  function sendText() {
    const value = textInput.value.trim();
    if (!value || !ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ text: value }));
    textInput.value = "";
  }

  textSendBtn.onclick = sendText;
  textInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendText();
  });
```

In `handleEvent()`, update the `"ready"` branch to show/hide the mic vs. text controls and skip the speaking-watchdog/playback setup in text mode:

```javascript
  function handleEvent(msg) {
    if (msg.event === "ready") {
      const textMode = isTextMode();
      if (!textMode) {
        outputSampleRate = msg.output_sample_rate;
        playCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: outputSampleRate });
        nextPlayTime = 0;
      }
      connected = true;
      configPanel.querySelectorAll("select, input").forEach((f) => (f.disabled = true));
      settingsPanel.open = false;
      connectBtn.textContent = "Disconnect";
      connectBtn.classList.add("disconnect");
      connectBtn.disabled = false;
      connectBtn.onclick = () => disconnect();
      if (textMode) {
        textCompose.hidden = false;
        textInput.disabled = false;
        textSendBtn.disabled = false;
        textInput.focus();
        setState("listening", "Type a message below");
      } else {
        muteBtn.disabled = false;
        setState("listening", "Listening — just start talking");
        startSpeakingWatchdog();
      }
    } else if (msg.event === "user_text") {
```

(everything from `} else if (msg.event === "user_text") {` onward is unchanged)

Update `disconnect()` to hide/reset the text-compose row, right after the existing `muteBtn.disabled = true;` line:

```javascript
    textCompose.hidden = true;
    textInput.disabled = true;
    textSendBtn.disabled = true;
```

- [ ] **Step 4: Verify (manual, per Step 1)**

Run: `uv run server.py`, then walk through the 4 steps in Step 1. Confirm both modes work and neither regresses the other.

- [ ] **Step 5: Commit**

```bash
git add static/index.html
git commit -m "feat: add Voice/Text mode toggle and text chat input to browser UI"
```

---

### Task 7: Browser UI — per-turn usage stats and session totals

**Files:**
- Modify: `static/index.html`

**Interfaces:**
- Consumes: `bot_text.usage` (already sent by the earlier LLM-provider sub-project; unchanged by this plan).
- Produces: nothing consumed by other tasks.

- [ ] **Step 1: Manual verification plan**

After Step 3, connect (voice or text mode) with a real LM Studio backend running and ask something. Confirm: (a) each bot reply shows a small stats line underneath with tokens/cost/tools, (b) the header count area accumulates a running token/cost total across multiple turns, (c) a turn whose `usage` is absent (e.g. if the LLM call fails/short-circuits) renders with no stats line and doesn't break anything.

- [ ] **Step 2: (no failing-test step — see Step 1)**

- [ ] **Step 3: Write the implementation**

Add CSS for the stats line, alongside the existing `.line` rules:

```css
  .line-stats {
    font-size: 11px;
    color: var(--text-faint);
    font-family: var(--font-mono);
    margin: -2px 0 8px 48px;
  }
```

Add a session-totals JS state and formatter near the existing `let turnCount = 0;` line:

```javascript
  let totalTokens = 0, totalCost = 0, totalCostKnown = false;

  function formatUsageLine(usage) {
    const parts = [];
    if (usage.input_tokens != null && usage.output_tokens != null) {
      const totalPart = usage.context_window != null
        ? `${usage.total_tokens} total / ${usage.context_window} ctx`
        : `${usage.total_tokens} total`;
      parts.push(`${usage.input_tokens} in / ${usage.output_tokens} out (${totalPart})`);
    }
    if (usage.cost_usd != null) parts.push(`$${usage.cost_usd.toFixed(4)}`);
    if (usage.tool_calls && usage.tool_calls.length) {
      parts.push(`tools: ${usage.tool_calls.map((c) => c.name).join(", ")}`);
    }
    return parts.join(" · ");
  }

  function updateSessionTotals(usage) {
    if (usage.total_tokens != null) totalTokens += usage.total_tokens;
    if (usage.cost_usd != null) { totalCost += usage.cost_usd; totalCostKnown = true; }
    const tokenPart = totalTokens ? `${totalTokens.toLocaleString()} tokens` : "";
    const costPart = totalCostKnown ? `$${totalCost.toFixed(4)}` : "";
    lineCount.textContent = [tokenPart, costPart].filter(Boolean).join(" · ")
      || (turnCount ? `${turnCount}${turnCount === 1 ? " message" : " messages"}` : "");
  }
```

Change `addLine()` to accept an optional usage payload and render the stats line:

```javascript
  function addLine(who, text, usage) {
    const line = document.createElement("div");
    line.className = "line " + who;
    const label = { you: "You", bot: "Bot", system: "—" }[who] || who;
    line.innerHTML = `<span class="who">${label}</span><span class="what"></span>`;
    line.querySelector(".what").textContent = text;
    transcript.appendChild(line);

    if (usage) {
      const statsLine = formatUsageLine(usage);
      if (statsLine) {
        const stats = document.createElement("div");
        stats.className = "line-stats";
        stats.textContent = statsLine;
        transcript.appendChild(stats);
      }
      updateSessionTotals(usage);
    }

    transcript.scrollTop = transcript.scrollHeight;
    if (who === "you" || who === "bot") {
      turnCount++;
      if (!usage) lineCount.textContent = turnCount + (turnCount === 1 ? " message" : " messages");
    }
  }
```

Change the `"bot_text"` branch in `handleEvent()` to pass `msg.usage` through:

```javascript
    } else if (msg.event === "bot_text") {
      addLine("bot", msg.text, msg.usage);
```

Reset the totals on each new connection — in `ws.onopen`, change:

```javascript
    ws.onopen = () => { transcript.innerHTML = ""; turnCount = 0; lineCount.textContent = ""; };
```

to:

```javascript
    ws.onopen = () => {
      transcript.innerHTML = "";
      turnCount = 0;
      totalTokens = 0;
      totalCost = 0;
      totalCostKnown = false;
      lineCount.textContent = "";
    };
```

- [ ] **Step 4: Verify (manual, per Step 1)**

- [ ] **Step 5: Commit**

```bash
git add static/index.html
git commit -m "feat: show per-turn token/cost/tool stats and running session totals in UI"
```

---

### Task 8: Browser UI — read-only History panel

**Files:**
- Modify: `static/index.html`

**Interfaces:**
- Consumes: `GET /sessions` and `GET /sessions/{id}` from Task 2.
- Produces: nothing consumed by other tasks — final task in this plan.

- [ ] **Step 1: Manual verification plan**

After Step 3, have at least one completed session (from earlier manual testing). Click the new "History" button in the topbar: confirm it lists past sessions (mode, turn count, timestamp) newest-first, and clicking one shows its turns read-only with each turn's stats line (from Task 7's `formatUsageLine`) where `usage` was stored. Confirm there's no way to "resume" a session from this panel (view-only, per spec).

- [ ] **Step 2: (no failing-test step — see Step 1)**

- [ ] **Step 3: Write the implementation**

Add a History button to the topbar, right after the existing `.conn-pill` element (inside `.topbar`):

```html
      <button class="secondary" id="historyBtn" style="width:auto;padding:6px 14px;">History</button>
```

Add a modal/panel markup right before the closing `</div>` of `.app`:

```html
    <div class="panel" id="historyModal" hidden style="position:fixed;inset:40px;z-index:10;overflow-y:auto;max-width:640px;margin:0 auto;">
      <div class="transcript-header">
        <h2 id="historyTitle">History</h2>
        <button class="secondary" id="historyCloseBtn" style="width:auto;padding:4px 10px;">Close</button>
      </div>
      <div id="historyBody" style="padding:16px 20px;"></div>
    </div>
```

Add element references and the fetch/render logic in the `<script>` block, near the other DOM-reference declarations:

```javascript
  const historyBtn = el("historyBtn"), historyModal = el("historyModal");
  const historyTitle = el("historyTitle"), historyBody = el("historyBody"), historyCloseBtn = el("historyCloseBtn");

  async function openHistoryList() {
    historyTitle.textContent = "History";
    historyBody.innerHTML = "Loading…";
    historyModal.hidden = false;
    const sessions = await (await fetch("/sessions")).json();
    if (!sessions.length) {
      historyBody.textContent = "No past sessions yet.";
      return;
    }
    historyBody.innerHTML = "";
    for (const s of sessions) {
      const row = document.createElement("div");
      row.style.cssText = "padding:10px 0;border-bottom:1px solid var(--border);cursor:pointer;";
      row.textContent = `${s.created_at}  ·  ${s.mode}  ·  ${s.turn_count} turns`;
      row.onclick = () => openHistoryDetail(s.id);
      historyBody.appendChild(row);
    }
  }

  async function openHistoryDetail(sessionId) {
    historyTitle.textContent = "Session " + sessionId.slice(0, 8);
    historyBody.innerHTML = "Loading…";
    const detail = await (await fetch(`/sessions/${sessionId}`)).json();
    historyBody.innerHTML = "";
    const back = document.createElement("button");
    back.className = "secondary";
    back.style.cssText = "width:auto;padding:4px 10px;margin-bottom:12px;";
    back.textContent = "← All sessions";
    back.onclick = openHistoryList;
    historyBody.appendChild(back);
    for (const turn of detail.turns) {
      const line = document.createElement("div");
      line.className = "line " + (turn.role === "user" ? "you" : "bot");
      const label = turn.role === "user" ? "You" : "Bot";
      line.innerHTML = `<span class="who">${label}</span><span class="what"></span>`;
      line.querySelector(".what").textContent = turn.text;
      historyBody.appendChild(line);
      if (turn.usage) {
        const statsLine = formatUsageLine(turn.usage);
        if (statsLine) {
          const stats = document.createElement("div");
          stats.className = "line-stats";
          stats.textContent = statsLine;
          historyBody.appendChild(stats);
        }
      }
    }
  }

  historyBtn.onclick = openHistoryList;
  historyCloseBtn.onclick = () => { historyModal.hidden = true; };
```

- [ ] **Step 4: Verify (manual, per Step 1)**

- [ ] **Step 5: Commit**

```bash
git add static/index.html
git commit -m "feat: add read-only session History panel to browser UI"
```

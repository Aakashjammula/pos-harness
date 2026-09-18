# Multi-User Platform (Phases 1-3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `pos-harness` from a single-tenant local tool into a multi-user application: pooled Postgres access, email+password accounts, per-user session isolation, and per-user API credentials encrypted at rest.

**Architecture:** A `ConnectionPool` replaces the single shared Postgres connection. All DDL moves to one `init_schema()` so the schema is created once. Auth is Argon2id password hashing plus a short-lived JWT access token and a rotating refresh token, both in httpOnly cookies. Provider API keys are stored per user as AES-256-GCM ciphertext and decrypted in memory only when building an LLM client, replacing the current ephemeral `/session-keys` + `key_token` flow.

**Tech Stack:** Python 3.14, FastAPI, psycopg 3 + psycopg-pool, argon2-cffi, PyJWT, cryptography (AES-GCM), Postgres 16, Next.js 16 / React 19 / Tailwind v4.

**Spec:** `docs/superpowers/specs/2026-09-18-multi-user-platform-design.md`

## Global Constraints

- **Phases 4 and 5 are NOT in this plan.** Their tables and columns (`user_settings`, `session_summaries`, `usage_events`, `turns.token_count`, `turns.pii_flags`) ARE created in Task 2 so the schema is built once, but nothing reads or writes them here. Do **not** build `GET`/`PUT /settings`, compaction, PII detection, or call limits — they belong to later plans.
- **Run everything from `backend/`.** All `uv run` commands assume that working directory.
- **Tests need a live Postgres.** `TEST_DATABASE_URL` defaults to `postgresql://pos:pos@localhost:5432/pos`. Start one with `docker compose up -d postgres` from the repo root.
- **Ruff is the lint gate:** `uv run ruff check .` must pass. Line length 120. Selected rules: `E`, `F`, `I`, `UP`, `B`.
- **Never log, echo, or return a secret.** Passwords, raw refresh tokens, and decrypted API keys must never appear in a response body, a URL/query string, or a log line. The codebase already avoids putting keys in URLs specifically because uvicorn's access log would persist them.
- **No comments explaining what code does.** This codebase writes comments only for non-obvious *why*. Match that.
- **Cookie/CORS rule (read before Task 7):** frontend `localhost:3000` and backend `localhost:8000` are different *origins* but the same *site* (SameSite ignores port), so `SameSite=Lax` cookies are sent on both fetch and WebSocket handshakes. This requires CORS with `allow_credentials=True` and an explicit origin allowlist — `allow_origins=["*"]` is rejected by browsers when credentials are included. The `Secure` flag must be env-driven (`COOKIE_SECURE`, default `false`) because local dev is plain HTTP.

---

### Task 1: Connection pool replaces the single connection + global lock

**Files:**
- Create: `backend/src/pos/db.py`
- Modify: `backend/pyproject.toml` (add `psycopg-pool`)
- Modify: `backend/src/pos/config.py`
- Test: `backend/tests/test_db.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `pos.db.create_pool(dsn: str, max_size: int = ...) -> ConnectionPool`. Every later task takes a `ConnectionPool` rather than a connection.

- [ ] **Step 1: Add the dependency**

In `backend/pyproject.toml`, add to `dependencies` (keep the list alphabetical):

```toml
    "psycopg-pool>=3.2.0",
```

Then run:

```bash
uv lock && uv sync
```

- [ ] **Step 2: Add pool config**

In `backend/src/pos/config.py`, below the existing `DATABASE_URL` line:

```python
DB_POOL_MAX_SIZE = int(os.environ.get("DB_POOL_MAX_SIZE", "10"))
```

- [ ] **Step 3: Write the failing test**

Create `backend/tests/test_db.py`:

```python
import os
import threading

from pos.db import create_pool

_DSN = os.environ.get("TEST_DATABASE_URL", "postgresql://pos:pos@localhost:5432/pos")


def test_pool_runs_concurrent_queries():
    pool = create_pool(_DSN, max_size=5)
    results = []

    def query():
        with pool.connection() as conn:
            row = conn.execute("SELECT 1 AS n").fetchone()
            results.append(row["n"])

    threads = [threading.Thread(target=query) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == [1] * 20
    pool.close()


def test_pool_uses_dict_rows():
    pool = create_pool(_DSN, max_size=2)
    with pool.connection() as conn:
        row = conn.execute("SELECT 42 AS answer").fetchone()
    assert row == {"answer": 42}
    pool.close()
```

- [ ] **Step 4: Run the test to verify it fails**

```bash
uv run pytest tests/test_db.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pos.db'`.

- [ ] **Step 5: Write the implementation**

Create `backend/src/pos/db.py`:

```python
"""Postgres connection pool shared by every store in the process.

Replaces the single connection + global lock the SQLite-era SessionStore
used: that serialized every query process-wide, which Postgres has no
reason to do."""

from __future__ import annotations

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from pos import config


def create_pool(dsn: str | None = None, max_size: int | None = None) -> ConnectionPool:
    return ConnectionPool(
        conninfo=dsn or config.DATABASE_URL,
        min_size=2,
        max_size=max_size or config.DB_POOL_MAX_SIZE,
        kwargs={"row_factory": dict_row, "autocommit": True},
        open=True,
    )
```

- [ ] **Step 6: Run the test to verify it passes**

```bash
uv run pytest tests/test_db.py -v
```

Expected: PASS (2 passed).

- [ ] **Step 7: Commit**

```bash
git add backend/pyproject.toml backend/uv.lock backend/src/pos/db.py backend/src/pos/config.py backend/tests/test_db.py
git commit -m "feat: add Postgres connection pool"
```

---

### Task 2: Create the full schema in one place

**Files:**
- Modify: `backend/src/pos/db.py`
- Test: `backend/tests/test_schema.py`

**Interfaces:**
- Consumes: `pos.db.create_pool`.
- Produces: `pos.db.init_schema(pool: ConnectionPool) -> None` — idempotent, creates every table, column, and index for phases 1-5.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_schema.py`:

```python
import os

from pos.db import create_pool, init_schema

_DSN = os.environ.get("TEST_DATABASE_URL", "postgresql://pos:pos@localhost:5432/pos")

_EXPECTED_TABLES = {
    "users",
    "user_settings",
    "refresh_tokens",
    "api_credentials",
    "sessions",
    "turns",
    "session_summaries",
    "usage_events",
}


def _pool():
    pool = create_pool(_DSN, max_size=3)
    init_schema(pool)
    return pool


def test_init_schema_creates_every_table():
    pool = _pool()
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        ).fetchall()
    names = {r["table_name"] for r in rows}
    assert _EXPECTED_TABLES <= names
    pool.close()


def test_init_schema_is_idempotent():
    pool = _pool()
    init_schema(pool)
    init_schema(pool)
    pool.close()


def test_deleting_a_user_cascades_to_sessions_and_turns():
    pool = _pool()
    with pool.connection() as conn:
        conn.execute("DELETE FROM users WHERE email = 'cascade@test'")
        user = conn.execute(
            "INSERT INTO users (email, password_hash) VALUES ('cascade@test', 'x') RETURNING id"
        ).fetchone()
        conn.execute(
            "INSERT INTO sessions (id, user_id, created_at, mode, llm_model) "
            "VALUES ('cascade-s1', %s, now(), 'text', 'm')",
            (user["id"],),
        )
        conn.execute(
            "INSERT INTO turns (session_id, role, text, created_at) "
            "VALUES ('cascade-s1', 'user', 'hi', now())"
        )

        conn.execute("DELETE FROM users WHERE id = %s", (user["id"],))

        assert conn.execute("SELECT 1 FROM sessions WHERE id = 'cascade-s1'").fetchone() is None
        assert conn.execute(
            "SELECT 1 FROM turns WHERE session_id = 'cascade-s1'"
        ).fetchone() is None
    pool.close()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/test_schema.py -v
```

Expected: FAIL with `ImportError: cannot import name 'init_schema'`.

- [ ] **Step 3: Write the schema**

Append to `backend/src/pos/db.py`:

```python
_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS user_settings (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    compact_trigger_fraction REAL,
    compact_keep_messages INTEGER,
    pii_strategy TEXT,
    pii_apply_to_output BOOLEAN,
    tool_calls_per_hour INTEGER,
    model_calls_per_hour INTEGER,
    max_cost_usd_per_day NUMERIC(10, 4),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS refresh_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL UNIQUE,
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_refresh_tokens_expires ON refresh_tokens(expires_at);
CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user ON refresh_tokens(user_id);

CREATE TABLE IF NOT EXISTS api_credentials (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    encrypted_payload BYTEA NOT NULL,
    nonce BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, provider)
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL,
    mode TEXT NOT NULL,
    tts_engine TEXT,
    llm_model TEXT NOT NULL,
    title TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_user_created ON sessions(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS turns (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    usage_json JSONB,
    created_at TIMESTAMPTZ NOT NULL,
    token_count INTEGER,
    pii_flags JSONB
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);

CREATE TABLE IF NOT EXISTS session_summaries (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    summary_text TEXT NOT NULL,
    covers_through_turn_id BIGINT NOT NULL,
    token_count INTEGER NOT NULL,
    model TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_summaries_session ON session_summaries(session_id, created_at DESC);

CREATE TABLE IF NOT EXISTS usage_events (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
    kind TEXT NOT NULL,
    name TEXT,
    status TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd NUMERIC(12, 6),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_usage_events_user_kind
    ON usage_events(user_id, kind, created_at DESC);
"""


def init_schema(pool: ConnectionPool) -> None:
    with pool.connection() as conn:
        conn.execute(_SCHEMA)
```

`sessions.user_id` is deliberately nullable at the database level even though
the application always sets it — a `NOT NULL` column cannot be added to a
table that already has rows in a developer's existing database, and the
ownership check lives in the queries either way.

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/test_schema.py -v
```

Expected: PASS (3 passed).

If the first run fails on `CREATE EXTENSION`, the test database user lacks
superuser rights; grant it or create the extension manually once with
`docker compose exec postgres psql -U pos -d pos -c 'CREATE EXTENSION pgcrypto'`.

- [ ] **Step 5: Commit**

```bash
git add backend/src/pos/db.py backend/tests/test_schema.py
git commit -m "feat: create full multi-user schema in init_schema"
```

---

### Task 3: Move SessionStore onto the pool and scope it to a user

**Files:**
- Modify: `backend/src/pos/storage.py`
- Modify: `backend/tests/test_storage.py`

**Interfaces:**
- Consumes: `pos.db.create_pool`, `pos.db.init_schema`.
- Produces: `SessionStore(pool: ConnectionPool)` with these signatures — later tasks call exactly these:
  - `create_session(session_id: str, user_id: str, mode: str, tts_engine: str | None, llm_model: str) -> None`
  - `add_turn(session_id: str, role: str, text: str, usage: dict | None = None) -> None`
  - `set_title(session_id: str, title: str) -> None`
  - `set_mode(session_id: str, mode: str) -> None`
  - `delete_session(session_id: str, user_id: str) -> bool`
  - `list_sessions(user_id: str, limit: int = 50) -> list[dict]`
  - `get_session(session_id: str, user_id: str) -> dict | None`

- [ ] **Step 1: Rewrite the tests**

Replace the top of `backend/tests/test_storage.py` (the imports and `_store()`) with:

```python
import os

from pos.db import create_pool, init_schema
from pos.storage import SessionStore

_DSN = os.environ.get("TEST_DATABASE_URL", "postgresql://pos:pos@localhost:5432/pos")


def _store():
    """A SessionStore on a clean database, plus one user that owns
    everything the test creates."""
    pool = create_pool(_DSN, max_size=5)
    init_schema(pool)
    with pool.connection() as conn:
        conn.execute("TRUNCATE users, sessions, turns RESTART IDENTITY CASCADE")
        user = conn.execute(
            "INSERT INTO users (email, password_hash) VALUES ('owner@test', 'x') RETURNING id"
        ).fetchone()
    store = SessionStore(pool)
    store.test_user_id = str(user["id"])
    return store
```

Then update every existing test in that file: `create_session` gains
`user_id=store.test_user_id`, and `list_sessions`/`get_session`/`delete_session`
gain `store.test_user_id`. For example:

```python
def test_create_and_list_sessions():
    store = _store()
    store.create_session("s1", store.test_user_id, mode="voice", tts_engine="kokoro", llm_model="lfm2.5-230m")
    store.create_session("s2", store.test_user_id, mode="text", tts_engine=None, llm_model="gpt-4o-mini")

    sessions = store.list_sessions(store.test_user_id)

    assert [s["id"] for s in sessions] == ["s2", "s1"]
    assert sessions[0]["mode"] == "text"
    assert sessions[0]["tts_engine"] is None
    assert sessions[1]["tts_engine"] == "kokoro"
    assert "created_at" in sessions[0]
```

Add this new isolation test at the end of the file:

```python
def test_one_user_cannot_see_or_delete_another_users_session():
    store = _store()
    with store._pool.connection() as conn:
        other = conn.execute(
            "INSERT INTO users (email, password_hash) VALUES ('other@test', 'x') RETURNING id"
        ).fetchone()
    other_id = str(other["id"])
    store.create_session("mine", store.test_user_id, mode="text", tts_engine=None, llm_model="m")

    assert store.list_sessions(other_id) == []
    assert store.get_session("mine", other_id) is None
    assert store.delete_session("mine", other_id) is False
    assert store.get_session("mine", store.test_user_id) is not None
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_storage.py -v
```

Expected: FAIL — `SessionStore.__init__` still takes a DSN string, and
`create_session` does not accept `user_id`.

- [ ] **Step 3: Rewrite SessionStore**

Replace the whole of `backend/src/pos/storage.py` with:

```python
"""Postgres-backed session/turn history. Owned entirely by the transport
layers (server.py, cli/local.py) -- Agent itself never imports this; it
only ever calls on_event(), and that's where storage hooks in.
A write failure here must never crash a live session -- callers wrap
add_turn()/create_session() in try/except, this module doesn't swallow
errors itself so a genuine bug surfaces during development."""

from __future__ import annotations

from datetime import UTC, datetime

from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


class SessionStore:
    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def create_session(
        self, session_id: str, user_id: str, mode: str, tts_engine: str | None, llm_model: str
    ) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO sessions (id, user_id, created_at, mode, tts_engine, llm_model) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (session_id, user_id, datetime.now(UTC), mode, tts_engine, llm_model),
            )

    def add_turn(self, session_id: str, role: str, text: str, usage: dict | None = None) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO turns (session_id, role, text, usage_json, created_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (
                    session_id, role, text,
                    Jsonb(usage) if usage is not None else None,
                    datetime.now(UTC),
                ),
            )

    def set_title(self, session_id: str, title: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("UPDATE sessions SET title = %s WHERE id = %s", (title, session_id))

    def set_mode(self, session_id: str, mode: str) -> None:
        """A session's stored mode reflects whichever mode it was most
        recently used in, not just how it was first created -- see
        server.py's ws_endpoint, which calls this when a resumed
        connection's mode differs from what's on record."""
        with self._pool.connection() as conn:
            conn.execute("UPDATE sessions SET mode = %s WHERE id = %s", (mode, session_id))

    def delete_session(self, session_id: str, user_id: str) -> bool:
        """Returns True if a session was deleted, False if the id is
        unknown or belongs to someone else. turns rows go with it via
        ON DELETE CASCADE."""
        with self._pool.connection() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE id = %s AND user_id = %s", (session_id, user_id)
            )
            return cursor.rowcount > 0

    def list_sessions(self, user_id: str, limit: int = 50) -> list[dict]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT s.id, s.created_at, s.mode, s.tts_engine, s.llm_model, s.title,
                       COUNT(t.id) AS turn_count
                FROM sessions s
                LEFT JOIN turns t ON t.session_id = s.id
                WHERE s.user_id = %s
                GROUP BY s.id
                ORDER BY s.created_at DESC
                LIMIT %s
                """,
                (user_id, limit),
            ).fetchall()
        return [{**row, "created_at": _iso(row["created_at"])} for row in rows]

    def get_session(self, session_id: str, user_id: str) -> dict | None:
        with self._pool.connection() as conn:
            session_row = conn.execute(
                "SELECT id, created_at, mode, tts_engine, llm_model, title "
                "FROM sessions WHERE id = %s AND user_id = %s",
                (session_id, user_id),
            ).fetchone()
            if session_row is None:
                return None
            turn_rows = conn.execute(
                "SELECT role, text, usage_json FROM turns WHERE session_id = %s ORDER BY id",
                (session_id,),
            ).fetchall()
        return {
            "session": {**session_row, "created_at": _iso(session_row["created_at"])},
            "turns": [
                {"role": r["role"], "text": r["text"], "usage": r["usage_json"]}
                for r in turn_rows
            ],
        }
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_storage.py -v
```

Expected: PASS (12 passed — the 11 existing tests plus the new isolation test).

- [ ] **Step 5: Commit**

```bash
git add backend/src/pos/storage.py backend/tests/test_storage.py
git commit -m "feat: scope SessionStore to a user and run it on the pool"
```

---

### Task 4: Password hashing

**Files:**
- Create: `backend/src/pos/auth/__init__.py`, `backend/src/pos/auth/passwords.py`
- Modify: `backend/pyproject.toml` (add `argon2-cffi`)
- Test: `backend/tests/test_auth_passwords.py`

**Interfaces:**
- Produces: `hash_password(password: str) -> str`, `verify_password(password: str, password_hash: str) -> bool`.

- [ ] **Step 1: Add the dependency**

Add to `backend/pyproject.toml` `dependencies`:

```toml
    "argon2-cffi>=25.1.0",
```

Then: `uv lock && uv sync`

- [ ] **Step 2: Write the failing test**

Create `backend/tests/test_auth_passwords.py`:

```python
from pos.auth.passwords import hash_password, verify_password


def test_hash_is_argon2id_and_not_the_password():
    digest = hash_password("correct horse battery staple")
    assert digest.startswith("$argon2id$")
    assert "correct horse battery staple" not in digest


def test_verify_accepts_the_right_password():
    digest = hash_password("s3cret")
    assert verify_password("s3cret", digest) is True


def test_verify_rejects_the_wrong_password():
    digest = hash_password("s3cret")
    assert verify_password("not-it", digest) is False


def test_two_hashes_of_the_same_password_differ():
    assert hash_password("same") != hash_password("same")


def test_verify_rejects_a_malformed_hash_instead_of_raising():
    assert verify_password("anything", "not-a-hash") is False
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
uv run pytest tests/test_auth_passwords.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pos.auth'`.

- [ ] **Step 4: Write the implementation**

Create `backend/src/pos/auth/__init__.py` (empty file).

Create `backend/src/pos/auth/passwords.py`:

```python
"""Argon2id password hashing. OWASP's Password Storage Cheat Sheet ranks
Argon2id above bcrypt: memory-hard, so GPU/ASIC cracking gains less, and
time and memory cost are tunable independently. argon2-cffi's defaults
already exceed OWASP's floor (19 MiB / 2 iterations / 1 parallelism)."""

from __future__ import annotations

from argon2 import PasswordHasher
from argon2.exceptions import Argon2Error

_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (Argon2Error, ValueError):
        return False
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
uv run pytest tests/test_auth_passwords.py -v
```

Expected: PASS (5 passed).

- [ ] **Step 6: Commit**

```bash
git add backend/pyproject.toml backend/uv.lock backend/src/pos/auth/ backend/tests/test_auth_passwords.py
git commit -m "feat: add argon2id password hashing"
```

---

### Task 5: Access and refresh tokens

**Files:**
- Create: `backend/src/pos/auth/tokens.py`
- Modify: `backend/pyproject.toml` (add `pyjwt`)
- Modify: `backend/src/pos/config.py`
- Test: `backend/tests/test_auth_tokens.py`

**Interfaces:**
- Produces:
  - `create_access_token(user_id: str) -> str`
  - `decode_access_token(token: str) -> str | None` (returns the user id, or `None` if invalid/expired)
  - `new_refresh_token() -> tuple[str, str]` (raw token, sha256 hash)
  - `hash_refresh_token(raw: str) -> str`
  - `ACCESS_TOKEN_TTL: timedelta`, `REFRESH_TOKEN_TTL: timedelta`

- [ ] **Step 1: Add the dependency**

Add to `backend/pyproject.toml` `dependencies`:

```toml
    "pyjwt>=2.10.0",
```

Then: `uv lock && uv sync`

- [ ] **Step 2: Add config**

Add to `backend/src/pos/config.py`:

```python
JWT_SECRET = os.environ.get("JWT_SECRET", "")
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "false").lower() == "true"
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]
```

- [ ] **Step 3: Write the failing test**

Create `backend/tests/test_auth_tokens.py`:

```python
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from pos import config
from pos.auth.tokens import (
    ACCESS_TOKEN_TTL,
    REFRESH_TOKEN_TTL,
    create_access_token,
    decode_access_token,
    hash_refresh_token,
    new_refresh_token,
)


@pytest.fixture(autouse=True)
def _secret(monkeypatch):
    monkeypatch.setattr(config, "JWT_SECRET", "test-secret")


def test_access_token_round_trips_the_user_id():
    token = create_access_token("user-123")
    assert decode_access_token(token) == "user-123"


def test_access_token_expires():
    token = jwt.encode(
        {"sub": "user-123", "exp": datetime.now(UTC) - timedelta(seconds=1)},
        "test-secret",
        algorithm="HS256",
    )
    assert decode_access_token(token) is None


def test_access_token_signed_with_another_secret_is_rejected():
    token = jwt.encode(
        {"sub": "user-123", "exp": datetime.now(UTC) + timedelta(minutes=5)},
        "wrong-secret",
        algorithm="HS256",
    )
    assert decode_access_token(token) is None


def test_garbage_token_is_rejected():
    assert decode_access_token("not.a.token") is None


def test_refresh_token_returns_raw_and_hash_and_never_stores_raw():
    raw, digest = new_refresh_token()
    assert raw != digest
    assert hash_refresh_token(raw) == digest
    assert len(digest) == 64


def test_ttls_match_the_spec():
    assert ACCESS_TOKEN_TTL == timedelta(minutes=15)
    assert REFRESH_TOKEN_TTL == timedelta(days=30)
```

- [ ] **Step 4: Run the test to verify it fails**

```bash
uv run pytest tests/test_auth_tokens.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pos.auth.tokens'`.

- [ ] **Step 5: Write the implementation**

Create `backend/src/pos/auth/tokens.py`:

```python
"""Short-lived JWT access tokens plus opaque refresh tokens.

Only the sha256 of a refresh token is ever stored, so a database leak
yields nothing usable -- same reasoning as password_hash."""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta

import jwt

from pos import config

ACCESS_TOKEN_TTL = timedelta(minutes=15)
REFRESH_TOKEN_TTL = timedelta(days=30)
_ALGORITHM = "HS256"


def create_access_token(user_id: str) -> str:
    now = datetime.now(UTC)
    return jwt.encode(
        {"sub": user_id, "iat": now, "exp": now + ACCESS_TOKEN_TTL},
        config.JWT_SECRET,
        algorithm=_ALGORITHM,
    )


def decode_access_token(token: str) -> str | None:
    try:
        payload = jwt.decode(token, config.JWT_SECRET, algorithms=[_ALGORITHM])
    except jwt.PyJWTError:
        return None
    subject = payload.get("sub")
    return subject if isinstance(subject, str) else None


def hash_refresh_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def new_refresh_token() -> tuple[str, str]:
    raw = secrets.token_urlsafe(48)
    return raw, hash_refresh_token(raw)
```

- [ ] **Step 6: Run the test to verify it passes**

```bash
uv run pytest tests/test_auth_tokens.py -v
```

Expected: PASS (6 passed).

- [ ] **Step 7: Commit**

```bash
git add backend/pyproject.toml backend/uv.lock backend/src/pos/auth/tokens.py backend/src/pos/config.py backend/tests/test_auth_tokens.py
git commit -m "feat: add JWT access tokens and hashed refresh tokens"
```

---

### Task 6: Credential encryption

**Files:**
- Create: `backend/src/pos/auth/crypto.py`
- Modify: `backend/pyproject.toml` (add `cryptography`)
- Modify: `backend/src/pos/config.py`
- Test: `backend/tests/test_auth_crypto.py`

**Interfaces:**
- Produces: `encrypt_payload(payload: dict) -> tuple[bytes, bytes]` returning `(ciphertext, nonce)`, and `decrypt_payload(ciphertext: bytes, nonce: bytes) -> dict`.

- [ ] **Step 1: Add the dependency and config**

Add to `backend/pyproject.toml` `dependencies`:

```toml
    "cryptography>=46.0.0",
```

Add to `backend/src/pos/config.py`:

```python
ENCRYPTION_KEY = os.environ.get("ENCRYPTION_KEY", "")
```

Then: `uv lock && uv sync`

- [ ] **Step 2: Write the failing test**

Create `backend/tests/test_auth_crypto.py`:

```python
import base64
import os

import pytest

from pos import config
from pos.auth.crypto import decrypt_payload, encrypt_payload


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setattr(config, "ENCRYPTION_KEY", base64.b64encode(os.urandom(32)).decode())


def test_round_trips_a_payload():
    payload = {"OPENAI_API_KEY": "sk-test-123"}
    ciphertext, nonce = encrypt_payload(payload)
    assert decrypt_payload(ciphertext, nonce) == payload


def test_ciphertext_does_not_contain_the_plaintext():
    ciphertext, _ = encrypt_payload({"OPENAI_API_KEY": "sk-test-123"})
    assert b"sk-test-123" not in ciphertext


def test_each_encryption_uses_a_fresh_nonce():
    _, nonce_a = encrypt_payload({"k": "v"})
    _, nonce_b = encrypt_payload({"k": "v"})
    assert nonce_a != nonce_b
    assert len(nonce_a) == 12


def test_tampered_ciphertext_is_rejected():
    ciphertext, nonce = encrypt_payload({"k": "v"})
    tampered = bytes([ciphertext[0] ^ 0xFF]) + ciphertext[1:]
    with pytest.raises(Exception):
        decrypt_payload(tampered, nonce)


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(config, "ENCRYPTION_KEY", "")
    with pytest.raises(RuntimeError, match="ENCRYPTION_KEY"):
        encrypt_payload({"k": "v"})
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
uv run pytest tests/test_auth_crypto.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pos.auth.crypto'`.

- [ ] **Step 4: Write the implementation**

Create `backend/src/pos/auth/crypto.py`:

```python
"""AES-256-GCM envelope encryption for stored provider credentials.

GCM needs a unique nonce per encryption under one key; the nonce isn't
secret and is stored alongside the ciphertext. Losing ENCRYPTION_KEY
makes every stored credential unrecoverable -- back it up like any other
production secret."""

from __future__ import annotations

import base64
import json
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from pos import config

_NONCE_BYTES = 12


def _key() -> bytes:
    if not config.ENCRYPTION_KEY:
        raise RuntimeError("ENCRYPTION_KEY is not set -- cannot store provider credentials")
    key = base64.b64decode(config.ENCRYPTION_KEY)
    if len(key) != 32:
        raise RuntimeError("ENCRYPTION_KEY must decode to exactly 32 bytes (AES-256)")
    return key


def encrypt_payload(payload: dict) -> tuple[bytes, bytes]:
    nonce = os.urandom(_NONCE_BYTES)
    ciphertext = AESGCM(_key()).encrypt(nonce, json.dumps(payload).encode(), None)
    return ciphertext, nonce


def decrypt_payload(ciphertext: bytes, nonce: bytes) -> dict:
    return json.loads(AESGCM(_key()).decrypt(nonce, ciphertext, None))
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
uv run pytest tests/test_auth_crypto.py -v
```

Expected: PASS (5 passed).

- [ ] **Step 6: Commit**

```bash
git add backend/pyproject.toml backend/uv.lock backend/src/pos/auth/crypto.py backend/src/pos/config.py backend/tests/test_auth_crypto.py
git commit -m "feat: add AES-GCM encryption for provider credentials"
```

---

### Task 7: UserStore — users, refresh tokens, credentials

**Files:**
- Create: `backend/src/pos/auth/store.py`
- Test: `backend/tests/test_auth_store.py`

**Interfaces:**
- Consumes: `pos.db` pool, `pos.auth.crypto`.
- Produces: `UserStore(pool)` with:
  - `create_user(email: str, password_hash: str) -> dict` (raises `EmailTaken` on duplicate)
  - `get_user_by_email(email: str) -> dict | None`
  - `get_user_by_id(user_id: str) -> dict | None`
  - `store_refresh_token(user_id: str, token_hash: str, expires_at: datetime) -> None`
  - `consume_refresh_token(token_hash: str) -> str | None` (returns user id and revokes it; `None` if unknown/expired/already revoked)
  - `user_for_revoked_token(token_hash: str) -> str | None` (who owned an already-revoked token — used to detect replay)
  - `revoke_all_refresh_tokens(user_id: str) -> None`
  - `save_credential(user_id: str, provider: str, payload: dict) -> None`
  - `get_credential(user_id: str, provider: str) -> dict | None`
  - `list_credential_providers(user_id: str) -> list[str]`
  - `delete_credential(user_id: str, provider: str) -> bool`
  - exception class `EmailTaken`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_auth_store.py`:

```python
import base64
import os
from datetime import UTC, datetime, timedelta

import pytest

from pos import config
from pos.auth.store import EmailTaken, UserStore
from pos.db import create_pool, init_schema

_DSN = os.environ.get("TEST_DATABASE_URL", "postgresql://pos:pos@localhost:5432/pos")


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setattr(config, "ENCRYPTION_KEY", base64.b64encode(os.urandom(32)).decode())


def _store():
    pool = create_pool(_DSN, max_size=5)
    init_schema(pool)
    with pool.connection() as conn:
        conn.execute("TRUNCATE users RESTART IDENTITY CASCADE")
    return UserStore(pool)


def test_create_and_fetch_user():
    store = _store()
    user = store.create_user("a@test", "hash-a")

    assert user["email"] == "a@test"
    assert store.get_user_by_email("a@test")["id"] == user["id"]
    assert store.get_user_by_id(user["id"])["email"] == "a@test"


def test_duplicate_email_raises():
    store = _store()
    store.create_user("a@test", "hash-a")
    with pytest.raises(EmailTaken):
        store.create_user("a@test", "hash-b")


def test_email_is_matched_case_insensitively():
    store = _store()
    store.create_user("a@test", "hash-a")
    assert store.get_user_by_email("A@TEST") is not None


def test_unknown_user_lookups_return_none():
    store = _store()
    assert store.get_user_by_email("nobody@test") is None


def test_refresh_token_is_single_use():
    store = _store()
    user = store.create_user("a@test", "h")
    expires = datetime.now(UTC) + timedelta(days=30)
    store.store_refresh_token(user["id"], "hash-1", expires)

    assert store.consume_refresh_token("hash-1") == user["id"]
    assert store.consume_refresh_token("hash-1") is None


def test_expired_refresh_token_is_rejected():
    store = _store()
    user = store.create_user("a@test", "h")
    store.store_refresh_token(user["id"], "hash-old", datetime.now(UTC) - timedelta(seconds=1))

    assert store.consume_refresh_token("hash-old") is None


def test_user_for_revoked_token_identifies_a_replay():
    store = _store()
    user = store.create_user("a@test", "h")
    expires = datetime.now(UTC) + timedelta(days=30)
    store.store_refresh_token(user["id"], "hash-1", expires)
    store.consume_refresh_token("hash-1")

    assert store.user_for_revoked_token("hash-1") == user["id"]
    assert store.user_for_revoked_token("never-existed") is None


def test_revoke_all_invalidates_every_token_for_that_user():
    store = _store()
    user = store.create_user("a@test", "h")
    expires = datetime.now(UTC) + timedelta(days=30)
    store.store_refresh_token(user["id"], "hash-1", expires)
    store.store_refresh_token(user["id"], "hash-2", expires)

    store.revoke_all_refresh_tokens(user["id"])

    assert store.consume_refresh_token("hash-1") is None
    assert store.consume_refresh_token("hash-2") is None


def test_credentials_round_trip_encrypted():
    store = _store()
    user = store.create_user("a@test", "h")

    store.save_credential(user["id"], "openai", {"OPENAI_API_KEY": "sk-1"})

    assert store.get_credential(user["id"], "openai") == {"OPENAI_API_KEY": "sk-1"}
    assert store.list_credential_providers(user["id"]) == ["openai"]


def test_saving_the_same_provider_twice_updates_in_place():
    store = _store()
    user = store.create_user("a@test", "h")

    store.save_credential(user["id"], "openai", {"OPENAI_API_KEY": "sk-1"})
    store.save_credential(user["id"], "openai", {"OPENAI_API_KEY": "sk-2"})

    assert store.get_credential(user["id"], "openai") == {"OPENAI_API_KEY": "sk-2"}
    assert store.list_credential_providers(user["id"]) == ["openai"]


def test_plaintext_is_never_written_to_the_table():
    store = _store()
    user = store.create_user("a@test", "h")
    store.save_credential(user["id"], "openai", {"OPENAI_API_KEY": "sk-secret"})

    with store._pool.connection() as conn:
        row = conn.execute("SELECT encrypted_payload FROM api_credentials").fetchone()
    assert b"sk-secret" not in bytes(row["encrypted_payload"])


def test_one_user_cannot_read_anothers_credentials():
    store = _store()
    a = store.create_user("a@test", "h")
    b = store.create_user("b@test", "h")
    store.save_credential(a["id"], "openai", {"OPENAI_API_KEY": "sk-a"})

    assert store.get_credential(b["id"], "openai") is None
    assert store.list_credential_providers(b["id"]) == []


def test_delete_credential():
    store = _store()
    user = store.create_user("a@test", "h")
    store.save_credential(user["id"], "openai", {"OPENAI_API_KEY": "sk-1"})

    assert store.delete_credential(user["id"], "openai") is True
    assert store.delete_credential(user["id"], "openai") is False
    assert store.get_credential(user["id"], "openai") is None
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_auth_store.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pos.auth.store'`.

- [ ] **Step 3: Write the implementation**

Create `backend/src/pos/auth/store.py`:

```python
"""Users, refresh tokens, and encrypted provider credentials.

Kept separate from SessionStore: identity is read on every authenticated
request, session history only when the UI asks for it, and mixing them
would make one class own two unrelated lifecycles."""

from __future__ import annotations

from datetime import datetime

import psycopg
from psycopg_pool import ConnectionPool

from pos.auth.crypto import decrypt_payload, encrypt_payload


class EmailTaken(Exception):
    pass


class UserStore:
    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def create_user(self, email: str, password_hash: str) -> dict:
        try:
            with self._pool.connection() as conn:
                return conn.execute(
                    "INSERT INTO users (email, password_hash) VALUES (%s, %s) "
                    "RETURNING id::text, email, created_at",
                    (email.lower(), password_hash),
                ).fetchone()
        except psycopg.errors.UniqueViolation as e:
            raise EmailTaken(email) from e

    def get_user_by_email(self, email: str) -> dict | None:
        with self._pool.connection() as conn:
            return conn.execute(
                "SELECT id::text, email, password_hash FROM users WHERE email = %s",
                (email.lower(),),
            ).fetchone()

    def get_user_by_id(self, user_id: str) -> dict | None:
        with self._pool.connection() as conn:
            return conn.execute(
                "SELECT id::text, email FROM users WHERE id = %s", (user_id,)
            ).fetchone()

    def store_refresh_token(self, user_id: str, token_hash: str, expires_at: datetime) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO refresh_tokens (user_id, token_hash, expires_at) VALUES (%s, %s, %s)",
                (user_id, token_hash, expires_at),
            )

    def consume_refresh_token(self, token_hash: str) -> str | None:
        """Single-use: revokes the token as it's read, so replaying one is
        always rejected."""
        with self._pool.connection() as conn:
            row = conn.execute(
                "UPDATE refresh_tokens SET revoked_at = now() "
                "WHERE token_hash = %s AND revoked_at IS NULL AND expires_at > now() "
                "RETURNING user_id::text",
                (token_hash,),
            ).fetchone()
        return row["user_id"] if row else None

    def user_for_revoked_token(self, token_hash: str) -> str | None:
        """Who owned a token that has already been revoked. A client
        presenting one is replaying a stolen or stale token."""
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT user_id::text FROM refresh_tokens "
                "WHERE token_hash = %s AND revoked_at IS NOT NULL",
                (token_hash,),
            ).fetchone()
        return row["user_id"] if row else None

    def revoke_all_refresh_tokens(self, user_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE refresh_tokens SET revoked_at = now() "
                "WHERE user_id = %s AND revoked_at IS NULL",
                (user_id,),
            )

    def save_credential(self, user_id: str, provider: str, payload: dict) -> None:
        ciphertext, nonce = encrypt_payload(payload)
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO api_credentials (user_id, provider, encrypted_payload, nonce) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (user_id, provider) DO UPDATE "
                "SET encrypted_payload = EXCLUDED.encrypted_payload, "
                "    nonce = EXCLUDED.nonce, updated_at = now()",
                (user_id, provider, ciphertext, nonce),
            )

    def get_credential(self, user_id: str, provider: str) -> dict | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT encrypted_payload, nonce FROM api_credentials "
                "WHERE user_id = %s AND provider = %s",
                (user_id, provider),
            ).fetchone()
        if row is None:
            return None
        return decrypt_payload(bytes(row["encrypted_payload"]), bytes(row["nonce"]))

    def list_credential_providers(self, user_id: str) -> list[str]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT provider FROM api_credentials WHERE user_id = %s ORDER BY provider",
                (user_id,),
            ).fetchall()
        return [r["provider"] for r in rows]

    def delete_credential(self, user_id: str, provider: str) -> bool:
        with self._pool.connection() as conn:
            cursor = conn.execute(
                "DELETE FROM api_credentials WHERE user_id = %s AND provider = %s",
                (user_id, provider),
            )
            return cursor.rowcount > 0
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_auth_store.py -v
```

Expected: PASS (13 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/src/pos/auth/store.py backend/tests/test_auth_store.py
git commit -m "feat: add UserStore for users, refresh tokens, and credentials"
```

---

### Task 8: Auth routes and the current-user dependency

**Files:**
- Create: `backend/src/pos/auth/routes.py`, `backend/src/pos/auth/deps.py`
- Test: `backend/tests/test_auth_routes.py`

**Interfaces:**
- Consumes: `UserStore`, `passwords`, `tokens`, `config`.
- Produces:
  - `pos.auth.deps.user_id_from_request(request) -> str | None` — reads the access-token cookie and returns the user id.
  - `pos.auth.deps.require_user_id(request) -> str` — FastAPI dependency; raises 401.
  - `pos.auth.deps.set_auth_cookies(response, access, refresh)`, `clear_auth_cookies(response)`.
  - `pos.auth.deps.ACCESS_COOKIE = "pos_access"`, `REFRESH_COOKIE = "pos_refresh"`.
  - `pos.auth.routes.build_auth_router(users: UserStore) -> APIRouter` mounting `/auth/signup`, `/auth/login`, `/auth/logout`, `/auth/refresh`, `/auth/me`, and `/credentials*`.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_auth_routes.py`:

```python
import base64
import os

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from pos import config
from pos.auth.routes import build_auth_router
from pos.auth.store import UserStore
from pos.db import create_pool, init_schema

_DSN = os.environ.get("TEST_DATABASE_URL", "postgresql://pos:pos@localhost:5432/pos")


@pytest.fixture(autouse=True)
def _secrets(monkeypatch):
    monkeypatch.setattr(config, "JWT_SECRET", "test-secret")
    monkeypatch.setattr(config, "ENCRYPTION_KEY", base64.b64encode(os.urandom(32)).decode())


def _client():
    pool = create_pool(_DSN, max_size=5)
    init_schema(pool)
    with pool.connection() as conn:
        conn.execute("TRUNCATE users RESTART IDENTITY CASCADE")
    app = FastAPI()
    app.include_router(build_auth_router(UserStore(pool)))
    return TestClient(app)


def test_signup_sets_cookies_and_returns_the_user():
    client = _client()
    resp = client.post("/auth/signup", json={"email": "a@test", "password": "pw-12345678"})

    assert resp.status_code == 200
    assert resp.json()["email"] == "a@test"
    assert "pos_access" in resp.cookies
    assert "pos_refresh" in resp.cookies


def test_signup_never_returns_the_password_or_its_hash():
    client = _client()
    body = client.post("/auth/signup", json={"email": "a@test", "password": "pw-12345678"}).text
    assert "pw-12345678" not in body
    assert "argon2" not in body


def test_duplicate_signup_is_rejected():
    client = _client()
    client.post("/auth/signup", json={"email": "a@test", "password": "pw-12345678"})
    resp = client.post("/auth/signup", json={"email": "a@test", "password": "pw-12345678"})
    assert resp.status_code == 409


def test_short_password_is_rejected():
    client = _client()
    resp = client.post("/auth/signup", json={"email": "a@test", "password": "short"})
    assert resp.status_code == 422


def test_login_with_the_right_password_succeeds():
    client = _client()
    client.post("/auth/signup", json={"email": "a@test", "password": "pw-12345678"})
    client.cookies.clear()

    resp = client.post("/auth/login", json={"email": "a@test", "password": "pw-12345678"})

    assert resp.status_code == 200
    assert "pos_access" in resp.cookies


def test_login_with_the_wrong_password_fails():
    client = _client()
    client.post("/auth/signup", json={"email": "a@test", "password": "pw-12345678"})
    resp = client.post("/auth/login", json={"email": "a@test", "password": "wrong-password"})
    assert resp.status_code == 401


def test_login_for_an_unknown_email_fails_the_same_way():
    client = _client()
    resp = client.post("/auth/login", json={"email": "nobody@test", "password": "pw-12345678"})
    assert resp.status_code == 401


def test_me_requires_authentication():
    client = _client()
    assert client.get("/auth/me").status_code == 401


def test_me_returns_the_signed_in_user():
    client = _client()
    client.post("/auth/signup", json={"email": "a@test", "password": "pw-12345678"})
    assert client.get("/auth/me").json()["email"] == "a@test"


def test_refresh_rotates_the_token_and_the_old_one_stops_working():
    client = _client()
    client.post("/auth/signup", json={"email": "a@test", "password": "pw-12345678"})
    first_refresh = client.cookies["pos_refresh"]

    assert client.post("/auth/refresh").status_code == 200

    client.cookies.set("pos_refresh", first_refresh)
    assert client.post("/auth/refresh").status_code == 401


def test_replaying_an_old_refresh_token_kills_the_whole_family():
    client = _client()
    client.post("/auth/signup", json={"email": "a@test", "password": "pw-12345678"})
    stolen = client.cookies["pos_refresh"]
    client.post("/auth/refresh")
    live = client.cookies["pos_refresh"]

    client.cookies.set("pos_refresh", stolen)
    assert client.post("/auth/refresh").status_code == 401

    client.cookies.set("pos_refresh", live)
    assert client.post("/auth/refresh").status_code == 401


def test_logout_clears_cookies_and_ends_the_session():
    client = _client()
    client.post("/auth/signup", json={"email": "a@test", "password": "pw-12345678"})

    assert client.post("/auth/logout").status_code == 200

    assert client.get("/auth/me").status_code == 401


def test_credentials_are_per_user_and_never_returned():
    client = _client()
    client.post("/auth/signup", json={"email": "a@test", "password": "pw-12345678"})
    client.put("/credentials/openai", json={"openai_api_key": "sk-secret"})

    listed = client.get("/credentials")
    assert listed.json() == {"configured": ["openai"]}
    assert "sk-secret" not in listed.text

    client.post("/auth/logout")
    client.post("/auth/signup", json={"email": "b@test", "password": "pw-12345678"})
    assert client.get("/credentials").json() == {"configured": []}


def test_delete_credential():
    client = _client()
    client.post("/auth/signup", json={"email": "a@test", "password": "pw-12345678"})
    client.put("/credentials/openai", json={"openai_api_key": "sk-secret"})

    assert client.delete("/credentials/openai").status_code == 200
    assert client.get("/credentials").json() == {"configured": []}
    assert client.delete("/credentials/openai").status_code == 404


def test_credentials_require_authentication():
    client = _client()
    assert client.get("/credentials").status_code == 401
    assert client.put("/credentials/openai", json={"openai_api_key": "x"}).status_code == 401
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_auth_routes.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pos.auth.routes'`.

- [ ] **Step 3: Write the dependency helpers**

Create `backend/src/pos/auth/deps.py`:

```python
"""Cookie handling and the current-user dependency.

The access token travels in an httpOnly cookie rather than a header or
localStorage: script-readable storage hands the token to any injected
script, and a query string would land in uvicorn's access log."""

from __future__ import annotations

from fastapi import HTTPException, Request, Response

from pos import config
from pos.auth.tokens import ACCESS_TOKEN_TTL, REFRESH_TOKEN_TTL, decode_access_token

ACCESS_COOKIE = "pos_access"
REFRESH_COOKIE = "pos_refresh"


def _set(response: Response, name: str, value: str, max_age: int) -> None:
    response.set_cookie(
        key=name,
        value=value,
        max_age=max_age,
        httponly=True,
        secure=config.COOKIE_SECURE,
        samesite="lax",
        path="/",
    )


def set_auth_cookies(response: Response, access_token: str, refresh_token: str) -> None:
    _set(response, ACCESS_COOKIE, access_token, int(ACCESS_TOKEN_TTL.total_seconds()))
    _set(response, REFRESH_COOKIE, refresh_token, int(REFRESH_TOKEN_TTL.total_seconds()))


def clear_auth_cookies(response: Response) -> None:
    for name in (ACCESS_COOKIE, REFRESH_COOKIE):
        response.delete_cookie(name, path="/")


def user_id_from_request(request: Request) -> str | None:
    token = request.cookies.get(ACCESS_COOKIE)
    return decode_access_token(token) if token else None


def require_user_id(request: Request) -> str:
    user_id = user_id_from_request(request)
    if user_id is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    return user_id
```

- [ ] **Step 4: Write the routes**

Create `backend/src/pos/auth/routes.py`:

```python
"""/auth/* and /credentials/* endpoints.

Kept out of cli/server.py, which is already large and owns the realtime
transport rather than account management."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, EmailStr, Field

from pos.auth.deps import (
    REFRESH_COOKIE,
    clear_auth_cookies,
    require_user_id,
    set_auth_cookies,
)
from pos.auth.passwords import hash_password, verify_password
from pos.auth.store import EmailTaken, UserStore
from pos.auth.tokens import (
    REFRESH_TOKEN_TTL,
    create_access_token,
    hash_refresh_token,
    new_refresh_token,
)

# Which request fields belong to which provider, and what environment
# variable each one becomes. Mirrors the mapping cli/server.py used for
# the old per-connection key_token flow.
PROVIDER_FIELDS: dict[str, dict[str, str]] = {
    "local": {"local_api_key": "LOCAL_API_KEY", "local_base_url": "LOCAL_BASE_URL"},
    "openai": {"openai_api_key": "OPENAI_API_KEY"},
    "azure": {
        "azure_api_key": "AZURE_OPENAI_API_KEY",
        "azure_endpoint": "AZURE_OPENAI_ENDPOINT",
        "azure_deployment": "AZURE_OPENAI_DEPLOYMENT",
    },
    "anthropic": {"anthropic_api_key": "ANTHROPIC_API_KEY"},
    "gemini": {"gemini_api_key": "GOOGLE_API_KEY"},
    "bedrock": {
        "bedrock_access_key_id": "AWS_ACCESS_KEY_ID",
        "bedrock_secret_access_key": "AWS_SECRET_ACCESS_KEY",
        "bedrock_region": "AWS_REGION",
    },
    "openrouter": {"openrouter_api_key": "OPENROUTER_API_KEY"},
    "tavily": {"tavily_api_key": "TAVILY_API_KEY"},
}


class Credentials(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


def build_auth_router(users: UserStore) -> APIRouter:
    router = APIRouter()

    def _issue(response: Response, user_id: str) -> None:
        raw_refresh, refresh_hash = new_refresh_token()
        users.store_refresh_token(user_id, refresh_hash, datetime.now(UTC) + REFRESH_TOKEN_TTL)
        set_auth_cookies(response, create_access_token(user_id), raw_refresh)

    @router.post("/auth/signup")
    async def signup(body: Credentials, response: Response):
        try:
            user = users.create_user(body.email, hash_password(body.password))
        except EmailTaken as e:
            raise HTTPException(status_code=409, detail="email already registered") from e
        _issue(response, user["id"])
        return {"id": user["id"], "email": user["email"]}

    @router.post("/auth/login")
    async def login(body: Credentials, response: Response):
        user = users.get_user_by_email(body.email)
        # Same 401 for an unknown email and a wrong password: distinguishing
        # them tells an attacker which addresses are registered.
        if user is None or not verify_password(body.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="invalid email or password")
        _issue(response, user["id"])
        return {"id": user["id"], "email": user["email"]}

    @router.post("/auth/logout")
    async def logout(request: Request, response: Response):
        raw = request.cookies.get(REFRESH_COOKIE)
        if raw:
            users.consume_refresh_token(hash_refresh_token(raw))
        clear_auth_cookies(response)
        return {"ok": True}

    @router.post("/auth/refresh")
    async def refresh(request: Request, response: Response):
        raw = request.cookies.get(REFRESH_COOKIE)
        token_hash = hash_refresh_token(raw) if raw else None
        user_id = users.consume_refresh_token(token_hash) if token_hash else None
        if user_id is None:
            # Presenting an already-revoked token means someone is replaying a
            # stolen or stale one. Which of the two it is can't be told apart,
            # so drop the whole family and make everyone sign in again.
            replayed_by = users.user_for_revoked_token(token_hash) if token_hash else None
            if replayed_by is not None:
                users.revoke_all_refresh_tokens(replayed_by)
                print(f"  refresh token replay detected for user {replayed_by} -- all tokens revoked")
            clear_auth_cookies(response)
            raise HTTPException(status_code=401, detail="invalid refresh token")
        _issue(response, user_id)
        return {"ok": True}

    @router.get("/auth/me")
    async def me(user_id: str = Depends(require_user_id)):
        user = users.get_user_by_id(user_id)
        if user is None:
            raise HTTPException(status_code=401, detail="not authenticated")
        return user

    @router.get("/credentials")
    async def list_credentials(user_id: str = Depends(require_user_id)):
        return {"configured": users.list_credential_providers(user_id)}

    @router.put("/credentials/{provider}")
    async def put_credential(
        provider: str, body: dict, user_id: str = Depends(require_user_id)
    ):
        fields = PROVIDER_FIELDS.get(provider)
        if fields is None:
            raise HTTPException(status_code=404, detail=f"unknown provider {provider!r}")
        payload = {
            env_name: str(body[field]).strip()
            for field, env_name in fields.items()
            if body.get(field) and str(body[field]).strip()
        }
        if not payload:
            raise HTTPException(status_code=422, detail="no credential fields provided")
        users.save_credential(user_id, provider, payload)
        return {"saved": provider}

    @router.delete("/credentials/{provider}")
    async def delete_credential(provider: str, user_id: str = Depends(require_user_id)):
        if not users.delete_credential(user_id, provider):
            raise HTTPException(status_code=404, detail="no stored credential for that provider")
        return {"deleted": provider}

    return router
```

- [ ] **Step 5: Add the `EmailStr` dependency**

`EmailStr` needs `email-validator`. Add to `backend/pyproject.toml` `dependencies`:

```toml
    "email-validator>=2.2.0",
```

Then: `uv lock && uv sync`

- [ ] **Step 6: Run the tests to verify they pass**

```bash
uv run pytest tests/test_auth_routes.py -v
```

Expected: PASS (15 passed).

- [ ] **Step 7: Commit**

```bash
git add backend/pyproject.toml backend/uv.lock backend/src/pos/auth/routes.py backend/src/pos/auth/deps.py backend/tests/test_auth_routes.py
git commit -m "feat: add auth and credentials endpoints"
```

---

### Task 9: Wire auth into the server and scope every session endpoint

**Files:**
- Modify: `backend/src/pos/cli/server.py`
- Modify: `backend/src/pos/cli/local.py`
- Modify: `backend/tests/test_server.py`

**Interfaces:**
- Consumes: everything from Tasks 1-8.
- Produces: `create_app(..., session_store=None, user_store=None)` — both default to stores built on a pool from `pos.db`.

- [ ] **Step 1: Update the server**

In `backend/src/pos/cli/server.py`:

1. Replace the `SessionKeysRequest` class and the `@app.post("/session-keys")` handler, the `_KEY_TOKEN_TTL_SECONDS`/`_key_tokens`/`_prune_expired_key_tokens` block, and the `key_token` query-param handling inside `ws_endpoint` — all of it is superseded by stored credentials.

2. Update the imports:

```python
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from pos.auth.deps import require_user_id, user_id_from_request
from pos.auth.routes import PROVIDER_FIELDS, build_auth_router
from pos.auth.store import UserStore
from pos.db import create_pool, init_schema
```

3. Change the `create_app` signature's store arguments to:

```python
    session_store: SessionStore | None = None,
    user_store: UserStore | None = None,
```

4. Replace the `store = session_store or SessionStore(config.DATABASE_URL)` line with:

```python
    if session_store is None or user_store is None:
        pool = create_pool()
        init_schema(pool)
        session_store = session_store or SessionStore(pool)
        user_store = user_store or UserStore(pool)
    store = session_store
    users = user_store
```

5. After `app = FastAPI()`, add CORS and the router:

```python
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(build_auth_router(users))
```

`allow_origins` must be an explicit list — a browser rejects
`allow_credentials` combined with `*`.

6. Scope the three session endpoints:

```python
    @app.get("/sessions")
    async def list_sessions(user_id: str = Depends(require_user_id)):
        return store.list_sessions(user_id)

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str, user_id: str = Depends(require_user_id)):
        result = store.get_session(session_id, user_id)
        if result is None:
            raise HTTPException(status_code=404, detail="session not found")
        return result

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str, user_id: str = Depends(require_user_id)):
        if not store.delete_session(session_id, user_id):
            raise HTTPException(status_code=404, detail="session not found")
        return {"deleted": True}
```

A session owned by someone else returns 404 rather than 403 — a 403 would
confirm the id exists.

7. Delete the `@app.get("/")` `FileResponse` handler and the `Path`/`FileResponse`
imports. The browser client is now the separate Next.js app in `frontend/`;
`backend/src/pos/static/index.html` stays on disk for reference but is no
longer served. **Also delete `test_index_page_is_served` from
`backend/tests/test_server.py` (around line 1108)** — it asserts that route
exists and will fail otherwise.

8. At the top of `ws_endpoint`, right after `await websocket.accept()`, authenticate:

```python
        user_id = user_id_from_request(websocket)
        if user_id is None:
            await websocket.send_json({"event": "error", "message": "not authenticated"})
            await websocket.close(code=1008)
            return
```

`user_id_from_request` reads `request.cookies`, which `WebSocket` also
provides, so the same helper works for both.

9. Replace the removed `env_overrides` block with stored credentials, placed
where the old `key_token` handling was:

```python
        stored = users.get_credential(user_id, params.get("provider", "local"))
        tavily = users.get_credential(user_id, "tavily")
        env_overrides = {**(stored or {}), **(tavily or {})} or None
```

10. Scope session creation and resume. In `ws_endpoint`, the `store.get_session`
call for `resume_session_id` becomes `store.get_session(resume_session_id, user_id)`,
and `store.create_session(...)` gains `user_id`:

```python
                store.create_session(
                    session_id, user_id, mode=mode,
                    tts_engine=stored_tts_engine, llm_model=llm_model,
                )
```

- [ ] **Step 2: Update the local CLI**

`cli/local.py` has no logged-in user. Give it a dedicated one so its
sessions satisfy the foreign key. Replace its `SessionStore(config.DATABASE_URL)`
line with:

```python
    pool = create_pool()
    init_schema(pool)
    users = UserStore(pool)
    cli_user = users.get_user_by_email("cli@localhost")
    if cli_user is None:
        cli_user = users.create_user("cli@localhost", hash_password(secrets.token_urlsafe(32)))
    session_store = SessionStore(pool)
    session_id = uuid.uuid4().hex
    session_store.create_session(
        session_id, cli_user["id"], mode="voice", tts_engine=args.tts, llm_model="lfm2.5-230m"
    )
```

with the matching imports (`secrets`, `create_pool`, `init_schema`, `UserStore`,
`hash_password`). The random password is never used to log in — the account
exists only to own CLI sessions.

- [ ] **Step 3: Update the server tests**

In `backend/tests/test_server.py`, replace `_fresh_store()` with a helper that
builds both stores plus an authenticated client:

```python
import base64

from pos.auth.store import UserStore
from pos.db import create_pool, init_schema

_TEST_DSN = os.environ.get("TEST_DATABASE_URL", "postgresql://pos:pos@localhost:5432/pos")


def _fresh_stores():
    pool = create_pool(_TEST_DSN, max_size=5)
    init_schema(pool)
    with pool.connection() as conn:
        conn.execute("TRUNCATE users, sessions, turns RESTART IDENTITY CASCADE")
    return SessionStore(pool), UserStore(pool)


def _sign_in(client, email="a@test"):
    resp = client.post("/auth/signup", json={"email": email, "password": "pw-12345678"})
    assert resp.status_code == 200
    return resp.json()["id"]
```

Add an autouse fixture at module level so the secrets exist for every test:

```python
@pytest.fixture(autouse=True)
def _secrets(monkeypatch):
    monkeypatch.setattr(config, "JWT_SECRET", "test-secret")
    monkeypatch.setattr(config, "ENCRYPTION_KEY", base64.b64encode(os.urandom(32)).decode())
```

Then update every `create_app(...)` call: `session_store=...`/`user_store=...`
from `_fresh_stores()`, and every test that opens a websocket or calls
`/sessions` must call `_sign_in(client)` first. Every `create_session(...)`
call in the file gains the user id returned by `_sign_in`.

Add these new tests:

```python
def test_sessions_endpoints_require_authentication(monkeypatch):
    client, _ = _make_client(monkeypatch)
    assert client.get("/sessions").status_code == 401
    assert client.get("/sessions/anything").status_code == 401
    assert client.delete("/sessions/anything").status_code == 401


def test_websocket_requires_authentication(monkeypatch):
    client, _ = _make_client(monkeypatch)
    with client.websocket_connect("/ws") as ws:
        message = ws.receive_json()
    assert message["event"] == "error"
    assert "not authenticated" in message["message"]


def test_one_user_cannot_see_anothers_sessions(monkeypatch):
    client, _ = _make_client(monkeypatch)
    _sign_in(client, "a@test")
    with client.websocket_connect("/ws?mode=text") as ws:
        ws.receive_json()
    assert len(client.get("/sessions").json()) == 1

    client.post("/auth/logout")
    _sign_in(client, "b@test")
    assert client.get("/sessions").json() == []
```

- [ ] **Step 4: Run the whole backend suite**

```bash
uv run pytest -v
```

Expected: PASS. `test_server.py` is slow (websocket + thread timing); allow
several minutes. If a websocket test hangs, check that `_sign_in` ran before
`websocket_connect` — an unauthenticated socket now closes immediately.

- [ ] **Step 5: Lint**

```bash
uv run ruff check .
```

Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add backend/src/pos/cli/server.py backend/src/pos/cli/local.py backend/tests/test_server.py
git commit -m "feat: require auth and scope sessions per user"
```

---

### Task 10: Environment and Compose wiring

**Files:**
- Modify: `docker-compose.yml`, `.env.example`, `README.md`

- [ ] **Step 1: Generate and document the secrets**

Add to `.env.example`:

```
# openssl rand -base64 32
JWT_SECRET=replace-me-with-a-random-32-byte-secret
# openssl rand -base64 32   (must decode to exactly 32 bytes)
ENCRYPTION_KEY=replace-me-with-a-random-base64-32-byte-key

COOKIE_SECURE=false
CORS_ORIGINS=http://localhost:3000
DB_POOL_MAX_SIZE=10
```

- [ ] **Step 2: Pass them to the backend service**

In `docker-compose.yml`, under `backend.environment`, add:

```yaml
      JWT_SECRET: ${JWT_SECRET:?JWT_SECRET must be set - see .env.example}
      ENCRYPTION_KEY: ${ENCRYPTION_KEY:?ENCRYPTION_KEY must be set - see .env.example}
      COOKIE_SECURE: ${COOKIE_SECURE:-false}
      CORS_ORIGINS: ${CORS_ORIGINS:-http://localhost:3000}
      DB_POOL_MAX_SIZE: ${DB_POOL_MAX_SIZE:-10}
```

The `:?` form makes Compose fail loudly rather than starting a server with an
empty signing key.

- [ ] **Step 3: Document it**

In `README.md`, under "Running with Docker Compose", add before the
`docker compose up` command:

```markdown
Copy `.env.example` to `.env` and fill in `JWT_SECRET` and `ENCRYPTION_KEY`
(both `openssl rand -base64 32`). The stack will not start without them.
Losing `ENCRYPTION_KEY` makes every stored API credential unrecoverable.
```

- [ ] **Step 4: Verify the stack starts**

```bash
cd .. && docker compose up -d --build && docker compose logs backend --tail 30
```

Expected: backend starts with no traceback. Then:

```bash
curl -i -X POST http://localhost:8000/auth/signup \
  -H 'Content-Type: application/json' \
  -d '{"email":"you@example.com","password":"pw-12345678"}'
```

Expected: `200` with two `set-cookie` headers.

- [ ] **Step 5: Commit**

```bash
git add docker-compose.yml .env.example README.md
git commit -m "feat: wire auth secrets through Compose"
```

---

### Task 11: Frontend auth client and login/signup pages

**Files:**
- Create: `frontend/src/lib/auth.ts`, `frontend/src/app/login/page.tsx`, `frontend/src/app/signup/page.tsx`
- Modify: `frontend/src/lib/api.ts`

**Interfaces:**
- Produces: `signup`, `login`, `logout`, `fetchMe` from `@/lib/auth`; `CurrentUser` type `{ id: string; email: string }`.

- [ ] **Step 1: Send cookies on every request**

Every `fetch` in `frontend/src/lib/api.ts` needs `credentials: "include"`, or
the browser will not attach the auth cookie cross-origin. Update each call,
e.g.:

```ts
const res = await fetch(`${API_URL}/options`, { credentials: "include" });
```

Do the same for `fetchSessions`, `fetchSession`, `deleteSession`, and delete
`fetchKeyTokenIfNeeded` entirely — the `key_token` flow no longer exists.
Also delete `keyProviderValidationError`'s use from `page.tsx` in Task 13.

- [ ] **Step 2: Write the auth client**

Create `frontend/src/lib/auth.ts`:

```ts
import { API_URL } from "./config";

export interface CurrentUser {
  id: string;
  email: string;
}

async function post(path: string, body?: unknown): Promise<Response> {
  return fetch(`${API_URL}${path}`, {
    method: "POST",
    credentials: "include",
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
}

export async function signup(email: string, password: string): Promise<CurrentUser> {
  const res = await post("/auth/signup", { email, password });
  if (res.status === 409) throw new Error("That email is already registered.");
  if (res.status === 422) throw new Error("Password must be at least 8 characters.");
  if (!res.ok) throw new Error("Couldn't create your account. Try again.");
  return res.json();
}

export async function login(email: string, password: string): Promise<CurrentUser> {
  const res = await post("/auth/login", { email, password });
  if (res.status === 401) throw new Error("Invalid email or password.");
  if (!res.ok) throw new Error("Couldn't sign you in. Try again.");
  return res.json();
}

export async function logout(): Promise<void> {
  await post("/auth/logout");
}

export async function fetchMe(): Promise<CurrentUser | null> {
  const res = await fetch(`${API_URL}/auth/me`, { credentials: "include" });
  if (res.status === 401) return null;
  if (!res.ok) throw new Error("Couldn't load your account.");
  return res.json();
}
```

- [ ] **Step 3: Build the two pages**

Create `frontend/src/app/login/page.tsx`:

```tsx
"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import Link from "next/link";
import { login } from "@/lib/auth";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await login(email, password);
      router.replace("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center bg-bg px-4">
      <form onSubmit={onSubmit} className="w-full max-w-sm">
        <h1 className="mb-6 text-xl font-semibold text-text">Sign in</h1>
        <label className="mb-1 block text-[12.5px] text-text-muted" htmlFor="email">Email</label>
        <input
          id="email" type="email" required autoComplete="email"
          value={email} onChange={(e) => setEmail(e.target.value)}
          className="mb-4 w-full rounded-lg border border-transparent bg-surface-sunken px-2.5 py-2 text-[13px] text-text outline-none focus:border-accent"
        />
        <label className="mb-1 block text-[12.5px] text-text-muted" htmlFor="password">Password</label>
        <input
          id="password" type="password" required autoComplete="current-password"
          value={password} onChange={(e) => setPassword(e.target.value)}
          className="mb-4 w-full rounded-lg border border-transparent bg-surface-sunken px-2.5 py-2 text-[13px] text-text outline-none focus:border-accent"
        />
        {error && <p className="mb-3 text-[12.5px] text-danger">{error}</p>}
        <button
          type="submit" disabled={busy}
          className="w-full rounded-lg bg-accent px-3 py-2 text-[13px] font-medium text-white hover:bg-accent-hover disabled:opacity-50"
        >
          {busy ? "Signing in…" : "Sign in"}
        </button>
        <p className="mt-4 text-[12.5px] text-text-faint">
          No account? <Link href="/signup" className="text-accent">Create one</Link>
        </p>
      </form>
    </main>
  );
}
```

Create `frontend/src/app/signup/page.tsx` as the same component with these
differences: import `signup` instead of `login` and call it; heading
"Create account"; `autoComplete="new-password"`; button text
"Create account" / "Creating…"; footer link `Already have an account?`
pointing to `/login`.

- [ ] **Step 4: Verify the pages build and render**

```bash
cd frontend && npx tsc --noEmit && npm run build
```

Expected: no type errors, build succeeds.

With the backend running (`docker compose up -d`), start the frontend
(`npm run dev`), open `http://localhost:3000/signup`, create an account, and
confirm you are redirected to `/`. In DevTools → Application → Cookies, confirm
`pos_access` and `pos_refresh` exist and are marked HttpOnly.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/auth.ts frontend/src/lib/api.ts frontend/src/app/login frontend/src/app/signup
git commit -m "feat: add frontend auth client and login/signup pages"
```

---

### Task 12: Auth guard and account menu

**Files:**
- Create: `frontend/src/components/AuthGuard.tsx`
- Modify: `frontend/src/app/page.tsx`, `frontend/src/components/Sidebar.tsx`

- [ ] **Step 1: Write the guard**

Create `frontend/src/components/AuthGuard.tsx`:

```tsx
"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { type CurrentUser, fetchMe } from "@/lib/auth";

export function AuthGuard({ children }: { children: (user: CurrentUser) => React.ReactNode }) {
  const router = useRouter();
  const [user, setUser] = useState<CurrentUser | null>(null);
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    let active = true;
    fetchMe()
      .then((me) => {
        if (!active) return;
        if (me === null) router.replace("/login");
        else setUser(me);
      })
      .catch(() => router.replace("/login"))
      .finally(() => active && setChecked(true));
    return () => {
      active = false;
    };
  }, [router]);

  if (!checked || !user) {
    return <div className="flex min-h-screen items-center justify-center bg-bg text-[13px] text-text-faint">Loading…</div>;
  }
  return <>{children(user)}</>;
}
```

- [ ] **Step 2: Wrap the app**

In `frontend/src/app/page.tsx`, rename the existing default export to
`VoiceAgent`, give it a `user: CurrentUser` prop, and add:

```tsx
export default function Home() {
  return <AuthGuard>{(user) => <VoiceAgent user={user} />}</AuthGuard>;
}
```

with `import { AuthGuard } from "@/components/AuthGuard";` and
`import type { CurrentUser } from "@/lib/auth";`.

- [ ] **Step 3: Add the account menu**

In `frontend/src/components/Sidebar.tsx`, add `userEmail: string` to
`SidebarProps` and render this as the last child of the sidebar container:

```tsx
      <div className="mt-auto border-t border-border px-2 pt-2">
        <div className="truncate px-2 py-1 text-[12px] text-text-faint">{userEmail}</div>
        <button
          type="button"
          onClick={async () => {
            await logout();
            window.location.href = "/login";
          }}
          className="w-full rounded-lg px-2 py-1.5 text-left text-[13px] text-text-muted hover:bg-surface-sunken hover:text-text"
        >
          Log out
        </button>
      </div>
```

with `import { logout } from "@/lib/auth";`. Pass `userEmail={user.email}`
from `page.tsx`.

- [ ] **Step 4: Verify**

```bash
cd frontend && npx tsc --noEmit && npm run build
```

Then in the browser: visiting `/` while signed out redirects to `/login`;
after signing in the chat UI loads; "Log out" returns you to `/login`, and
going back to `/` redirects again.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/AuthGuard.tsx frontend/src/app/page.tsx frontend/src/components/Sidebar.tsx
git commit -m "feat: gate the app behind auth and add an account menu"
```

---

### Task 13: Saveable credentials in Settings

**Files:**
- Create: `frontend/src/lib/credentials.ts`
- Modify: `frontend/src/components/SettingsPanel.tsx`, `frontend/src/app/page.tsx`, `frontend/src/hooks/useVoiceSession.ts`

- [ ] **Step 1: Write the credentials client**

Create `frontend/src/lib/credentials.ts`:

```ts
import { API_URL } from "./config";
import type { ApiKeyFields, KeyProvider } from "./types";

const PROVIDER_PAYLOAD: Record<string, (k: ApiKeyFields) => Record<string, string>> = {
  local: (k) => ({ local_api_key: k.localApiKey, local_base_url: k.localBaseUrl }),
  openai: (k) => ({ openai_api_key: k.openaiApiKey }),
  azure: (k) => ({
    azure_api_key: k.azureApiKey,
    azure_endpoint: k.azureEndpoint,
    azure_deployment: k.azureDeployment,
  }),
  anthropic: (k) => ({ anthropic_api_key: k.anthropicApiKey }),
  gemini: (k) => ({ gemini_api_key: k.geminiApiKey }),
  bedrock: (k) => ({
    bedrock_access_key_id: k.bedrockAccessKeyId,
    bedrock_secret_access_key: k.bedrockSecretAccessKey,
    bedrock_region: k.bedrockRegion,
  }),
  openrouter: (k) => ({ openrouter_api_key: k.openrouterApiKey }),
  tavily: (k) => ({ tavily_api_key: k.tavilyApiKey }),
};

export async function fetchConfiguredProviders(): Promise<string[]> {
  const res = await fetch(`${API_URL}/credentials`, { credentials: "include" });
  if (!res.ok) throw new Error("Couldn't load saved credentials.");
  return (await res.json()).configured;
}

export async function saveCredential(provider: Exclude<KeyProvider, "">, keys: ApiKeyFields): Promise<void> {
  const res = await fetch(`${API_URL}/credentials/${provider}`, {
    method: "PUT",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(PROVIDER_PAYLOAD[provider](keys)),
  });
  if (res.status === 422) throw new Error("Fill in at least one field before saving.");
  if (!res.ok) throw new Error("Couldn't save. Try again.");
}

export async function saveTavilyKey(keys: ApiKeyFields): Promise<void> {
  const res = await fetch(`${API_URL}/credentials/tavily`, {
    method: "PUT",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tavily_api_key: keys.tavilyApiKey }),
  });
  if (!res.ok) throw new Error("Couldn't save. Try again.");
}

export async function removeCredential(provider: string): Promise<void> {
  await fetch(`${API_URL}/credentials/${provider}`, { method: "DELETE", credentials: "include" });
}
```

- [ ] **Step 2: Add Save/Remove to the Settings panel**

In `SettingsPanel.tsx`, add props `configured: string[]` and
`onCredentialsChanged: () => void`. Inside each provider's block (the
`settings.provider === "..."` sections), append this control row — shown here
for `openai`, repeated with the matching provider name in each block:

```tsx
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={async () => {
                    await saveCredential("openai", settings.keys);
                    onCredentialsChanged();
                  }}
                  className="rounded-lg bg-accent px-3 py-1.5 text-[12.5px] font-medium text-white hover:bg-accent-hover"
                >
                  Save
                </button>
                {configured.includes("openai") && (
                  <button
                    type="button"
                    onClick={async () => {
                      await removeCredential("openai");
                      onCredentialsChanged();
                    }}
                    className="rounded-lg px-3 py-1.5 text-[12.5px] text-danger hover:bg-danger-tint"
                  >
                    Remove
                  </button>
                )}
              </div>
```

Change `isProviderConfigured(p.value, settings.keys)` in the provider radio
list to `configured.includes(p.value)` so the dot reflects what is *saved*,
not what is typed. Replace the panel's explanatory paragraph with:

```tsx
          Saved securely on the server, encrypted at rest, and never sent back
          to the browser. Leave blank to use the server&apos;s own environment.
```

- [ ] **Step 3: Remove the old per-connection key flow**

In `frontend/src/app/page.tsx`:
- Delete the `fetchKeyTokenIfNeeded` and `keyProviderValidationError` imports and their use inside `doConnect`; `keyToken` is gone.
- Add `const [configured, setConfigured] = useState<string[]>([])` plus a
  `refreshConfigured` callback calling `fetchConfiguredProviders()`, invoked on
  mount and passed to `SettingsPanel` as `onCredentialsChanged`.
- Pass `configured={configured}` to `SettingsPanel`.

In `frontend/src/hooks/useVoiceSession.ts`, remove `keyToken` from the connect
options type and from the `/ws` query string; add `provider` instead so the
backend knows which stored credential to load:

```ts
  if (opts.provider) params.set("provider", opts.provider);
```

and pass `provider: settings.provider` from `doConnect`.

- [ ] **Step 4: Verify end to end**

```bash
cd frontend && npx tsc --noEmit && npm run build
```

Then with the full stack running: sign in, open Settings, pick a provider,
enter a key, click **Save**, reload the page — the provider's dot should still
show configured while the input is empty (the value is never sent back).
Connect and confirm a conversation works. Click **Remove** and confirm the dot
clears.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/credentials.ts frontend/src/components/SettingsPanel.tsx frontend/src/app/page.tsx frontend/src/hooks/useVoiceSession.ts
git commit -m "feat: save provider credentials per user from Settings"
```

---

### Task 14: Final verification

- [ ] **Step 1: Backend suite and lint**

```bash
cd backend && uv run pytest -v && uv run ruff check .
```

Expected: all tests pass, `All checks passed!`.

- [ ] **Step 2: Frontend checks**

```bash
cd ../frontend && npx tsc --noEmit && npx eslint . && npm run build
```

Expected: clean.

- [ ] **Step 3: Full stack smoke test**

```bash
cd .. && docker compose down && docker compose up -d --build
```

Walk the whole path in a browser: sign up → land on the chat UI → save a
provider credential → hold a short conversation → confirm it appears in the
sidebar → log out → log in as a second account → confirm the sidebar is empty
and the first account's session is not visible.

- [ ] **Step 4: Confirm the removed surface is gone**

```bash
cd backend && grep -rn "session-keys\|key_token" src/ tests/
```

Expected: no matches. If any remain, they are leftovers from Task 9.

- [ ] **Step 5: Final commit**

```bash
cd .. && git add -A && git commit -m "chore: verify multi-user platform phases 1-3"
```

---

## What this plan deliberately does not do

- No `GET`/`PUT /settings` endpoints. `user_settings` exists as a table but has
  no reader until Phase 4/5, and shipping an API that configures nothing is
  dead code.
- No compaction, PII detection, or call limits — Phases 4 and 5, each with
  their own plan.
- No Alembic. Schema still lands via `init_schema()` at startup. That is
  correct only while no environment holds data worth preserving; the spec's
  "Migration path" section records when that stops being true.
- No frontend test framework, matching the project's existing stance.

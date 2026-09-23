"""The app's own tables, living in the same SQLite file as LangGraph's
checkpointer tables.

LangGraph's checkpointer stores raw conversation state, keyed by
`thread_id` -- it has no concept of a session's title or which folder it
belongs to. These two tables hold exactly that: the metadata the sidebar
needs that the checkpointer doesn't track.
"""

from __future__ import annotations

import os
import json
import sqlite3
import sys
from pathlib import Path


def _data_dir() -> Path:
    """Where the database lives.

    Three cases, in order:

    1. `POS_DATA_DIR` if set -- the escape hatch, and what the container
       image uses to point at a mounted volume.
    2. A source checkout: `repo_root/data/`, so a developer's chats sit
       beside the code and survive a reinstall of the package.
    3. Installed: the OS user-data folder. Not beside the package -- an
       installed venv is disposable, and `uv tool upgrade` replacing it
       would take every stored conversation with it.

    Returns:
        The directory to put `checkpoint.db` in. Not created here;
        `connect()` does that.
    """
    override = os.environ.get("POS_DATA_DIR")
    if override:
        return Path(override)

    # .../src/pos/db.py -> repo_root. Only a checkout has this shape.
    repo_root = Path(__file__).resolve().parent.parent.parent
    if (repo_root / "pyproject.toml").is_file():
        return repo_root / "data"

    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(base) / "pos"


DB_DIR = _data_dir()
DB_PATH = DB_DIR / "checkpoint.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    folder TEXT,
    title TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    model TEXT,
    finish_reason TEXT,
    reasoning_effort TEXT,
    usage_json TEXT,
    tool_calls_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_turns_thread_id ON turns (thread_id);

-- What a deleted chat cost, without the chat.
--
-- Deleting a conversation should remove the conversation, not the record of
-- what was spent on it: the provider still billed for it, so a usage page
-- that forgets is simply wrong. One row per day and model, holding counts
-- only -- no messages, no tool results, nothing readable.
CREATE TABLE IF NOT EXISTS usage_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    day TEXT NOT NULL,
    model TEXT,
    turns INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read INTEGER NOT NULL DEFAULT 0,
    cache_creation INTEGER NOT NULL DEFAULT 0,
    cost REAL NOT NULL DEFAULT 0
);
"""


def connect() -> sqlite3.Connection:
    """Opens a connection to the app database, creating its tables if new.

    Returns:
        A connection with `row_factory` set so query results behave like
        dicts (`row["field"]`) instead of positional tuples.
    """
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _ledger_session(conn: sqlite3.Connection, session_id: str) -> None:
    """Rolls a session's usage into the ledger before its turns go.

    Grouped by day and model, so a long chat becomes a handful of rows
    rather than one per message, and nothing readable survives.

    Args:
        conn: An open connection, inside the caller's transaction.
        session_id: The session about to be deleted.
    """
    rows = conn.execute(
        """SELECT model, usage_json, created_at FROM turns
           WHERE thread_id = ? AND role = 'assistant' AND usage_json IS NOT NULL""",
        (session_id,),
    ).fetchall()

    buckets: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        try:
            usage = json.loads(row["usage_json"])
        except (TypeError, ValueError):
            continue
        key = ((row["created_at"] or "")[:10], row["model"] or "unknown")
        bucket = buckets.setdefault(
            key,
            {"turns": 0, "input": 0, "output": 0, "total": 0, "read": 0, "written": 0, "cost": 0.0},
        )
        bucket["turns"] += 1
        bucket["input"] += usage.get("input_tokens") or 0
        bucket["output"] += usage.get("output_tokens") or 0
        bucket["total"] += usage.get("total_tokens") or 0
        bucket["read"] += usage.get("cache_read") or 0
        bucket["written"] += usage.get("cache_creation") or 0
        bucket["cost"] += usage.get("cost_usd") or 0.0

    conn.executemany(
        """INSERT INTO usage_ledger
           (day, model, turns, input_tokens, output_tokens, total_tokens, cache_read, cache_creation, cost)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (day, model, b["turns"], b["input"], b["output"], b["total"], b["read"], b["written"], b["cost"])
            for (day, model), b in buckets.items()
        ],
    )


def delete_session(session_id: str) -> bool:
    """Removes a session and its stored turns, keeping what it cost.

    The conversation goes; the spend stays, as ledger rows (see
    `_ledger_session`). The LangGraph checkpoint for the same thread is
    deleted separately, by the caller, since it belongs to the checkpointer
    rather than to us.

    Args:
        session_id: The session (and thread) id to remove.

    Returns:
        True if a session row was actually deleted.
    """
    with connect() as conn:
        _ledger_session(conn, session_id)
        conn.execute("DELETE FROM turns WHERE thread_id = ?", (session_id,))
        cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return cursor.rowcount > 0


def clear_ledger() -> int:
    """Empties the deleted-chat ledger.

    Totals only ever grow otherwise, with no way to start again. Surviving
    chats are untouched -- this clears only the record of deleted ones.

    Returns:
        How many ledger rows were removed.
    """
    with connect() as conn:
        cursor = conn.execute("DELETE FROM usage_ledger")
        return cursor.rowcount

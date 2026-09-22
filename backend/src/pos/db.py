"""The app's own tables, living in the same SQLite file as LangGraph's
checkpointer tables.

LangGraph's checkpointer stores raw conversation state, keyed by
`thread_id` -- it has no concept of a session's title or which folder it
belongs to. These two tables hold exactly that: the metadata the sidebar
needs that the checkpointer doesn't track.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

# repo_root/data/, not backend/data/ -- resolved from this file's own
# location so it's the same regardless of the working directory the
# server is started from.
DB_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
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
"""


def connect() -> sqlite3.Connection:
    """Opens a connection to the app database, creating its tables if new.

    Returns:
        A connection with `row_factory` set so query results behave like
        dicts (`row["field"]`) instead of positional tuples.
    """
    DB_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def delete_session(session_id: str) -> bool:
    """Removes a session and its stored turns.

    The LangGraph checkpoint for the same thread is deleted separately, by
    the caller, since it belongs to the checkpointer rather than to us.

    Args:
        session_id: The session (and thread) id to remove.

    Returns:
        True if a session row was actually deleted.
    """
    with connect() as conn:
        conn.execute("DELETE FROM turns WHERE thread_id = ?", (session_id,))
        cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return cursor.rowcount > 0

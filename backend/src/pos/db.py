"""The app's own tables, living in the same SQLite file as LangGraph's
checkpointer tables.

LangGraph's checkpointer stores raw conversation state, keyed by
`thread_id` -- it has no concept of a session's title or which folder it
belongs to. These two tables hold exactly that: the metadata the sidebar
needs that the checkpointer doesn't track.
"""

from __future__ import annotations

import os
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

    # .../backend/src/pos/db.py -> repo_root. Only a checkout has this shape.
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    if (repo_root / "backend" / "pyproject.toml").is_file():
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

"""SQLite-backed session/turn history. Owned entirely by the transport
layers (server.py, main.py) -- Agent itself never imports this; it only
ever calls on_event(), and that's where storage hooks in (see
docs/superpowers/specs/2026-09-10-session-storage-text-mode-trace-ui-design.md).
A write failure here must never crash a live session -- callers wrap
add_turn()/create_session() in try/except, this module doesn't swallow
errors itself so a genuine bug surfaces during development."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import UTC, datetime


class SessionStore:
    def __init__(self, path: str = "sessions.db"):
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
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
            # Migration for DBs created before `title` existed -- CREATE
            # TABLE IF NOT EXISTS above is a no-op on an already-existing
            # file, so the column has to be added separately. Sqlite has
            # no "ADD COLUMN IF NOT EXISTS"; ignore the one error it
            # raises when the column is already there.
            try:
                self._conn.execute("ALTER TABLE sessions ADD COLUMN title TEXT")
            except sqlite3.OperationalError:
                pass
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
                (session_id, datetime.now(UTC).isoformat(), mode, tts_engine, llm_model),
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
                    datetime.now(UTC).isoformat(),
                ),
            )
            self._conn.commit()

    def set_title(self, session_id: str, title: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ?", (title, session_id)
            )
            self._conn.commit()

    def set_mode(self, session_id: str, mode: str) -> None:
        """A session's stored mode reflects whichever mode it was most
        recently used in, not just how it was first created -- see
        server.py's ws_endpoint, which calls this when a resumed
        connection's mode differs from what's on record."""
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET mode = ? WHERE id = ?", (mode, session_id)
            )
            self._conn.commit()

    def delete_session(self, session_id: str) -> bool:
        """Returns True if a session was deleted, False if id was unknown."""
        with self._lock:
            self._conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
            cursor = self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            self._conn.commit()
            return cursor.rowcount > 0

    def list_sessions(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT s.id, s.created_at, s.mode, s.tts_engine, s.llm_model, s.title,
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
                "SELECT id, created_at, mode, tts_engine, llm_model, title FROM sessions WHERE id = ?",
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

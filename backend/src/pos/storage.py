"""Postgres-backed session/turn history. Owned entirely by the transport
layers (server.py, main.py) -- Agent itself never imports this; it only
ever calls on_event(), and that's where storage hooks in (see
docs/superpowers/specs/2026-09-10-session-storage-text-mode-trace-ui-design.md).
A write failure here must never crash a live session -- callers wrap
add_turn()/create_session() in try/except, this module doesn't swallow
errors itself so a genuine bug surfaces during development."""

from __future__ import annotations

import threading
from datetime import UTC, datetime

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


class SessionStore:
    def __init__(self, dsn: str):
        self._conn = psycopg.connect(dsn, autocommit=False, row_factory=dict_row)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL,
                    mode TEXT NOT NULL,
                    tts_engine TEXT,
                    llm_model TEXT NOT NULL,
                    title TEXT
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    id BIGSERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id),
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    usage_json JSONB,
                    created_at TIMESTAMPTZ NOT NULL
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
                "VALUES (%s, %s, %s, %s, %s)",
                (session_id, datetime.now(UTC), mode, tts_engine, llm_model),
            )
            self._conn.commit()

    def add_turn(self, session_id: str, role: str, text: str, usage: dict | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO turns (session_id, role, text, usage_json, created_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (
                    session_id, role, text,
                    Jsonb(usage) if usage is not None else None,
                    datetime.now(UTC),
                ),
            )
            self._conn.commit()

    def set_title(self, session_id: str, title: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET title = %s WHERE id = %s", (title, session_id)
            )
            self._conn.commit()

    def set_mode(self, session_id: str, mode: str) -> None:
        """A session's stored mode reflects whichever mode it was most
        recently used in, not just how it was first created -- see
        server.py's ws_endpoint, which calls this when a resumed
        connection's mode differs from what's on record."""
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET mode = %s WHERE id = %s", (mode, session_id)
            )
            self._conn.commit()

    def delete_session(self, session_id: str) -> bool:
        """Returns True if a session was deleted, False if id was unknown."""
        with self._lock:
            self._conn.execute("DELETE FROM turns WHERE session_id = %s", (session_id,))
            cursor = self._conn.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
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
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
        return [{**row, "created_at": _iso(row["created_at"])} for row in rows]

    def get_session(self, session_id: str) -> dict | None:
        with self._lock:
            session_row = self._conn.execute(
                "SELECT id, created_at, mode, tts_engine, llm_model, title FROM sessions WHERE id = %s",
                (session_id,),
            ).fetchone()
            if session_row is None:
                return None
            turn_rows = self._conn.execute(
                "SELECT role, text, usage_json FROM turns WHERE session_id = %s ORDER BY id",
                (session_id,),
            ).fetchall()
        session_row = {**session_row, "created_at": _iso(session_row["created_at"])}
        return {
            "session": session_row,
            "turns": [
                {
                    "role": row["role"],
                    "text": row["text"],
                    "usage": row["usage_json"],
                }
                for row in turn_rows
            ],
        }

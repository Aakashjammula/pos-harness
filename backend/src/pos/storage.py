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

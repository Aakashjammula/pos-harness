import os

from pos.db import create_pool, init_schema

_DSN = os.environ.get("TEST_DATABASE_URL", "postgresql://pos:pos@localhost:5432/pos")

_EXPECTED_TABLES = {
    "users",
    "user_settings",
    "user_pii_rules",
    "refresh_tokens",
    "magic_link_tokens",
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


def test_users_can_be_created_without_a_password():
    pool = _pool()
    with pool.connection() as conn:
        conn.execute("DELETE FROM users WHERE email = 'passwordless@test.com'")
        user = conn.execute(
            "INSERT INTO users (email, password_hash) VALUES ('passwordless@test.com', NULL) "
            "RETURNING password_hash"
        ).fetchone()
        assert user["password_hash"] is None
    pool.close()

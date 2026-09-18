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

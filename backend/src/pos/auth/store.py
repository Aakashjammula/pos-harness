"""Users, refresh tokens, and encrypted provider credentials.

Kept separate from SessionStore: identity is read on every authenticated
request, session history only when the UI asks for it, and mixing them
would make one class own two unrelated lifecycles."""

from __future__ import annotations

from datetime import datetime, timedelta

import psycopg
from cryptography.exceptions import InvalidTag
from psycopg_pool import ConnectionPool

from pos.auth.crypto import decrypt_payload_ex, encrypt_payload


class EmailTaken(Exception):
    pass


class UsernameTaken(Exception):
    pass


class UserStore:
    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def create_user(
        self,
        email: str,
        password_hash: str | None = None,
        name: str | None = None,
        username: str | None = None,
    ) -> dict:
        try:
            with self._pool.connection() as conn:
                return conn.execute(
                    "INSERT INTO users (email, password_hash, name, username) "
                    "VALUES (%s, %s, %s, %s) "
                    "RETURNING id::text, email, name, username, created_at",
                    (email.lower(), password_hash, name, username),
                ).fetchone()
        except psycopg.errors.UniqueViolation as e:
            # Two unique constraints reach here and the caller owes the
            # user different words for each, so read which one fired
            # rather than reporting every collision as a taken email.
            if e.diag.constraint_name == "users_username_lower_key":
                raise UsernameTaken(username or "") from e
            raise EmailTaken(email) from e

    def get_or_create_user_by_email(self, email: str) -> dict:
        """For magic-link sign-in: a first-time email creates a
        passwordless account, an existing one just logs in -- there's no
        separate signup step. Falls back to a fetch on the (rare) race
        where two requests create the same email concurrently."""
        existing = self.get_user_by_email(email)
        if existing is not None:
            return existing
        try:
            return self.create_user(email, password_hash=None)
        except EmailTaken:
            return self.get_user_by_email(email)

    def get_user_by_email(self, email: str) -> dict | None:
        with self._pool.connection() as conn:
            return conn.execute(
                "SELECT id::text, email, password_hash, email_verified_at FROM users WHERE email = %s",
                (email.lower(),),
            ).fetchone()

    def get_user_by_id(self, user_id: str) -> dict | None:
        with self._pool.connection() as conn:
            return conn.execute(
                "SELECT id::text, email, name, username, created_at, "
                "(email_verified_at IS NOT NULL) AS email_verified, (password_hash IS NOT NULL) AS has_password "
                "FROM users WHERE id = %s",
                (user_id,),
            ).fetchone()

    def update_profile(self, user_id: str, name: str | None = None, username: str | None = None) -> None:
        """Change the display name and/or username. UsernameTaken if another account has it."""
        try:
            with self._pool.connection() as conn:
                conn.execute(
                    "UPDATE users SET name = COALESCE(%s, name), username = COALESCE(%s, username), "
                    "updated_at = now() WHERE id = %s",
                    (name, username, user_id),
                )
        except psycopg.errors.UniqueViolation as e:
            if e.diag.constraint_name == "users_username_lower_key":
                raise UsernameTaken(username or "") from e
            raise

    def delete_user(self, user_id: str) -> None:
        """Remove the account and everything that belongs to it (every dependent table
        cascades from users), plus any pending sign-in links for its address."""
        with self._pool.connection() as conn:
            row = conn.execute("SELECT email FROM users WHERE id = %s", (user_id,)).fetchone()
            conn.execute("DELETE FROM users WHERE id = %s", (user_id,))
            if row:
                conn.execute("DELETE FROM magic_link_tokens WHERE email = %s", (row["email"],))

    def get_password_hash(self, user_id: str) -> str | None:
        with self._pool.connection() as conn:
            row = conn.execute("SELECT password_hash FROM users WHERE id = %s", (user_id,)).fetchone()
        return row["password_hash"] if row else None

    def set_password_hash(self, user_id: str, password_hash: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE users SET password_hash = %s, updated_at = now() WHERE id = %s", (password_hash, user_id)
            )

    def mark_email_verified(self, user_id: str) -> bool:
        """True if this changed anything (it was not verified before)."""
        with self._pool.connection() as conn:
            cur = conn.execute(
                "UPDATE users SET email_verified_at = now(), updated_at = now() "
                "WHERE id = %s AND email_verified_at IS NULL",
                (user_id,),
            )
        return cur.rowcount > 0

    def reset_unverified_account(self, user_id: str) -> None:
        """Hand an account whose address was never proven to the person who has
        now proven it. Whoever created it (possibly someone who signed up with
        the victim's address in advance) keeps nothing: password, login
        sessions, saved credentials (which could point the victim's chats at a
        server the squatter controls), tool choices and chats are all removed."""
        with self._pool.connection() as conn:
            conn.execute("UPDATE users SET password_hash = NULL, updated_at = now() WHERE id = %s", (user_id,))
            conn.execute("DELETE FROM refresh_tokens WHERE user_id = %s", (user_id,))
            conn.execute("DELETE FROM auth_sessions WHERE user_id = %s", (user_id,))
            conn.execute("DELETE FROM api_credentials WHERE user_id = %s", (user_id,))
            conn.execute("DELETE FROM user_tool_settings WHERE user_id = %s", (user_id,))
            conn.execute("DELETE FROM sessions WHERE user_id = %s", (user_id,))

    def store_refresh_token(
        self, user_id: str, token_hash: str, expires_at: datetime, auth_session_id: str | None = None
    ) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO refresh_tokens (user_id, token_hash, expires_at, auth_session_id) "
                "VALUES (%s, %s, %s, %s)",
                (user_id, token_hash, expires_at, auth_session_id),
            )

    def consume_refresh_token_with_session(self, token_hash: str) -> tuple[str, str | None] | None:
        """Single-use: revokes the token as it's read, so replaying one is
        always rejected. Returns (user id, login session id or None for a token
        issued before sessions existed)."""
        with self._pool.connection() as conn:
            row = conn.execute(
                "UPDATE refresh_tokens SET revoked_at = now() "
                "WHERE token_hash = %s AND revoked_at IS NULL AND expires_at > now() "
                "RETURNING user_id::text, auth_session_id::text",
                (token_hash,),
            ).fetchone()
        return (row["user_id"], row["auth_session_id"]) if row else None

    def consume_refresh_token(self, token_hash: str) -> str | None:
        found = self.consume_refresh_token_with_session(token_hash)
        return found[0] if found else None

    # --- login sessions (one per signed-in browser/device) ---------------------------------

    def create_auth_session(self, user_id: str, user_agent: str | None, ip: str | None) -> str:
        with self._pool.connection() as conn:
            row = conn.execute(
                "INSERT INTO auth_sessions (user_id, user_agent, ip) VALUES (%s, %s, %s) RETURNING id::text",
                (user_id, (user_agent or "")[:300] or None, ip),
            ).fetchone()
        return row["id"]

    def auth_session_active(self, session_id: str, user_id: str) -> bool:
        try:
            with self._pool.connection() as conn:
                row = conn.execute(
                    "SELECT 1 FROM auth_sessions WHERE id = %s AND user_id = %s AND revoked_at IS NULL",
                    (session_id, user_id),
                ).fetchone()
        except Exception:   # a malformed id is simply not an active session
            return False
        return row is not None

    def touch_auth_session(self, session_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("UPDATE auth_sessions SET last_seen_at = now() WHERE id = %s", (session_id,))

    def list_auth_sessions(self, user_id: str) -> list[dict]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT id::text, user_agent, ip, created_at, last_seen_at FROM auth_sessions "
                "WHERE user_id = %s AND revoked_at IS NULL ORDER BY last_seen_at DESC",
                (user_id,),
            ).fetchall()
        return list(rows)

    def revoke_auth_session(self, user_id: str, session_id: str) -> bool:
        """End one device's session. Its refresh tokens are DELETED, not marked
        revoked: a revoked-but-present token is what replay detection reads as
        theft, and that would sign the user's other devices out too."""
        try:
            with self._pool.connection() as conn:
                cur = conn.execute(
                    "UPDATE auth_sessions SET revoked_at = now() "
                    "WHERE id = %s AND user_id = %s AND revoked_at IS NULL",
                    (session_id, user_id),
                )
                if cur.rowcount == 0:
                    return False
                conn.execute("DELETE FROM refresh_tokens WHERE auth_session_id = %s", (session_id,))
        except Exception:   # e.g. not a UUID
            return False
        return True

    def revoke_all_auth_sessions(self, user_id: str, except_id: str | None = None) -> int:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "UPDATE auth_sessions SET revoked_at = now() "
                "WHERE user_id = %s AND revoked_at IS NULL AND (%s::uuid IS NULL OR id <> %s::uuid) "
                "RETURNING id::text",
                (user_id, except_id, except_id),
            ).fetchall()
            ids = [r["id"] for r in rows]
            if ids:
                conn.execute("DELETE FROM refresh_tokens WHERE auth_session_id = ANY(%s::uuid[])", (ids,))
        return len(ids)

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
        """Theft response: end every refresh token AND every login session, so
        access tokens that were copied die with them."""
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE refresh_tokens SET revoked_at = now() "
                "WHERE user_id = %s AND revoked_at IS NULL",
                (user_id,),
            )
            conn.execute(
                "UPDATE auth_sessions SET revoked_at = now() WHERE user_id = %s AND revoked_at IS NULL",
                (user_id,),
            )

    def store_magic_link_token(self, email: str, token_hash: str, expires_at: datetime) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO magic_link_tokens (email, token_hash, expires_at) VALUES (%s, %s, %s)",
                (email.lower(), token_hash, expires_at),
            )

    def consume_magic_link_token(self, token_hash: str) -> str | None:
        """Single-use, like consume_refresh_token: revokes the token as
        it's read, so replaying a link is always rejected."""
        with self._pool.connection() as conn:
            row = conn.execute(
                "UPDATE magic_link_tokens SET consumed_at = now() "
                "WHERE token_hash = %s AND consumed_at IS NULL AND expires_at > now() "
                "RETURNING email",
                (token_hash,),
            ).fetchone()
        return row["email"] if row else None

    def recent_magic_link_request(self, email: str, within: timedelta) -> bool:
        """True if a token for this email was requested within `within` --
        throttles repeat requests so one email address can't be spammed
        with sign-in links."""
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM magic_link_tokens "
                "WHERE email = %s AND created_at > now() - %s "
                "LIMIT 1",
                (email.lower(), within),
            ).fetchone()
        return row is not None

    @staticmethod
    def _aad(user_id: str, provider: str) -> bytes:
        return f"{user_id}:{provider}".encode()

    def save_credential(self, user_id: str, provider: str, payload: dict) -> None:
        ciphertext, nonce = encrypt_payload(payload, self._aad(user_id, provider))
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
        try:
            payload, needs_upgrade = decrypt_payload_ex(
                bytes(row["encrypted_payload"]), bytes(row["nonce"]), self._aad(user_id, provider)
            )
        except InvalidTag:
            # Copied from another slot, corrupted, or the key is gone. Treat it as not
            # saved (the person is asked to enter it again) rather than failing every
            # request for that user. Nothing secret goes in the log line.
            print(f"  credential for user {user_id} / {provider} could not be decrypted -- treating it as unset")
            return None
        if needs_upgrade:   # written before owner binding, or under a retired key: re-encrypt now
            self.save_credential(user_id, provider, payload)
        return payload

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

    def get_tool_settings(self, user_id: str) -> dict[str, bool]:
        """The user's own on/off choices, {tool id: enabled}. A tool with no row
        has never been touched -- callers fall back to the tool's default."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT tool_id, enabled FROM user_tool_settings WHERE user_id = %s", (user_id,)
            ).fetchall()
        return {r["tool_id"]: r["enabled"] for r in rows}

    def set_tool_enabled(self, user_id: str, tool_id: str, enabled: bool) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO user_tool_settings (user_id, tool_id, enabled) VALUES (%s, %s, %s) "
                "ON CONFLICT (user_id, tool_id) DO UPDATE "
                "SET enabled = EXCLUDED.enabled, updated_at = now()",
                (user_id, tool_id, enabled),
            )

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

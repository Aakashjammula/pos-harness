import base64
import os

import pytest
from cryptography.exceptions import InvalidTag

from pos import config
from pos.auth.crypto import decrypt_payload_ex, encrypt_payload
from pos.auth.store import UserStore
from pos.db import create_pool, init_schema

_pool = None
PAYLOAD = {"OPENAI_API_KEY": "sk-secret"}


def _key():
    return base64.b64encode(os.urandom(32)).decode()


@pytest.fixture(autouse=True)
def _keys(monkeypatch):
    monkeypatch.setattr(config, "ENCRYPTION_KEY", _key())
    monkeypatch.setattr(config, "ENCRYPTION_KEY_PREVIOUS", "")


def _store():
    global _pool
    if _pool is None:
        _pool = create_pool(os.environ["TEST_DATABASE_URL"], max_size=5)
        init_schema(_pool)
    with _pool.connection() as conn:
        conn.execute("TRUNCATE users, magic_link_tokens RESTART IDENTITY CASCADE")
    return UserStore(_pool)


def _user(store, n):
    return store.create_user(f"u{n}@test.com", None, name="N", username=f"user_{n}")["id"]


def test_round_trip_with_associated_data():
    ct, nonce = encrypt_payload(PAYLOAD, b"user-1:openai")

    payload, needs_upgrade = decrypt_payload_ex(ct, nonce, b"user-1:openai")

    assert payload == PAYLOAD and needs_upgrade is False


def test_ciphertext_cannot_be_decrypted_under_another_owner_or_provider():
    ct, nonce = encrypt_payload(PAYLOAD, b"user-1:openai")

    for other in (b"user-2:openai", b"user-1:anthropic"):
        with pytest.raises(InvalidTag):
            decrypt_payload_ex(ct, nonce, other)


def test_each_encryption_uses_a_fresh_nonce():
    a = encrypt_payload(PAYLOAD, b"x")
    b = encrypt_payload(PAYLOAD, b"x")
    assert a[1] != b[1] and a[0] != b[0]


def test_a_row_written_before_associated_data_existed_is_still_readable_and_flagged():
    ct, nonce = encrypt_payload(PAYLOAD, None)             # the old format

    payload, needs_upgrade = decrypt_payload_ex(ct, nonce, b"user-1:openai")

    assert payload == PAYLOAD and needs_upgrade is True


def test_a_previous_key_still_decrypts_during_rotation(monkeypatch):
    old = config.ENCRYPTION_KEY
    ct, nonce = encrypt_payload(PAYLOAD, b"a:b")
    monkeypatch.setattr(config, "ENCRYPTION_KEY", _key())            # the new current key
    with pytest.raises(InvalidTag):
        decrypt_payload_ex(ct, nonce, b"a:b")                          # not yet told about the old one

    monkeypatch.setattr(config, "ENCRYPTION_KEY_PREVIOUS", old)

    payload, needs_upgrade = decrypt_payload_ex(ct, nonce, b"a:b")
    assert payload == PAYLOAD and needs_upgrade is True                # re-encrypt under the current key


def test_several_previous_keys_can_be_listed(monkeypatch):
    first, second = config.ENCRYPTION_KEY, _key()
    ct, nonce = encrypt_payload(PAYLOAD, b"a:b")
    monkeypatch.setattr(config, "ENCRYPTION_KEY", second)
    monkeypatch.setattr(config, "ENCRYPTION_KEY_PREVIOUS", f"{_key()}, {first}")

    assert decrypt_payload_ex(ct, nonce, b"a:b")[0] == PAYLOAD


def test_a_wrong_key_fails_loudly(monkeypatch):
    ct, nonce = encrypt_payload(PAYLOAD, b"a:b")
    monkeypatch.setattr(config, "ENCRYPTION_KEY", _key())

    with pytest.raises(InvalidTag):
        decrypt_payload_ex(ct, nonce, b"a:b")


# --- through the store ----------------------------------------------------------------------------------

def test_the_store_round_trips_and_binds_credentials_to_their_owner():
    store = _store()
    a = _user(store, 1)

    store.save_credential(a, "openai", PAYLOAD)

    assert store.get_credential(a, "openai") == PAYLOAD


def test_a_legacy_row_is_returned_and_rewritten_in_the_new_format():
    store = _store()
    uid = _user(store, 1)
    ct, nonce = encrypt_payload(PAYLOAD, None)                          # as an older version stored it
    with store._pool.connection() as conn:
        conn.execute(
            "INSERT INTO api_credentials (user_id, provider, encrypted_payload, nonce) VALUES (%s, %s, %s, %s)",
            (uid, "openai", ct, nonce),
        )

    assert store.get_credential(uid, "openai") == PAYLOAD                # still works

    with store._pool.connection() as conn:
        row = conn.execute("SELECT encrypted_payload, nonce FROM api_credentials WHERE user_id = %s", (uid,)).fetchone()
    _, upgraded = decrypt_payload_ex(bytes(row["encrypted_payload"]), bytes(row["nonce"]), f"{uid}:openai".encode())
    assert upgraded is False                                             # now bound to its owner


def test_copying_one_users_encrypted_row_to_another_does_not_hand_over_the_credential(capsys):
    store = _store()
    a, b = _user(store, 1), _user(store, 2)
    store.save_credential(a, "openai", PAYLOAD)
    with store._pool.connection() as conn:                               # an attacker with database write access
        row = conn.execute("SELECT encrypted_payload, nonce FROM api_credentials WHERE user_id = %s", (a,)).fetchone()
        conn.execute(
            "INSERT INTO api_credentials (user_id, provider, encrypted_payload, nonce) VALUES (%s, %s, %s, %s)",
            (b, "openai", row["encrypted_payload"], row["nonce"]),
        )

    assert store.get_credential(b, "openai") is None                     # treated as absent, not decrypted
    assert store.get_credential(a, "openai") == PAYLOAD
    assert "sk-secret" not in capsys.readouterr().out                    # and the failure logs no secret


def test_a_row_encrypted_under_a_previous_key_is_upgraded_on_read(monkeypatch):
    store = _store()
    uid = _user(store, 1)
    old = config.ENCRYPTION_KEY
    store.save_credential(uid, "openai", PAYLOAD)
    monkeypatch.setattr(config, "ENCRYPTION_KEY", _key())
    monkeypatch.setattr(config, "ENCRYPTION_KEY_PREVIOUS", old)

    assert store.get_credential(uid, "openai") == PAYLOAD

    monkeypatch.setattr(config, "ENCRYPTION_KEY_PREVIOUS", "")           # the old key can now be retired
    assert store.get_credential(uid, "openai") == PAYLOAD

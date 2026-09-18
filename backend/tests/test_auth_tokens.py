from datetime import UTC, datetime, timedelta

import jwt
import pytest

from pos import config
from pos.auth.tokens import (
    ACCESS_TOKEN_TTL,
    MAGIC_LINK_TOKEN_TTL,
    REFRESH_TOKEN_TTL,
    create_access_token,
    decode_access_token,
    hash_magic_link_token,
    hash_refresh_token,
    new_magic_link_token,
    new_refresh_token,
)


@pytest.fixture(autouse=True)
def _secret(monkeypatch):
    monkeypatch.setattr(config, "JWT_SECRET", "test-secret")


def test_access_token_round_trips_the_user_id():
    token = create_access_token("user-123")
    assert decode_access_token(token) == "user-123"


def test_access_token_expires():
    token = jwt.encode(
        {"sub": "user-123", "exp": datetime.now(UTC) - timedelta(seconds=1)},
        "test-secret",
        algorithm="HS256",
    )
    assert decode_access_token(token) is None


def test_access_token_signed_with_another_secret_is_rejected():
    token = jwt.encode(
        {"sub": "user-123", "exp": datetime.now(UTC) + timedelta(minutes=5)},
        "wrong-secret",
        algorithm="HS256",
    )
    assert decode_access_token(token) is None


def test_garbage_token_is_rejected():
    assert decode_access_token("not.a.token") is None


def test_refresh_token_returns_raw_and_hash_and_never_stores_raw():
    raw, digest = new_refresh_token()
    assert raw != digest
    assert hash_refresh_token(raw) == digest
    assert len(digest) == 64


def test_ttls_match_the_spec():
    assert ACCESS_TOKEN_TTL == timedelta(minutes=15)
    assert REFRESH_TOKEN_TTL == timedelta(days=30)
    assert MAGIC_LINK_TOKEN_TTL == timedelta(minutes=15)


def test_magic_link_token_returns_raw_and_hash_and_never_stores_raw():
    raw, digest = new_magic_link_token()
    assert raw != digest
    assert hash_magic_link_token(raw) == digest
    assert len(digest) == 64

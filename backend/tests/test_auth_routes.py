import base64
import os

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from pos import config
from pos.auth.routes import build_auth_router
from pos.auth.store import UserStore
from pos.db import create_pool, init_schema

_DSN = os.environ.get("TEST_DATABASE_URL", "postgresql://pos:pos@localhost:5432/pos")


@pytest.fixture(autouse=True)
def _secrets(monkeypatch):
    monkeypatch.setattr(config, "JWT_SECRET", "test-secret")
    monkeypatch.setattr(config, "ENCRYPTION_KEY", base64.b64encode(os.urandom(32)).decode())


def _client():
    pool = create_pool(_DSN, max_size=5)
    init_schema(pool)
    with pool.connection() as conn:
        conn.execute("TRUNCATE users RESTART IDENTITY CASCADE")
    app = FastAPI()
    app.include_router(build_auth_router(UserStore(pool)))
    return TestClient(app)


def test_signup_sets_cookies_and_returns_the_user():
    client = _client()
    resp = client.post("/auth/signup", json={"email": "a@test.com", "password": "pw-12345678"})

    assert resp.status_code == 200
    assert resp.json()["email"] == "a@test.com"
    assert "pos_access" in resp.cookies
    assert "pos_refresh" in resp.cookies


def test_signup_never_returns_the_password_or_its_hash():
    client = _client()
    body = client.post("/auth/signup", json={"email": "a@test.com", "password": "pw-12345678"}).text
    assert "pw-12345678" not in body
    assert "argon2" not in body


def test_duplicate_signup_is_rejected():
    client = _client()
    client.post("/auth/signup", json={"email": "a@test.com", "password": "pw-12345678"})
    resp = client.post("/auth/signup", json={"email": "a@test.com", "password": "pw-12345678"})
    assert resp.status_code == 409


def test_short_password_is_rejected():
    client = _client()
    resp = client.post("/auth/signup", json={"email": "a@test.com", "password": "short"})
    assert resp.status_code == 422


def test_login_with_the_right_password_succeeds():
    client = _client()
    client.post("/auth/signup", json={"email": "a@test.com", "password": "pw-12345678"})
    client.cookies.clear()

    resp = client.post("/auth/login", json={"email": "a@test.com", "password": "pw-12345678"})

    assert resp.status_code == 200
    assert "pos_access" in resp.cookies


def test_login_with_the_wrong_password_fails():
    client = _client()
    client.post("/auth/signup", json={"email": "a@test.com", "password": "pw-12345678"})
    resp = client.post("/auth/login", json={"email": "a@test.com", "password": "wrong-password"})
    assert resp.status_code == 401


def test_login_for_an_unknown_email_fails_the_same_way():
    client = _client()
    resp = client.post("/auth/login", json={"email": "nobody@test.com", "password": "pw-12345678"})
    assert resp.status_code == 401


def test_me_requires_authentication():
    client = _client()
    assert client.get("/auth/me").status_code == 401


def test_me_returns_the_signed_in_user():
    client = _client()
    client.post("/auth/signup", json={"email": "a@test.com", "password": "pw-12345678"})
    assert client.get("/auth/me").json()["email"] == "a@test.com"


def test_refresh_rotates_the_token_and_the_old_one_stops_working():
    client = _client()
    client.post("/auth/signup", json={"email": "a@test.com", "password": "pw-12345678"})
    first_refresh = client.cookies["pos_refresh"]

    assert client.post("/auth/refresh").status_code == 200

    client.cookies.set("pos_refresh", first_refresh)
    assert client.post("/auth/refresh").status_code == 401


def test_replaying_an_old_refresh_token_kills_the_whole_family():
    client = _client()
    client.post("/auth/signup", json={"email": "a@test.com", "password": "pw-12345678"})
    stolen = client.cookies["pos_refresh"]
    client.post("/auth/refresh")
    live = client.cookies["pos_refresh"]

    client.cookies.set("pos_refresh", stolen)
    assert client.post("/auth/refresh").status_code == 401

    client.cookies.set("pos_refresh", live)
    assert client.post("/auth/refresh").status_code == 401


def test_logout_clears_cookies_and_ends_the_session():
    client = _client()
    client.post("/auth/signup", json={"email": "a@test.com", "password": "pw-12345678"})

    assert client.post("/auth/logout").status_code == 200

    assert client.get("/auth/me").status_code == 401


def test_credentials_are_per_user_and_never_returned():
    client = _client()
    client.post("/auth/signup", json={"email": "a@test.com", "password": "pw-12345678"})
    client.put("/credentials/openai", json={"openai_api_key": "sk-secret"})

    listed = client.get("/credentials")
    assert listed.json() == {"configured": ["openai"]}
    assert "sk-secret" not in listed.text

    client.post("/auth/logout")
    client.post("/auth/signup", json={"email": "b@test.com", "password": "pw-12345678"})
    assert client.get("/credentials").json() == {"configured": []}


def test_delete_credential():
    client = _client()
    client.post("/auth/signup", json={"email": "a@test.com", "password": "pw-12345678"})
    client.put("/credentials/openai", json={"openai_api_key": "sk-secret"})

    assert client.delete("/credentials/openai").status_code == 200
    assert client.get("/credentials").json() == {"configured": []}
    assert client.delete("/credentials/openai").status_code == 404


def test_credentials_require_authentication():
    client = _client()
    assert client.get("/credentials").status_code == 401
    assert client.put("/credentials/openai", json={"openai_api_key": "x"}).status_code == 401

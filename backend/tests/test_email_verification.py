import base64
import os

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from pos import config
from pos.auth.routes import build_auth_router
from pos.auth.store import UserStore
from pos.db import create_pool, init_schema
from pos.storage import SessionStore

_pool = None
sent: list[tuple[str, str]] = []
PW = "pw-12345678"
ATTACKER_PW = "attackers-own-password"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(config, "JWT_SECRET", "test-secret-" + "x" * 32)
    monkeypatch.setattr(config, "ENCRYPTION_KEY", base64.b64encode(os.urandom(32)).decode())
    sent.clear()

    async def capture(email, link_url):
        sent.append((email, link_url))

    monkeypatch.setattr("pos.auth.routes.send_magic_link_email", capture)


def _app():
    global _pool
    if _pool is None:
        _pool = create_pool(os.environ["TEST_DATABASE_URL"], max_size=5)
        init_schema(_pool)
    with _pool.connection() as conn:
        conn.execute("TRUNCATE users, magic_link_tokens RESTART IDENTITY CASCADE")
    users = UserStore(_pool)
    app = FastAPI()
    app.include_router(build_auth_router(users))
    return app, users, SessionStore(_pool)


def _token(email):
    _, url = next(pair for pair in reversed(sent) if pair[0] == email)
    return url.rsplit("token=", 1)[1]


def _signup(client, email, password=PW, username="user_a"):
    resp = client.post("/auth/signup", json={"name": "N", "username": username, "email": email, "password": password})
    assert resp.status_code == 200
    return resp.json()["id"]


def _me(client):
    resp = client.get("/auth/me")
    return resp.json() if resp.status_code == 200 else None


def test_the_pre_hijack_attack_is_neutralised_when_the_real_owner_signs_in_by_link():
    app, users, chats = _app()
    attacker = TestClient(app)
    user_id = _signup(attacker, "victim@test.com", ATTACKER_PW, "squatter")
    users.save_credential(user_id, "openai", {"OPENAI_API_KEY": "sk-planted"})
    users.save_credential(user_id, "local", {"LOCAL_BASE_URL": "http://attacker.example/v1"})
    chats.create_session("planted", user_id, mode="text", tts_engine=None, llm_model="m")

    victim = TestClient(app)                                   # a different browser: no session for that account
    victim.post("/auth/magic-link/request", json={"email": "victim@test.com"})
    assert victim.post("/auth/magic-link/verify", json={"token": _token("victim@test.com")}).status_code == 200

    # the victim is in, on a clean account...
    me = _me(victim)
    assert me["email"] == "victim@test.com" and me["email_verified"] is True
    assert users.list_credential_providers(user_id) == []              # planted credentials are gone
    assert chats.list_sessions(user_id) == []                          # so are planted chats
    # ...and the attacker is locked out of it
    stranger = TestClient(app)
    assert stranger.post("/auth/login", json={"email": "victim@test.com", "password": ATTACKER_PW}).status_code == 401
    assert attacker.get("/auth/me").status_code == 401                 # their existing session is dead
    assert attacker.post("/auth/refresh").status_code == 401


def test_verifying_from_the_browser_that_is_already_signed_in_keeps_everything():
    app, users, chats = _app()
    owner = TestClient(app)
    user_id = _signup(owner, "owner@test.com")
    users.save_credential(user_id, "openai", {"OPENAI_API_KEY": "sk-mine"})
    chats.create_session("mine", user_id, mode="text", tts_engine=None, llm_model="m")
    assert _me(owner)["email_verified"] is False

    assert owner.post("/auth/verify-email/request").json() == {"sent": True}
    assert owner.post("/auth/magic-link/verify", json={"token": _token("owner@test.com")}).status_code == 200

    assert _me(owner)["email_verified"] is True
    assert users.list_credential_providers(user_id) == ["openai"]
    assert [s["id"] for s in chats.list_sessions(user_id)] == ["mine"]
    assert TestClient(app).post("/auth/login", json={"email": "owner@test.com", "password": PW}).status_code == 200


def test_a_link_opened_in_a_different_browser_is_treated_as_a_takeover_even_for_the_real_owner():
    """Safe by default: the app cannot tell an owner on their phone from a victim."""
    app, users, _ = _app()
    laptop = TestClient(app)
    user_id = _signup(laptop, "owner@test.com")
    laptop.post("/auth/verify-email/request")

    phone = TestClient(app)
    phone.post("/auth/magic-link/verify", json={"token": _token("owner@test.com")})

    assert _me(phone)["email_verified"] is True
    assert TestClient(app).post("/auth/login", json={"email": "owner@test.com", "password": PW}).status_code == 401
    assert users.list_credential_providers(user_id) == []


def test_another_signed_in_user_opening_the_link_is_not_the_account_holder():
    app, users, _ = _app()
    a = TestClient(app)
    a_id = _signup(a, "a@test.com", ATTACKER_PW, "user_a")
    a.post("/auth/verify-email/request")
    token = _token("a@test.com")
    b = TestClient(app)
    _signup(b, "b@test.com", PW, "user_b")

    b.post("/auth/magic-link/verify", json={"token": token})

    assert users.get_user_by_id(a_id)["email_verified"] is True
    assert TestClient(app).post("/auth/login", json={"email": "a@test.com", "password": ATTACKER_PW}).status_code == 401


def test_an_already_verified_account_signs_in_by_link_without_losing_anything():
    app, users, chats = _app()
    owner = TestClient(app)
    user_id = _signup(owner, "owner@test.com")
    owner.post("/auth/verify-email/request")
    owner.post("/auth/magic-link/verify", json={"token": _token("owner@test.com")})       # verified now
    with users._pool.connection() as conn:                # the 60s request cooldown would swallow the next mail
        conn.execute("DELETE FROM magic_link_tokens")
    users.save_credential(user_id, "openai", {"OPENAI_API_KEY": "sk-mine"})
    chats.create_session("mine", user_id, mode="text", tts_engine=None, llm_model="m")

    other_browser = TestClient(app)
    other_browser.post("/auth/magic-link/request", json={"email": "owner@test.com"})
    other_browser.post("/auth/magic-link/verify", json={"token": _token("owner@test.com")})

    assert _me(other_browser)["email"] == "owner@test.com"
    assert users.list_credential_providers(user_id) == ["openai"]
    assert [s["id"] for s in chats.list_sessions(user_id)] == ["mine"]
    assert TestClient(app).post("/auth/login", json={"email": "owner@test.com", "password": PW}).status_code == 200


def test_a_brand_new_email_signing_in_by_link_gets_a_verified_account():
    app, _, _ = _app()
    client = TestClient(app)
    client.post("/auth/magic-link/request", json={"email": "new@test.com"})

    client.post("/auth/magic-link/verify", json={"token": _token("new@test.com")})

    assert _me(client)["email_verified"] is True


def test_a_passwordless_signup_is_verified_by_its_link():
    app, _, _ = _app()
    client = TestClient(app)
    client.post("/auth/signup", json={"name": "N", "username": "user_n", "email": "n@test.com"})

    client.post("/auth/magic-link/verify", json={"token": _token("n@test.com")})

    assert _me(client)["email_verified"] is True


def test_verify_email_request_needs_a_session_and_is_a_no_op_once_verified():
    app, _, _ = _app()
    assert TestClient(app).post("/auth/verify-email/request").status_code == 401

    owner = TestClient(app)
    _signup(owner, "owner@test.com")
    owner.post("/auth/verify-email/request")
    owner.post("/auth/magic-link/verify", json={"token": _token("owner@test.com")})
    sent.clear()

    assert owner.post("/auth/verify-email/request").json() == {"sent": False, "already_verified": True}
    assert sent == []


def test_a_planted_credential_never_survives_a_takeover_even_if_the_attacker_kept_a_session():
    app, users, _ = _app()
    attacker = TestClient(app)
    user_id = _signup(attacker, "victim@test.com", ATTACKER_PW, "squatter")
    users.save_credential(user_id, "local", {"LOCAL_BASE_URL": "http://attacker.example/v1"})
    second_device = TestClient(app)
    second_device.post("/auth/login", json={"email": "victim@test.com", "password": ATTACKER_PW})

    victim = TestClient(app)
    victim.post("/auth/magic-link/request", json={"email": "victim@test.com"})
    victim.post("/auth/magic-link/verify", json={"token": _token("victim@test.com")})

    assert second_device.get("/auth/me").status_code == 401
    assert attacker.get("/auth/me").status_code == 401
    assert users.get_credential(user_id, "local") is None

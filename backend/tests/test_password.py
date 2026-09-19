import base64
import os

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from pos import config
from pos.auth.routes import build_auth_router
from pos.auth.store import UserStore
from pos.db import create_pool, init_schema

_pool = None
PW = "pw-12345678"
NEW = "brand-new-password"
sent: list[tuple[str, str]] = []


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
    app = FastAPI()
    app.include_router(build_auth_router(UserStore(_pool)))
    return app


def _signup(client, email="a@test.com", password=PW):
    body = {"name": "A", "username": "user_a", "email": email}
    if password:
        body["password"] = password
    assert client.post("/auth/signup", json=body).status_code == 200


def _login(client, password, email="a@test.com"):
    return client.post("/auth/login", json={"email": email, "password": password}).status_code


def test_a_passwordless_account_can_set_a_password_without_a_current_one():
    app = _app()
    client = TestClient(app)
    client.post("/auth/magic-link/request", json={"email": "a@test.com"})
    token = sent[-1][1].rsplit("token=", 1)[1]
    client.post("/auth/magic-link/verify", json={"token": token})

    resp = client.put("/auth/password", json={"new_password": NEW})

    assert resp.status_code == 200
    assert _login(TestClient(app), NEW) == 200


def test_changing_a_password_needs_the_current_one():
    app = _app()
    client = TestClient(app)
    _signup(client)

    assert client.put("/auth/password", json={"new_password": NEW}).status_code == 401
    wrong = {"current_password": "wrong-one-x", "new_password": NEW}
    assert client.put("/auth/password", json=wrong).status_code == 401
    assert _login(TestClient(app), PW) == 200                    # nothing changed


def test_a_correct_current_password_changes_it():
    app = _app()
    client = TestClient(app)
    _signup(client)

    resp = client.put("/auth/password", json={"current_password": PW, "new_password": NEW})

    assert resp.status_code == 200
    assert _login(TestClient(app), NEW) == 200
    assert _login(TestClient(app), PW) == 401


def test_a_weak_new_password_is_refused():
    app = _app()
    client = TestClient(app)
    _signup(client)

    assert client.put("/auth/password", json={"current_password": PW, "new_password": "short"}).status_code == 422
    assert _login(TestClient(app), PW) == 200


def test_changing_the_password_signs_out_the_other_devices_but_not_this_one():
    app = _app()
    laptop, phone = TestClient(app, headers={"user-agent": "Laptop"}), TestClient(app, headers={"user-agent": "Phone"})
    _signup(laptop)
    _login(phone, PW)

    laptop.put("/auth/password", json={"current_password": PW, "new_password": NEW})

    assert phone.get("/auth/me").status_code == 401
    assert phone.post("/auth/refresh").status_code == 401
    assert laptop.get("/auth/me").status_code == 200
    assert [s["user_agent"] for s in laptop.get("/auth/sessions").json()["sessions"]] == ["Laptop"]


def test_the_password_route_requires_a_session():
    app = _app()
    assert TestClient(app).put("/auth/password", json={"new_password": NEW}).status_code == 401


def test_the_response_and_errors_never_contain_a_password():
    app = _app()
    client = TestClient(app)
    _signup(client)

    ok = client.put("/auth/password", json={"current_password": PW, "new_password": NEW})
    bad = client.put("/auth/password", json={"current_password": "wrong-one-x", "new_password": NEW})

    for resp in (ok, bad):
        assert PW not in resp.text and NEW not in resp.text and "wrong-one-x" not in resp.text


@pytest.mark.parametrize("case", ["unknown email", "passwordless account", "wrong password", "right password"])
def test_login_always_does_exactly_one_password_verification(monkeypatch, case):
    """Otherwise an unknown email answers ~40ms faster and reveals who has an account."""
    import pos.auth.routes as routes

    app = _app()
    other = TestClient(app)
    _signup(other, "a@test.com")
    with _pool.connection() as conn:                    # a passwordless account
        conn.execute("INSERT INTO users (email, password_hash) VALUES ('p@test.com', NULL)")
    calls = []
    real = routes.verify_password
    monkeypatch.setattr(routes, "verify_password", lambda pw, h: calls.append(h) or real(pw, h))
    email, password = {
        "unknown email": ("nobody@test.com", PW),
        "passwordless account": ("p@test.com", PW),
        "wrong password": ("a@test.com", "wrong-one-x"),
        "right password": ("a@test.com", PW),
    }[case]

    status = TestClient(app).post("/auth/login", json={"email": email, "password": password}).status_code

    assert len(calls) == 1
    assert status == (200 if case == "right password" else 401)


def test_the_dummy_hash_can_never_authenticate_anyone():
    from pos.auth.passwords import DUMMY_HASH, verify_password

    assert DUMMY_HASH.startswith("$argon2id$")
    for guess in ("", "password", "dummy", PW):
        assert verify_password(guess, DUMMY_HASH) is False

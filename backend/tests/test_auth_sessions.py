import base64
import os
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from pos import config
from pos.auth.routes import build_auth_router
from pos.auth.store import UserStore
from pos.auth.tokens import REFRESH_TOKEN_TTL, create_access_token, new_refresh_token
from pos.db import create_pool, init_schema

_pool = None
PW = "pw-12345678"


@pytest.fixture(autouse=True)
def _secrets(monkeypatch):
    monkeypatch.setattr(config, "JWT_SECRET", "test-secret-" + "x" * 32)
    monkeypatch.setattr(config, "ENCRYPTION_KEY", base64.b64encode(os.urandom(32)).decode())

    async def _no_mail(email, link_url):
        return None

    monkeypatch.setattr("pos.auth.routes.send_magic_link_email", _no_mail)


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
    return app, users


def _device(app, agent="Browser/1.0"):
    """One browser: its own cookie jar, its own user agent."""
    return TestClient(app, headers={"user-agent": agent})


def _signup(client, email="a@test.com", username="user_a"):
    resp = client.post("/auth/signup", json={"name": "A", "username": username, "email": email, "password": PW})
    assert resp.status_code == 200
    return resp.json()["id"]


def _login(client, email="a@test.com"):
    assert client.post("/auth/login", json={"email": email, "password": PW}).status_code == 200


def _sessions(client):
    resp = client.get("/auth/sessions")
    assert resp.status_code == 200
    return resp.json()["sessions"]


def test_signing_in_creates_a_listed_session_that_records_the_device():
    app, _ = _app()
    a = _device(app, "Firefox/130 (Linux)")
    _signup(a)

    sessions = _sessions(a)

    assert len(sessions) == 1
    assert sessions[0]["current"] is True
    assert sessions[0]["user_agent"] == "Firefox/130 (Linux)"
    assert {"id", "created_at", "last_seen_at", "ip"} <= set(sessions[0])


def test_each_sign_in_is_its_own_session_and_only_this_device_is_current():
    app, _ = _app()
    a, b = _device(app, "Laptop"), _device(app, "Phone")
    _signup(a)
    _login(b)

    from_b = _sessions(b)

    assert sorted(s["user_agent"] for s in from_b) == ["Laptop", "Phone"]
    assert [s["user_agent"] for s in from_b if s["current"]] == ["Phone"]


def test_revoking_another_device_kills_its_access_token_at_once_and_leaves_this_one_alone():
    app, _ = _app()
    a, b = _device(app, "Laptop"), _device(app, "Phone")
    _signup(a)
    _login(b)
    laptop_id = next(s["id"] for s in _sessions(b) if s["user_agent"] == "Laptop")

    assert b.delete(f"/auth/sessions/{laptop_id}").status_code == 200

    assert a.get("/auth/me").status_code == 401                 # the revoked device is out immediately
    assert b.get("/auth/me").status_code == 200                 # this one is untouched


def test_a_revoked_device_trying_to_refresh_does_not_sign_everyone_out():
    """Revocation must not look like token theft: replay detection revokes ALL sessions."""
    app, _ = _app()
    a, b = _device(app, "Laptop"), _device(app, "Phone")
    _signup(a)
    _login(b)
    laptop_id = next(s["id"] for s in _sessions(b) if s["user_agent"] == "Laptop")
    b.delete(f"/auth/sessions/{laptop_id}")

    assert a.post("/auth/refresh").status_code == 401           # the revoked device cannot come back
    assert b.get("/auth/me").status_code == 200                 # ...and that did not cost this device its session


def test_logout_kills_the_access_token_not_just_the_browser_cookie():
    app, _ = _app()
    a = _device(app)
    _signup(a)
    stolen = a.cookies.get("pos_access")

    a.post("/auth/logout")

    thief = _device(app)
    thief.cookies.set("pos_access", stolen)
    assert thief.get("/auth/me").status_code == 401


def test_one_user_cannot_list_or_revoke_anothers_sessions():
    app, _ = _app()
    a, b = _device(app, "A-laptop"), _device(app, "B-laptop")
    _signup(a, "a@test.com", "user_a")
    _signup(b, "b@test.com", "user_b")
    a_session = _sessions(a)[0]["id"]

    assert [s["user_agent"] for s in _sessions(b)] == ["B-laptop"]        # only their own
    assert b.delete(f"/auth/sessions/{a_session}").status_code == 404     # not found, not forbidden: no probing
    assert a.get("/auth/me").status_code == 200


def test_unknown_session_ids_are_404():
    app, _ = _app()
    a = _device(app)
    _signup(a)
    assert a.delete("/auth/sessions/00000000-0000-0000-0000-000000000000").status_code == 404
    assert a.delete("/auth/sessions/not-a-uuid").status_code == 404


def test_revoke_others_keeps_the_current_session():
    app, _ = _app()
    a, b, c = _device(app, "1"), _device(app, "2"), _device(app, "3")
    _signup(a)
    _login(b)
    _login(c)

    resp = c.post("/auth/sessions/revoke-others")

    assert resp.json() == {"revoked": 2}
    assert a.get("/auth/me").status_code == 401 and b.get("/auth/me").status_code == 401
    assert c.get("/auth/me").status_code == 200
    assert [s["user_agent"] for s in _sessions(c)] == ["3"]


def test_revoking_your_own_current_session_signs_you_out_and_clears_the_cookies():
    app, _ = _app()
    a = _device(app)
    _signup(a)
    own = _sessions(a)[0]["id"]

    resp = a.delete(f"/auth/sessions/{own}")

    assert resp.status_code == 200
    assert a.get("/auth/me").status_code == 401


def test_refresh_keeps_the_same_session_and_bumps_last_seen():
    app, _ = _app()
    a = _device(app)
    _signup(a)
    before = _sessions(a)[0]

    assert a.post("/auth/refresh").status_code == 200
    after = _sessions(a)

    assert len(after) == 1 and after[0]["id"] == before["id"]
    assert after[0]["last_seen_at"] >= before["last_seen_at"]


def test_a_refresh_token_from_before_sessions_existed_is_upgraded():
    app, users = _app()
    a = _device(app)
    user_id = _signup(a)
    raw, token_hash = new_refresh_token()
    users.store_refresh_token(user_id, token_hash, datetime.now(UTC) + REFRESH_TOKEN_TTL)   # no session: legacy row
    legacy = _device(app, "Legacy")
    legacy.cookies.set("pos_refresh", raw)

    assert legacy.post("/auth/refresh").status_code == 200

    assert legacy.get("/auth/me").status_code == 200
    assert "Legacy" in [s["user_agent"] for s in _sessions(legacy)]


def test_an_access_token_without_a_session_id_is_rejected():
    app, _ = _app()
    a = _device(app)
    user_id = _signup(a)
    old_style = create_access_token(user_id)          # what tokens looked like before sessions existed

    stranger = _device(app)
    stranger.cookies.set("pos_access", old_style)

    assert stranger.get("/auth/me").status_code == 401


def test_an_expired_or_forged_session_id_is_rejected():
    app, _ = _app()
    a = _device(app)
    user_id = _signup(a)
    forged = create_access_token(user_id, "00000000-0000-0000-0000-000000000000")

    stranger = _device(app)
    stranger.cookies.set("pos_access", forged)

    assert stranger.get("/auth/me").status_code == 401


def test_deleting_the_user_removes_their_sessions():
    app, users = _app()
    a = _device(app)
    user_id = _signup(a)
    with users._pool.connection() as conn:
        conn.execute("DELETE FROM users WHERE id = %s", (user_id,))
        left = conn.execute("SELECT count(*) AS n FROM auth_sessions").fetchone()["n"]
    assert left == 0


def test_sessions_routes_require_authentication():
    app, _ = _app()
    anon = _device(app)
    assert anon.get("/auth/sessions").status_code == 401
    assert anon.delete("/auth/sessions/x").status_code == 401
    assert anon.post("/auth/sessions/revoke-others").status_code == 401


def test_a_stale_tab_reusing_a_rotated_refresh_token_still_revokes_everything():
    """Theft detection is unchanged for genuine replays."""
    app, _ = _app()
    a, b = _device(app, "Laptop"), _device(app, "Phone")
    _signup(a)
    _login(b)
    old_refresh = a.cookies.get("pos_refresh")
    a.post("/auth/refresh")                              # rotates: old_refresh is now spent

    replayer = _device(app)
    replayer.cookies.set("pos_refresh", old_refresh)
    assert replayer.post("/auth/refresh").status_code == 401

    assert b.get("/auth/me").status_code == 401           # the whole family was revoked


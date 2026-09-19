import base64
import os

import pytest
from fakes import FakeLlm, FakeStt, FakeTts, FakeVad
from starlette.testclient import TestClient
from test_server import _fresh_stores

from pos import config
from pos.cli.server import create_app

PW = "pw-12345678"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(config, "JWT_SECRET", "test-secret-" + "x" * 32)
    monkeypatch.setattr(config, "ENCRYPTION_KEY", base64.b64encode(os.urandom(32)).decode())


def _world():
    chats, users = _fresh_stores()
    app = create_app(
        stt=FakeStt("x"), tts_engines={"kokoro": FakeTts}, llm_factory=lambda m: FakeLlm("an answer"),
        vad_factory=lambda **kw: FakeVad(1, 3), default_tts_engine="kokoro", session_store=chats, user_store=users,
    )
    return app, users, chats


def _signup(app, email, username, password=PW):
    client = TestClient(app)
    body = {"name": "Name " + username, "username": username, "email": email}
    if password:
        body["password"] = password
    resp = client.post("/auth/signup", json=body)
    assert resp.status_code == 200
    return client, resp.json()["id"]


# --- profile --------------------------------------------------------------------------------------------

def test_me_includes_the_profile_fields_the_settings_page_shows():
    app, _, _ = _world()
    client, _ = _signup(app, "a@test.com", "user_a")

    me = client.get("/auth/me").json()

    assert me["email"] == "a@test.com" and me["username"] == "user_a" and me["name"] == "Name user_a"
    assert me["email_verified"] is False and me["has_password"] is True and "created_at" in me
    assert "password_hash" not in me


def test_name_and_username_can_be_changed_together_or_alone():
    app, _, _ = _world()
    client, _ = _signup(app, "a@test.com", "user_a")

    assert client.patch("/auth/me", json={"name": "New Name"}).json()["name"] == "New Name"
    assert client.patch("/auth/me", json={"username": "fresh_name"}).json()["username"] == "fresh_name"
    both = client.patch("/auth/me", json={"name": "Third", "username": "third_one"}).json()
    assert (both["name"], both["username"]) == ("Third", "third_one")
    assert client.get("/auth/me").json()["username"] == "third_one"


def test_a_taken_username_is_refused_case_insensitively_and_keeps_the_old_one():
    app, _, _ = _world()
    _signup(app, "b@test.com", "Taken_Name")
    client, _ = _signup(app, "a@test.com", "user_a")

    assert client.patch("/auth/me", json={"username": "taken_name"}).status_code == 409
    assert client.get("/auth/me").json()["username"] == "user_a"


def test_keeping_your_own_username_is_not_a_conflict():
    app, _, _ = _world()
    client, _ = _signup(app, "a@test.com", "user_a")

    assert client.patch("/auth/me", json={"username": "user_a", "name": "Same"}).status_code == 200


@pytest.mark.parametrize("body", [{"username": "ab"}, {"username": "has space"}, {"username": "x" * 40}, {"name": ""}])
def test_profile_values_follow_the_same_rules_as_signup(body):
    app, _, _ = _world()
    client, _ = _signup(app, "a@test.com", "user_a")

    assert client.patch("/auth/me", json=body).status_code == 422


def test_the_email_cannot_be_changed_through_the_profile():
    app, _, _ = _world()
    client, _ = _signup(app, "a@test.com", "user_a")

    client.patch("/auth/me", json={"email": "someone-else@test.com", "name": "N"})

    assert client.get("/auth/me").json()["email"] == "a@test.com"


def test_editing_a_profile_only_touches_the_callers_own_account():
    app, _, _ = _world()
    a, _ = _signup(app, "a@test.com", "user_a")
    b, _ = _signup(app, "b@test.com", "user_b")

    a.patch("/auth/me", json={"name": "Changed"})

    assert b.get("/auth/me").json()["name"] == "Name user_b"


def test_account_routes_require_a_session():
    app, _, _ = _world()
    anon = TestClient(app)
    assert anon.patch("/auth/me", json={"name": "x"}).status_code == 401
    assert anon.get("/auth/export").status_code == 401
    assert anon.post("/auth/account/delete", json={"confirm_email": "a@test.com"}).status_code == 401


# --- export ---------------------------------------------------------------------------------------------

def test_export_contains_only_the_callers_chats_and_never_secrets():
    app, users, chats = _world()
    a, a_id = _signup(app, "a@test.com", "user_a")
    _, b_id = _signup(app, "b@test.com", "user_b")
    chats.create_session("a-chat", a_id, mode="text", tts_engine=None, llm_model="m")
    chats.add_turn("a-chat", "user", "my question")
    chats.add_turn("a-chat", "assistant", "an answer")
    chats.create_session("b-chat", b_id, mode="text", tts_engine=None, llm_model="m")
    chats.add_turn("b-chat", "user", "someone else's private words")
    users.save_credential(a_id, "openai", {"OPENAI_API_KEY": "sk-my-secret-key"})

    resp = a.get("/auth/export")

    assert resp.status_code == 200
    assert "attachment" in resp.headers["content-disposition"]
    body = resp.json()
    assert body["user"]["email"] == "a@test.com"
    assert [s["id"] for s in body["sessions"]] == ["a-chat"]
    assert [t["text"] for t in body["sessions"][0]["turns"]] == ["my question", "an answer"]
    assert "someone else's private words" not in resp.text
    assert "sk-my-secret-key" not in resp.text and "password_hash" not in resp.text and "argon2" not in resp.text


def test_export_of_an_account_with_no_chats_is_valid():
    app, _, _ = _world()
    client, _ = _signup(app, "a@test.com", "user_a")

    assert client.get("/auth/export").json()["sessions"] == []


# --- deletion -------------------------------------------------------------------------------------------

def test_deleting_an_account_needs_the_email_typed_and_the_password():
    app, _, _ = _world()
    client, _ = _signup(app, "a@test.com", "user_a")

    def delete(**body):
        return client.post("/auth/account/delete", json=body).status_code

    assert delete(confirm_email="wrong@test.com", password=PW) == 400
    assert delete(confirm_email="a@test.com") == 401
    assert delete(confirm_email="a@test.com", password="nope-nope-1") == 401
    assert client.get("/auth/me").status_code == 200            # still here after every refusal


def test_a_confirmed_deletion_removes_the_account_and_everything_in_it():
    app, users, chats = _world()
    a, a_id = _signup(app, "a@test.com", "user_a")
    b, b_id = _signup(app, "b@test.com", "user_b")
    chats.create_session("a-chat", a_id, mode="text", tts_engine=None, llm_model="m")
    chats.add_turn("a-chat", "user", "hello")
    users.save_credential(a_id, "openai", {"OPENAI_API_KEY": "k"})
    users.set_tool_enabled(a_id, "web_search", False)
    chats.create_session("b-chat", b_id, mode="text", tts_engine=None, llm_model="m")

    resp = a.post("/auth/account/delete", json={"confirm_email": "A@Test.com", "password": PW})

    assert resp.status_code == 200
    assert a.get("/auth/me").status_code == 401                                   # signed out
    assert TestClient(app).post("/auth/login", json={"email": "a@test.com", "password": PW}).status_code == 401
    assert users.get_user_by_id(a_id) is None
    assert chats.get_session("a-chat", a_id) is None
    assert users.list_credential_providers(a_id) == [] and users.get_tool_settings(a_id) == {}
    assert b.get("/auth/me").status_code == 200                                   # nobody else was touched
    assert chats.get_session("b-chat", b_id) is not None


def test_a_passwordless_account_confirms_with_the_typed_email_alone(monkeypatch):
    sent = []

    async def capture(email, link_url):
        sent.append(link_url)

    monkeypatch.setattr("pos.auth.routes.send_magic_link_email", capture)
    app, users, _ = _world()
    with users._pool.connection() as conn:     # a link from another test inside the 60s cooldown would suppress ours
        conn.execute("DELETE FROM magic_link_tokens")
    client = TestClient(app)
    client.post("/auth/signup", json={"name": "N", "username": "user_n", "email": "n@test.com"})
    client.post("/auth/magic-link/verify", json={"token": sent[-1].rsplit("token=", 1)[1]})
    assert client.get("/auth/me").json()["has_password"] is False

    assert client.post("/auth/account/delete", json={"confirm_email": "wrong@test.com"}).status_code == 400
    assert client.post("/auth/account/delete", json={"confirm_email": "n@test.com"}).status_code == 200
    assert users.get_user_by_email("n@test.com") is None


def test_the_deleted_accounts_devices_stop_working():
    app, _, _ = _world()
    laptop, _ = _signup(app, "a@test.com", "user_a")
    phone = TestClient(app)
    phone.post("/auth/login", json={"email": "a@test.com", "password": PW})

    laptop.post("/auth/account/delete", json={"confirm_email": "a@test.com", "password": PW})

    assert phone.get("/auth/me").status_code == 401 and phone.post("/auth/refresh").status_code == 401

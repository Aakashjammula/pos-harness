import base64
import json
import os

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from pos import config
from pos.auth.routes import build_auth_router
from pos.auth.store import UserStore
from pos.db import create_pool, init_schema

_DSN = os.environ.get("TEST_DATABASE_URL", "postgresql://pos:pos@localhost:5432/pos")

_pool = None


@pytest.fixture(autouse=True)
def _secrets(monkeypatch):
    monkeypatch.setattr(config, "JWT_SECRET", "test-secret")
    monkeypatch.setattr(config, "ENCRYPTION_KEY", base64.b64encode(os.urandom(32)).decode())


sent_magic_links: list[tuple[str, str]] = []


@pytest.fixture(autouse=True)
def _stub_mailer(monkeypatch):
    """Never hit real SMTP in tests -- record what would have been sent."""
    sent_magic_links.clear()

    async def _fake_send(email: str, link_url: str) -> None:
        sent_magic_links.append((email, link_url))

    monkeypatch.setattr("pos.auth.routes.send_magic_link_email", _fake_send)


def _shared_pool():
    """One ConnectionPool reused by every test in this module -- a fresh
    pool per test exhausts Postgres's default max_connections=100 once
    enough tests accumulate in one process."""
    global _pool
    if _pool is None:
        _pool = create_pool(_DSN, max_size=5)
        init_schema(_pool)
    return _pool


def _client():
    pool = _shared_pool()
    with pool.connection() as conn:
        conn.execute("TRUNCATE users, magic_link_tokens RESTART IDENTITY CASCADE")
    app = FastAPI()
    app.include_router(build_auth_router(UserStore(pool)))
    return TestClient(app)


def _signup(client, email: str, password: str | None = "pw-12345678"):
    """A valid signup request. Signup requires a name and a unique username
    as well as email; derive both from the email so two different emails in
    one test never collide on username."""
    local = "".join(c for c in email.split("@")[0] if c.isalnum())
    body = {"name": "Test User", "username": f"user_{local}", "email": email}
    if password is not None:
        body["password"] = password
    return client.post("/auth/signup", json=body)


def test_signup_sets_cookies_and_returns_the_user():
    client = _client()
    resp = _signup(client, "a@test.com", "pw-12345678")

    assert resp.status_code == 200
    assert resp.json()["email"] == "a@test.com"
    assert "pos_access" in resp.cookies
    assert "pos_refresh" in resp.cookies


def test_signup_never_returns_the_password_or_its_hash():
    client = _client()
    body = _signup(client, "a@test.com", "pw-12345678").text
    assert "pw-12345678" not in body
    assert "argon2" not in body


def test_duplicate_signup_is_rejected():
    client = _client()
    _signup(client, "a@test.com", "pw-12345678")
    resp = _signup(client, "a@test.com", "pw-12345678")
    assert resp.status_code == 409


def test_short_password_is_rejected():
    client = _client()
    resp = _signup(client, "a@test.com", "short")
    assert resp.status_code == 422


def test_login_with_the_right_password_succeeds():
    client = _client()
    _signup(client, "a@test.com", "pw-12345678")
    client.cookies.clear()

    resp = client.post("/auth/login", json={"email": "a@test.com", "password": "pw-12345678"})

    assert resp.status_code == 200
    assert "pos_access" in resp.cookies


def test_login_with_the_wrong_password_fails():
    client = _client()
    _signup(client, "a@test.com", "pw-12345678")
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
    _signup(client, "a@test.com", "pw-12345678")
    assert client.get("/auth/me").json()["email"] == "a@test.com"


def test_refresh_rotates_the_token_and_the_old_one_stops_working():
    client = _client()
    _signup(client, "a@test.com", "pw-12345678")
    first_refresh = client.cookies["pos_refresh"]

    assert client.post("/auth/refresh").status_code == 200

    client.cookies.set("pos_refresh", first_refresh)
    assert client.post("/auth/refresh").status_code == 401


def test_replaying_an_old_refresh_token_kills_the_whole_family():
    client = _client()
    _signup(client, "a@test.com", "pw-12345678")
    stolen = client.cookies["pos_refresh"]
    client.post("/auth/refresh")
    live = client.cookies["pos_refresh"]

    client.cookies.set("pos_refresh", stolen)
    assert client.post("/auth/refresh").status_code == 401

    client.cookies.set("pos_refresh", live)
    assert client.post("/auth/refresh").status_code == 401


def test_logout_clears_cookies_and_ends_the_session():
    client = _client()
    _signup(client, "a@test.com", "pw-12345678")

    assert client.post("/auth/logout").status_code == 200

    assert client.get("/auth/me").status_code == 401


def test_credentials_are_per_user_and_never_returned():
    client = _client()
    _signup(client, "a@test.com", "pw-12345678")
    client.put("/credentials/openai", json={"openai_api_key": "sk-secret"})

    listed = client.get("/credentials")
    assert listed.json()["configured"] == ["openai"]
    assert "sk-secret" not in listed.text

    client.post("/auth/logout")
    _signup(client, "b@test.com", "pw-12345678")
    assert client.get("/credentials").json()["configured"] == []


def test_delete_credential():
    client = _client()
    _signup(client, "a@test.com", "pw-12345678")
    client.put("/credentials/openai", json={"openai_api_key": "sk-secret"})

    assert client.delete("/credentials/openai").status_code == 200
    assert client.get("/credentials").json()["configured"] == []
    assert client.delete("/credentials/openai").status_code == 404


def test_credentials_require_authentication():
    client = _client()
    assert client.get("/credentials").status_code == 401
    assert client.put("/credentials/openai", json={"openai_api_key": "x"}).status_code == 401


def _requested_token(email: str) -> str:
    """The tests can't read the real email, so pull the raw token out of
    the fake mailer's recorded link_url instead."""
    _, link_url = next(pair for pair in sent_magic_links if pair[0] == email)
    return link_url.rsplit("token=", 1)[1]


def test_magic_link_request_always_returns_ok(monkeypatch):
    client = _client()
    resp = client.post("/auth/magic-link/request", json={"email": "new@test.com"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert len(sent_magic_links) == 1


def test_magic_link_verify_creates_a_passwordless_account_and_signs_in():
    client = _client()
    client.post("/auth/magic-link/request", json={"email": "new@test.com"})
    token = _requested_token("new@test.com")

    resp = client.post("/auth/magic-link/verify", json={"token": token})

    assert resp.status_code == 200
    assert resp.json()["email"] == "new@test.com"
    assert "pos_access" in resp.cookies
    assert client.get("/auth/me").json()["email"] == "new@test.com"


def test_magic_link_verify_signs_in_an_existing_account():
    client = _client()
    signup = _signup(client, "a@test.com", "pw-12345678")
    existing_id = signup.json()["id"]
    client.cookies.clear()

    client.post("/auth/magic-link/request", json={"email": "a@test.com"})
    token = _requested_token("a@test.com")
    resp = client.post("/auth/magic-link/verify", json={"token": token})

    assert resp.json()["id"] == existing_id


def test_magic_link_token_is_single_use():
    client = _client()
    client.post("/auth/magic-link/request", json={"email": "new@test.com"})
    token = _requested_token("new@test.com")

    assert client.post("/auth/magic-link/verify", json={"token": token}).status_code == 200
    assert client.post("/auth/magic-link/verify", json={"token": token}).status_code == 401


def test_unknown_magic_link_token_is_rejected():
    client = _client()
    assert client.post("/auth/magic-link/verify", json={"token": "not-a-real-token"}).status_code == 401


def test_repeated_magic_link_requests_are_throttled():
    client = _client()
    client.post("/auth/magic-link/request", json={"email": "new@test.com"})
    client.post("/auth/magic-link/request", json={"email": "new@test.com"})

    assert len(sent_magic_links) == 1


def test_login_rejects_password_for_a_magic_link_only_account():
    client = _client()
    client.post("/auth/magic-link/request", json={"email": "new@test.com"})
    token = _requested_token("new@test.com")
    client.post("/auth/magic-link/verify", json={"token": token})
    client.post("/auth/logout")

    resp = client.post("/auth/login", json={"email": "new@test.com", "password": "anything123"})
    assert resp.status_code == 401


def test_models_endpoint_lists_with_the_saved_credential_and_never_returns_it(monkeypatch):
    seen = {}

    def fake_list(provider, env, **kw):
        seen.update(provider=provider, env=dict(env))
        return [{"id": "gemini-a", "label": "Gemini A"}]

    monkeypatch.setattr("pos.auth.routes.list_models", fake_list)
    client = _client()
    _signup(client, "a@test.com")
    client.put("/credentials/gemini", json={"gemini_api_key": "g-secret"})

    resp = client.get("/credentials/gemini/models")

    assert resp.status_code == 200
    assert resp.json() == {"models": [{"id": "gemini-a", "label": "Gemini A"}]}
    assert seen["provider"] == "gemini"
    assert seen["env"]["GOOGLE_API_KEY"] == "g-secret"
    assert "g-secret" not in resp.text


def test_models_endpoint_requires_auth_a_saved_credential_and_a_listable_provider(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)     # a server-wide key would make it listable
    client = _client()
    assert client.get("/credentials/openai/models").status_code == 401

    _signup(client, "a@test.com")
    assert client.get("/credentials/openai/models").status_code == 404       # nothing saved yet
    assert client.get("/credentials/tavily/models").status_code == 404       # not an LLM provider


def test_models_endpoint_turns_provider_failures_into_502(monkeypatch):
    from pos.llm.model_listing import ModelListError

    def fake_list(provider, env, **kw):
        raise ModelListError("the provider rejected this key (unauthorized)")

    monkeypatch.setattr("pos.auth.routes.list_models", fake_list)
    client = _client()
    _signup(client, "a@test.com")
    client.put("/credentials/openai", json={"openai_api_key": "sk-bad"})

    resp = client.get("/credentials/openai/models")

    assert resp.status_code == 502
    assert "rejected this key" in resp.json()["detail"]


def test_tools_list_reports_key_requirement_and_configured_state(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    client = _client()
    _signup(client, "a@test.com")

    tools = {t["id"]: t for t in client.get("/tools").json()["tools"]}

    assert tools["get_current_time"]["requires_key"] is False
    assert tools["get_current_time"]["active"] is True
    ws = tools["web_search"]
    assert ws["requires_key"] is True and ws["credential_provider"] == "tavily"
    assert ws["credential_fields"] == ["tavily_api_key"]
    assert ws["configured"] is False and ws["active"] is False

    client.put("/credentials/tavily", json={"tavily_api_key": "tvly-x"})
    ws = {t["id"]: t for t in client.get("/tools").json()["tools"]}["web_search"]
    assert ws["configured"] is True and ws["active"] is True


def test_a_tool_can_be_switched_off_and_on_per_user():
    client = _client()
    _signup(client, "a@test.com")

    assert client.put("/tools/get_current_time", json={"enabled": False}).json() == {
        "id": "get_current_time", "enabled": False,
    }
    assert {t["id"]: t for t in client.get("/tools").json()["tools"]}["get_current_time"]["active"] is False

    client.put("/tools/get_current_time", json={"enabled": True})
    assert {t["id"]: t for t in client.get("/tools").json()["tools"]}["get_current_time"]["active"] is True


def test_enabling_a_key_tool_without_its_key_is_refused_and_unknown_tools_404(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    client = _client()
    _signup(client, "a@test.com")

    assert client.put("/tools/web_search", json={"enabled": True}).status_code == 409
    assert client.put("/tools/nope", json={"enabled": True}).status_code == 404
    assert client.put("/tools/web_search", json={"enabled": False}).status_code == 200   # disabling always ok


def test_one_users_tool_switches_do_not_affect_another():
    client = _client()
    _signup(client, "a@test.com")
    client.put("/tools/get_current_time", json={"enabled": False})
    client.post("/auth/logout")

    _signup(client, "b@test.com")
    tools = {t["id"]: t for t in client.get("/tools").json()["tools"]}

    assert tools["get_current_time"]["active"] is True


def test_tools_routes_require_auth():
    client = _client()
    assert client.get("/tools").status_code == 401
    assert client.put("/tools/web_search", json={"enabled": True}).status_code == 401


def test_passwordless_signup_reports_when_the_email_could_not_be_sent(monkeypatch):
    async def boom(email, link_url):
        raise RuntimeError("smtp down")

    monkeypatch.setattr("pos.auth.routes.send_magic_link_email", boom)
    client = _client()

    resp = _signup(client, "a@test.com", password=None)

    assert resp.status_code == 200                      # the account really was created
    body = resp.json()
    assert body["magic_link_sent"] is False
    assert "email" in body["message"].lower()
    assert "pos_access" not in resp.cookies             # and it still must not sign anyone in


def test_passwordless_signup_reports_success_when_the_email_is_sent():
    client = _client()

    body = _signup(client, "a@test.com", password=None).json()

    assert body["magic_link_sent"] is True and "message" not in body


def test_a_throttled_repeat_request_still_counts_as_sent():
    client = _client()
    client.post("/auth/magic-link/request", json={"email": "a@test.com"})    # a link is already outstanding

    body = _signup(client, "a@test.com", password=None).json()

    assert body["magic_link_sent"] is True                                    # the user does have a link coming


def test_signup_with_a_password_is_unaffected_by_mail_problems(monkeypatch):
    async def boom(email, link_url):
        raise RuntimeError("smtp down")

    monkeypatch.setattr("pos.auth.routes.send_magic_link_email", boom)
    client = _client()

    resp = _signup(client, "a@test.com")                # has a password: no email involved

    assert resp.json()["magic_link_sent"] is False and "message" not in resp.json()
    assert "pos_access" in resp.cookies


def test_models_endpoint_uses_a_key_set_on_the_server_when_the_user_saved_none(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-server")
    seen = {}

    def fake_list(provider, env, **kw):
        seen["key"] = env.get("OPENAI_API_KEY")
        return [{"id": "m", "label": "m"}]

    monkeypatch.setattr("pos.auth.routes.list_models", fake_list)
    client = _client()
    _signup(client, "a@test.com")

    resp = client.get("/credentials/openai/models")

    assert resp.status_code == 200 and resp.json() == {"models": [{"id": "m", "label": "m"}]}
    assert seen["key"] == "sk-server"
    assert "sk-server" not in resp.text


def test_a_users_own_key_wins_over_the_servers_for_listing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-server")
    seen = {}
    monkeypatch.setattr(
        "pos.auth.routes.list_models", lambda provider, env, **kw: seen.update(key=env.get("OPENAI_API_KEY")) or []
    )
    client = _client()
    _signup(client, "a@test.com")
    client.put("/credentials/openai", json={"openai_api_key": "sk-mine"})

    client.get("/credentials/openai/models")

    assert seen["key"] == "sk-mine"


def test_models_endpoint_404s_when_neither_the_user_nor_the_server_has_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = _client()
    _signup(client, "a@test.com")

    assert client.get("/credentials/openai/models").status_code == 404


def test_credentials_list_shows_saved_settings_but_never_a_key():
    client = _client()
    _signup(client, "a@test.com")
    client.put(
        "/credentials/local", json={"local_base_url": "http://192.168.1.5:1234/v1", "local_api_key": "lm-secret"}
    )
    client.put("/credentials/azure", json={"azure_api_key": "az-secret", "azure_endpoint": "https://r.openai.azure.com/",
                                          "azure_deployment": "my-dep"})
    client.put("/credentials/openai", json={"openai_api_key": "sk-secret"})

    body = client.get("/credentials").json()

    assert body["configured"] == ["azure", "local", "openai"]
    assert body["public"]["local"] == {"LOCAL_BASE_URL": "http://192.168.1.5:1234/v1"}
    assert body["public"]["azure"] == {"AZURE_OPENAI_ENDPOINT": "https://r.openai.azure.com/",
                                       "AZURE_OPENAI_DEPLOYMENT": "my-dep"}
    assert "openai" not in body["public"]                          # a provider with only a key shows nothing
    for secret in ("lm-secret", "az-secret", "sk-secret"):
        assert secret not in json.dumps(body)


def test_one_users_saved_settings_are_not_visible_to_another():
    a = _client()
    _signup(a, "a@test.com")
    a.put("/credentials/local", json={"local_base_url": "http://192.168.1.5:1234/v1"})
    b = TestClient(a.app)
    _signup(b, "b@test.com")

    assert b.get("/credentials").json() == {"configured": [], "public": {}, "hints": {}}


def test_a_saved_key_is_shown_only_as_its_last_four_characters():
    client = _client()
    _signup(client, "a@test.com")
    client.put("/credentials/openai", json={"openai_api_key": "sk-proj-abcdefghij1234"})
    client.put("/credentials/gemini", json={"gemini_api_key": "AIzaSyVeryLongSecretKeyWXYZ"})

    body = client.get("/credentials").json()

    assert body["hints"] == {"gemini": "••••WXYZ", "openai": "••••1234"}
    text = json.dumps(body)
    assert "sk-proj-abcdefghij" not in text and "AIzaSyVeryLong" not in text     # nothing but the last four


def test_a_short_secret_gets_no_hint_at_all():
    client = _client()
    _signup(client, "a@test.com")
    client.put("/credentials/openai", json={"openai_api_key": "short-key-1"})   # 11 characters

    assert client.get("/credentials").json()["hints"] == {}


def test_the_models_endpoint_returns_every_model_with_a_chat_flag(monkeypatch):
    seen = {}

    def fake_list(provider, env, include_all=False):
        seen["include_all"] = include_all
        return [{"id": "gemini-2.5-flash", "label": "Flash", "chat": True},
                {"id": "lyria-3", "label": "Lyria", "chat": False}]

    monkeypatch.setattr("pos.auth.routes.list_models", fake_list)
    client = _client()
    _signup(client, "a@test.com")
    client.put("/credentials/gemini", json={"gemini_api_key": "g-secret-key-123456"})

    body = client.get("/credentials/gemini/models").json()

    assert seen["include_all"] is True                     # the UI decides what to show, so it gets it all
    assert [(m["id"], m["chat"]) for m in body["models"]] == [("gemini-2.5-flash", True), ("lyria-3", False)]

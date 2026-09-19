import base64
import os

import pytest
from fakes import FakeLlm, FakeStt, FakeTts, FakeVad
from starlette.testclient import TestClient
from test_server import _fresh_stores, _sign_in

from pos import config
from pos.cli.server import create_app


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(config, "JWT_SECRET", "test-secret-" + "x" * 32)
    monkeypatch.setattr(config, "ENCRYPTION_KEY", base64.b64encode(os.urandom(32)).decode())


def _client(**flags):
    for name, value in flags.items():
        setattr(config, name, value)
    session_store, user_store = _fresh_stores()
    app = create_app(
        stt=FakeStt("x"), tts_engines={"kokoro": FakeTts}, llm_factory=lambda m: FakeLlm(),
        vad_factory=lambda **kw: FakeVad(1, 3), default_tts_engine="kokoro",
        session_store=session_store, user_store=user_store,
    )
    return TestClient(app)


_BASELINE = {
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "x-frame-options": "DENY",
    "content-security-policy": "default-src 'none'; frame-ancestors 'none'",
}


@pytest.mark.parametrize("method,path,status", [
    ("GET", "/options", 200),            # a normal response
    ("GET", "/auth/me", 401),            # an error response
    ("GET", "/does-not-exist", 404),     # a framework-generated one
    ("POST", "/auth/login", 422),        # a validation failure
])
def test_every_kind_of_response_carries_the_baseline_headers(method, path, status):
    client = _client()

    resp = client.request(method, path, json={} if method == "POST" else None)

    assert resp.status_code == status
    for name, value in _BASELINE.items():
        assert resp.headers.get(name) == value, name


@pytest.mark.parametrize("path", ["/auth/me", "/credentials", "/tools", "/sessions"])
def test_account_data_is_never_cached(path, monkeypatch):
    client = _client()
    _sign_in(client)

    resp = client.get(path)

    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "no-store"


def test_a_streamed_chat_keeps_its_own_cache_header_and_still_streams():
    client = _client()
    _sign_in(client)

    with client.stream("POST", "/chat/stream", json={"message": "hi"}) as resp:
        body = "".join(resp.iter_lines())

    assert resp.headers["cache-control"] == "no-cache"                # not overwritten
    assert resp.headers["x-content-type-options"] == "nosniff" and "event: done" in body


def test_hsts_is_sent_only_when_cookies_are_secure(monkeypatch):
    monkeypatch.setattr(config, "COOKIE_SECURE", False)
    assert "strict-transport-security" not in _client().get("/options").headers

    monkeypatch.setattr(config, "COOKIE_SECURE", True)
    assert _client().get("/options").headers["strict-transport-security"].startswith("max-age=")


def test_the_api_docs_are_off_by_default(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_API_DOCS", False)
    client = _client()

    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404, path


def test_the_api_docs_can_be_switched_on_and_are_not_blocked_by_the_csp(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_API_DOCS", True)
    client = _client()

    docs = client.get("/docs")

    assert docs.status_code == 200 and client.get("/openapi.json").status_code == 200
    assert "content-security-policy" not in docs.headers          # Swagger UI loads its assets from a CDN
    assert docs.headers["x-frame-options"] == "DENY"


def test_a_rejected_password_is_not_echoed_back_in_the_error():
    client = _client()

    resp = client.post("/auth/login", json={"email": "a@test.com", "password": "pw-x9!"})

    assert resp.status_code == 422
    assert "pw-x9!" not in resp.text
    detail = resp.json()["detail"]
    assert detail and set(detail[0]) == {"loc", "msg", "type"}       # still says which field and why


def test_no_submitted_value_ever_appears_in_a_validation_error():
    client = _client()

    resp = client.post("/auth/signup", json={
        "name": "n", "username": "bad name!!", "email": "not-an-email-XYZ", "password": "tiny-Q7",
    })

    assert resp.status_code == 422
    for secret in ("bad name!!", "not-an-email-XYZ", "tiny-Q7"):
        assert secret not in resp.text


def test_a_malformed_json_body_does_not_echo_it_either():
    client = _client()

    resp = client.post(
        "/auth/login", content=b'{"password": "leaky-ZZ9", ', headers={"content-type": "application/json"}
    )

    assert resp.status_code == 422 and "leaky-ZZ9" not in resp.text

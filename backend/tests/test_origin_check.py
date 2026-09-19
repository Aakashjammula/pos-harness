import base64
import os

import pytest
from fakes import FakeLlm, FakeStt, FakeTts, FakeVad
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from test_server import _fresh_stores, _sign_in

from pos import config
from pos.cli.server import create_app

FRONTEND = "http://localhost:3000"
EVIL = "https://evil.example"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(config, "JWT_SECRET", "test-secret-" + "x" * 32)
    monkeypatch.setattr(config, "ENCRYPTION_KEY", base64.b64encode(os.urandom(32)).decode())
    monkeypatch.setattr(config, "CORS_ORIGINS", [FRONTEND])


def _client():
    session_store, user_store = _fresh_stores()
    app = create_app(
        stt=FakeStt("x"), tts_engines={"kokoro": FakeTts}, llm_factory=lambda m: FakeLlm(),
        vad_factory=lambda **kw: FakeVad(1, 3), default_tts_engine="kokoro",
        session_store=session_store, user_store=user_store,
    )
    client = TestClient(app)
    _sign_in(client)
    return client


def test_a_state_changing_request_from_a_foreign_origin_is_refused():
    client = _client()

    resp = client.post("/auth/logout", headers={"Origin": EVIL})

    assert resp.status_code == 403
    assert client.get("/auth/me").status_code == 200            # and it did nothing


@pytest.mark.parametrize("method,path,body", [
    ("POST", "/chat/stream", {"message": "hi"}),
    ("PUT", "/credentials/openai", {"openai_api_key": "k"}),
    ("DELETE", "/sessions/x", None),
    ("PUT", "/tools/get_current_time", {"enabled": False}),
    ("POST", "/auth/sessions/revoke-others", None),
])
def test_every_kind_of_write_is_covered(method, path, body):
    client = _client()

    resp = client.request(method, path, json=body, headers={"Origin": EVIL})

    assert resp.status_code == 403


def test_the_real_frontend_origin_is_allowed():
    client = _client()

    assert client.post("/auth/logout", headers={"Origin": FRONTEND}).status_code == 200


def test_requests_with_no_origin_header_are_allowed_because_a_browser_always_sends_one_for_writes():
    """The CLI client, curl and tests send none; none of them can be tricked by a web page."""
    client = _client()

    assert client.post("/auth/refresh").status_code == 200


def test_the_null_origin_used_by_sandboxed_pages_is_refused():
    client = _client()

    assert client.post("/auth/logout", headers={"Origin": "null"}).status_code == 403


def test_origin_matching_ignores_case_and_a_trailing_slash_but_not_the_port():
    client = _client()

    assert client.post("/auth/refresh", headers={"Origin": "HTTP://LOCALHOST:3000/"}).status_code == 200
    assert client.post("/auth/refresh", headers={"Origin": "http://localhost:4000"}).status_code == 403


def test_reads_from_any_origin_are_left_to_cors():
    client = _client()

    resp = client.get("/options", headers={"Origin": EVIL})

    assert resp.status_code == 200
    assert "access-control-allow-origin" not in resp.headers        # the browser will not let the page read it


def test_the_cors_preflight_from_the_frontend_still_works():
    client = _client()

    resp = client.options(
        "/credentials", headers={"Origin": FRONTEND, "Access-Control-Request-Method": "PUT"}
    )

    assert resp.status_code == 200 and resp.headers["access-control-allow-origin"] == FRONTEND


def test_a_websocket_from_a_foreign_origin_is_refused_before_it_is_accepted():
    client = _client()

    with pytest.raises(WebSocketDisconnect) as info:
        with client.websocket_connect("/ws?mode=voice", headers={"Origin": EVIL}):
            pass

    assert info.value.code == 1008


def test_a_websocket_from_the_frontend_is_accepted():
    client = _client()

    with client.websocket_connect("/ws?mode=voice", headers={"Origin": FRONTEND}) as ws:
        assert ws.receive_json()["event"] == "ready"


def test_a_streamed_chat_from_the_frontend_still_works_through_the_middleware():
    """The check must not buffer or break server-sent events."""
    client = _client()

    with client.stream("POST", "/chat/stream", json={"message": "hi"}, headers={"Origin": FRONTEND}) as resp:
        body = "".join(resp.iter_lines())

    assert resp.status_code == 200 and "event: done" in body

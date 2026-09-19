import base64
import json
import os

import pytest
from fakes import FakeLlm, FakeStt, FakeTts
from starlette.testclient import TestClient
from test_server import _PROVIDER_ENV, _fresh_stores, _sign_in

from pos import config
from pos.cli.server import create_app


@pytest.fixture(autouse=True)
def _secrets(monkeypatch):
    monkeypatch.setattr(config, "JWT_SECRET", "test-secret")
    monkeypatch.setattr(config, "ENCRYPTION_KEY", base64.b64encode(os.urandom(32)).decode())


def _events(resp):
    """[(event, data)] from an SSE response body."""
    out, event = [], None
    for line in resp.iter_lines():
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: "):
            out.append((event, json.loads(line[6:])))
    return out


def _client(llm):
    session_store, user_store = _fresh_stores()
    app = create_app(
        stt=FakeStt("x"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: llm,
        default_tts_engine="kokoro",
        session_store=session_store,
        user_store=user_store,
    )
    client = TestClient(app)
    _sign_in(client)
    return client


def test_streams_session_then_tokens_then_done_and_saves_both_turns():
    client = _client(FakeLlm(reply="hello there", fake_title="Greeting"))

    with client.stream("POST", "/chat/stream", json={"message": "hi"}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = _events(resp)

    names = [n for n, _ in events]
    assert names[0] == "session"
    assert names.count("token") == 2                                   # FakeLlm yields one piece per word
    assert names.index("done") > max(i for i, n in enumerate(names) if n == "token")
    assert "".join(d["text"] for n, d in events if n == "token") == "hello there "
    done = next(d for n, d in events if n == "done")
    assert done["text"] == "hello there "
    assert next(d for n, d in events if n == "title") == {"title": "Greeting"}

    turns = client.get(f"/sessions/{events[0][1]['id']}").json()["turns"]
    assert [(t["role"], t["text"]) for t in turns] == [("user", "hi"), ("assistant", "hello there ")]


def test_done_carries_usage_and_latency():
    client = _client(FakeLlm(reply="ok", fake_usage={"total_tokens": 7}))

    with client.stream("POST", "/chat/stream", json={"message": "hi"}) as resp:
        done = next(d for n, d in _events(resp) if n == "done")

    assert done["usage"] == {"total_tokens": 7}
    assert set(done["latency"]) == {"ttft", "total"}


def test_continuing_a_session_sends_its_history_and_does_not_retitle():
    llm = FakeLlm(reply="ok", fake_title="First")
    client = _client(llm)
    with client.stream("POST", "/chat/stream", json={"message": "one"}) as resp:
        session_id = _events(resp)[0][1]["id"]

    with client.stream("POST", "/chat/stream", json={"message": "two", "session_id": session_id}) as resp:
        names = [n for n, _ in _events(resp)]

    assert llm.calls[-1][-1] == {"role": "user", "content": "two"}
    assert {"role": "user", "content": "one"} in llm.calls[-1]
    assert "title" not in names and len(llm.title_calls) == 1


def test_pre_stream_errors_use_plain_http_statuses():
    client = _client(FakeLlm())

    assert client.post("/chat/stream", json={"message": "   "}).status_code == 422
    assert client.post("/chat/stream", json={"message": "hi", "session_id": "nope"}).status_code == 404
    assert TestClient(client.app).post("/chat/stream", json={"message": "hi"}).status_code == 401


def test_a_session_belonging_to_someone_else_is_a_404():
    client = _client(FakeLlm())
    with client.stream("POST", "/chat/stream", json={"message": "hi"}) as resp:
        session_id = _events(resp)[0][1]["id"]
    client.post("/auth/logout")
    _sign_in(client, "other@test.com")

    assert client.post("/chat/stream", json={"message": "hi", "session_id": session_id}).status_code == 404


def test_no_llm_configured_is_409(monkeypatch):
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    session_store, user_store = _fresh_stores()
    app = create_app(          # the real LLM path with nothing configured
        stt=FakeStt("x"), tts_engines={"kokoro": FakeTts}, default_tts_engine="kokoro",
        session_store=session_store, user_store=user_store,
    )
    client = TestClient(app)
    _sign_in(client)

    resp = client.post("/chat/stream", json={"message": "hi"})

    assert resp.status_code == 409 and "no LLM configured" in resp.json()["detail"]


def test_a_failing_llm_emits_an_error_event_and_saves_no_assistant_turn():
    class Boom(FakeLlm):
        def stream(self, messages, cancel, usage=None):
            raise RuntimeError("provider exploded")
            yield  # pragma: no cover  (makes this a generator)

    client = _client(Boom())

    with client.stream("POST", "/chat/stream", json={"message": "hi"}) as resp:
        events = _events(resp)

    assert events[-1][0] == "error" and "provider exploded" in events[-1][1]["message"]
    assert "done" not in [n for n, _ in events]
    turns = client.get(f"/sessions/{events[0][1]['id']}").json()["turns"]
    assert [t["role"] for t in turns] == ["user"]


def test_block_list_content_streams_as_text():
    """The Gemini regression end to end: pieces reach the client as plain text."""
    from langchain_core.messages import AIMessageChunk

    from pos.llm.langchain_llm import LangChainLlm

    class Runnable:
        def stream(self, messages):
            yield AIMessageChunk(content=[{"type": "text", "text": "hel"}])
            yield AIMessageChunk(content=[{"type": "text", "text": "lo"}])

    llm = LangChainLlm.__new__(LangChainLlm)          # skip provider resolution: only stream() is under test
    llm.system_prompt, llm.max_tool_rounds, llm.tools, llm._tools_by_name = "s", 3, [], {}
    llm._runnable, llm._context_window = Runnable(), None
    llm.provider = type("P", (), {"name": "gemini", "model": "m"})()
    client = _client(llm)

    with client.stream("POST", "/chat/stream", json={"message": "hi"}) as resp:
        events = _events(resp)

    assert "".join(d["text"] for n, d in events if n == "token") == "hello"

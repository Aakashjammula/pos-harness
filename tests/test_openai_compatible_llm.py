import threading
from unittest.mock import MagicMock

from asr_test.llm.openai_compatible import OpenAiCompatibleLlm


def _fake_chunk(content):
    chunk = MagicMock()
    chunk.choices = [MagicMock(delta=MagicMock(content=content))]
    return chunk


def test_stream_is_stateless_and_prepends_system_prompt(monkeypatch):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = [
        _fake_chunk("hel"), _fake_chunk("lo"), _fake_chunk(None),
    ]
    monkeypatch.setattr("asr_test.llm.openai_compatible.OpenAI", lambda **kw: mock_client)

    llm = OpenAiCompatibleLlm(system_prompt="sys", warmup=False)
    cancel = threading.Event()
    result = list(llm.stream([{"role": "user", "content": "hi"}], cancel))

    assert result == ["hel", "lo"]
    sent = mock_client.chat.completions.create.call_args.kwargs["messages"]
    assert sent[0] == {"role": "system", "content": "sys"}
    assert sent[1] == {"role": "user", "content": "hi"}
    assert not hasattr(llm, "history")
    # Regression guard: last_ttft/last_total used to be shared mutable
    # instance state, which raced when one LLM instance was shared across
    # concurrent sessions (server.py's provider cache does this
    # deliberately for memory). Agent now measures timing locally around
    # stream() instead of reading it back from the engine afterward — the
    # engine should expose no such state at all.
    assert not hasattr(llm, "last_ttft")
    assert not hasattr(llm, "last_total")


def test_stream_stops_on_cancel(monkeypatch):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = [
        _fake_chunk("a"), _fake_chunk("b"), _fake_chunk("c"),
    ]
    monkeypatch.setattr("asr_test.llm.openai_compatible.OpenAI", lambda **kw: mock_client)

    llm = OpenAiCompatibleLlm(warmup=False)
    cancel = threading.Event()
    pieces = []
    for i, piece in enumerate(llm.stream([{"role": "user", "content": "hi"}], cancel)):
        pieces.append(piece)
        if i == 0:
            cancel.set()

    assert pieces == ["a"]


def test_stream_accepts_and_ignores_usage_param(monkeypatch):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = [_fake_chunk("hi"), _fake_chunk(None)]
    monkeypatch.setattr("asr_test.llm.openai_compatible.OpenAI", lambda **kw: mock_client)

    llm = OpenAiCompatibleLlm(warmup=False)
    cancel = threading.Event()
    usage = {}
    result = list(llm.stream([{"role": "user", "content": "hi"}], cancel, usage))

    assert result == ["hi"]
    assert usage == {}  # OpenAiCompatibleLlm doesn't populate it — accepted for interface conformance only

import threading
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessageChunk

from asr_test.llm.langchain_llm import LangChainLlm


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Default every test in this file to no real network call for the
    # context-window lookup (Task 2b's local branch hits LM Studio's
    # REST API) — test_context_window_resolved_once_at_construction
    # below overrides this per-test with its own monkeypatch.setattr.
    monkeypatch.setattr("asr_test.llm.langchain_llm.get_context_window", lambda provider: None)


def _text_chunk(content):
    return AIMessageChunk(content=content)


def _tool_call_chunk(name, args, call_id):
    return AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": name, "args": str(args), "id": call_id, "index": 0}],
    )


def _usage_chunk(input_tokens, output_tokens):
    chunk = AIMessageChunk(content="")
    chunk.usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    return chunk


class _FakeRunnable:
    """Stands in for `ChatOpenAI(...).bind_tools([...])`. `rounds` is a
    list of chunk-lists, one per stream() call — lets a test script a
    tool-call round followed by a final text round."""

    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.stream_calls = []

    def stream(self, messages):
        self.stream_calls.append(list(messages))
        return iter(self.rounds.pop(0))


def _make_llm(monkeypatch, runnable, tools=()):
    mock_model = MagicMock()
    mock_model.bind_tools.return_value = runnable
    mock_model.stream = runnable.stream  # used verbatim when tools=[] (no bind_tools wrapping)
    monkeypatch.setattr("asr_test.llm.langchain_llm.ChatOpenAI", lambda **kw: mock_model)
    return LangChainLlm(tools=list(tools), warmup=False)


def test_stream_yields_plain_text_with_no_tool_calls(monkeypatch):
    runnable = _FakeRunnable([[_text_chunk("hel"), _text_chunk("lo")]])
    llm = _make_llm(monkeypatch, runnable, tools=[])

    result = list(llm.stream([{"role": "user", "content": "hi"}], threading.Event()))

    assert result == ["hel", "lo"]
    assert len(runnable.stream_calls) == 1


def test_stream_executes_tool_call_then_streams_final_answer(monkeypatch):
    fake_tool = MagicMock()
    fake_tool.name = "get_current_time"
    fake_tool.invoke.return_value = "Monday, 2026-09-10 12:00 UTC"

    runnable = _FakeRunnable(
        [
            [_tool_call_chunk("get_current_time", {}, "call_1")],
            [_text_chunk("It's "), _text_chunk("Monday.")],
        ]
    )
    llm = _make_llm(monkeypatch, runnable, tools=[fake_tool])

    result = list(llm.stream([{"role": "user", "content": "what day is it"}], threading.Event()))

    assert result == ["It's ", "Monday."]
    assert len(runnable.stream_calls) == 2
    fake_tool.invoke.assert_called_once_with({})

    # second round's messages include the tool-call AI message + its ToolMessage result
    second_round_messages = runnable.stream_calls[1]
    tool_messages = [m for m in second_round_messages if m.__class__.__name__ == "ToolMessage"]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "Monday, 2026-09-10 12:00 UTC"
    assert tool_messages[0].tool_call_id == "call_1"


def test_stream_stops_on_cancel_mid_round(monkeypatch):
    runnable = _FakeRunnable([[_text_chunk("a"), _text_chunk("b"), _text_chunk("c")]])
    llm = _make_llm(monkeypatch, runnable, tools=[])
    cancel = threading.Event()

    pieces = []
    for i, piece in enumerate(llm.stream([{"role": "user", "content": "hi"}], cancel)):
        pieces.append(piece)
        if i == 0:
            cancel.set()

    assert pieces == ["a"]


def test_stream_bounds_tool_loop_at_max_tool_rounds(monkeypatch):
    def infinite_tool_call_round():
        return [_tool_call_chunk("get_current_time", {}, "call_x")]

    fake_tool = MagicMock()
    fake_tool.name = "get_current_time"
    fake_tool.invoke.return_value = "irrelevant"

    runnable = _FakeRunnable([infinite_tool_call_round() for _ in range(5)])
    llm = _make_llm(monkeypatch, runnable, tools=[fake_tool])
    llm.max_tool_rounds = 2

    result = list(llm.stream([{"role": "user", "content": "hi"}], threading.Event()))

    assert result == []
    assert len(runnable.stream_calls) == 2


def test_local_provider_builds_chat_openai_with_lm_studio_defaults(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured_kwargs = {}

    def fake_chat_openai(**kwargs):
        captured_kwargs.update(kwargs)
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = _FakeRunnable([[_text_chunk("hi")]])
        return mock_model

    monkeypatch.setattr("asr_test.llm.langchain_llm.ChatOpenAI", fake_chat_openai)

    llm = LangChainLlm(tools=[], warmup=False)

    assert llm.provider.name == "local"
    assert captured_kwargs["base_url"] == "http://localhost:1234/v1"
    assert captured_kwargs["api_key"] == "lm-studio"
    assert captured_kwargs["model"] == "lfm2.5-230m"
    assert captured_kwargs["stream_usage"] is True


def test_openai_provider_builds_chat_openai_without_base_url(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    captured_kwargs = {}

    def fake_chat_openai(**kwargs):
        captured_kwargs.update(kwargs)
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = _FakeRunnable([[_text_chunk("hi")]])
        return mock_model

    monkeypatch.setattr("asr_test.llm.langchain_llm.ChatOpenAI", fake_chat_openai)

    llm = LangChainLlm(tools=[], warmup=False)

    assert llm.provider.name == "openai"
    assert captured_kwargs["base_url"] is None
    assert captured_kwargs["api_key"] == "sk-test"
    assert captured_kwargs["model"] == "gpt-4o-mini"


def test_azure_provider_builds_azure_chat_openai(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "my-deployment")
    captured_kwargs = {}

    def fake_azure_chat_openai(**kwargs):
        captured_kwargs.update(kwargs)
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = _FakeRunnable([[_text_chunk("hi")]])
        return mock_model

    monkeypatch.setattr("asr_test.llm.langchain_llm.AzureChatOpenAI", fake_azure_chat_openai)

    llm = LangChainLlm(tools=[], warmup=False)

    assert llm.provider.name == "azure"
    assert captured_kwargs["azure_endpoint"] == "https://example.openai.azure.com/"
    assert captured_kwargs["azure_deployment"] == "my-deployment"
    assert captured_kwargs["api_key"] == "azure-key"
    assert captured_kwargs["api_version"] == "2026-01-01-preview"


def test_model_kwarg_overrides_env_derived_model(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    captured_kwargs = {}

    def fake_chat_openai(**kwargs):
        captured_kwargs.update(kwargs)
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = _FakeRunnable([[_text_chunk("hi")]])
        return mock_model

    monkeypatch.setattr("asr_test.llm.langchain_llm.ChatOpenAI", fake_chat_openai)

    llm = LangChainLlm(model="gpt-4o-mini", tools=[], warmup=False)

    assert llm.provider.model == "gpt-4o-mini"
    assert captured_kwargs["model"] == "gpt-4o-mini"


def test_warmup_failure_message_omits_lm_studio_hint_for_non_local(monkeypatch, capsys):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mock_model = MagicMock()
    mock_model.invoke.side_effect = RuntimeError("boom")
    mock_model.bind_tools.return_value = _FakeRunnable([])
    monkeypatch.setattr("asr_test.llm.langchain_llm.ChatOpenAI", lambda **kw: mock_model)

    LangChainLlm(tools=[], warmup=True, warmup_attempts=1)

    out = capsys.readouterr().out
    assert "llm warm-up failed" in out
    assert "LM Studio" not in out


def test_warmup_retries_on_failure_and_succeeds_before_attempts_exhausted(monkeypatch, capsys):
    monkeypatch.setattr("asr_test.llm.langchain_llm.time.sleep", lambda seconds: None)
    mock_model = MagicMock()
    mock_model.invoke.side_effect = [ConnectionError("not up yet"), MagicMock()]
    mock_model.bind_tools.return_value = _FakeRunnable([])
    monkeypatch.setattr("asr_test.llm.langchain_llm.ChatOpenAI", lambda **kw: mock_model)

    LangChainLlm(tools=[], warmup=True, warmup_attempts=3, warmup_backoff_base=1.0)

    out = capsys.readouterr().out
    assert mock_model.invoke.call_count == 2
    assert "llm warm-up:" in out
    assert "llm warm-up failed" not in out


def test_warmup_gives_up_and_logs_after_exhausting_attempts(monkeypatch, capsys):
    monkeypatch.setattr("asr_test.llm.langchain_llm.time.sleep", lambda seconds: None)
    mock_model = MagicMock()
    mock_model.invoke.side_effect = ConnectionError("still not up")
    mock_model.bind_tools.return_value = _FakeRunnable([])
    monkeypatch.setattr("asr_test.llm.langchain_llm.ChatOpenAI", lambda **kw: mock_model)

    LangChainLlm(tools=[], warmup=True, warmup_attempts=3, warmup_backoff_base=1.0)

    out = capsys.readouterr().out
    assert mock_model.invoke.call_count == 3
    assert "llm warm-up failed after 3 attempts" in out


def test_context_window_resolved_once_at_construction(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    mock_model = MagicMock()
    mock_model.bind_tools.return_value = _FakeRunnable([[_text_chunk("hi")]])
    monkeypatch.setattr("asr_test.llm.langchain_llm.ChatOpenAI", lambda **kw: mock_model)
    calls = []

    def fake_get_context_window(provider):
        calls.append(provider)
        return 131072

    monkeypatch.setattr("asr_test.llm.langchain_llm.get_context_window", fake_get_context_window)

    llm = LangChainLlm(tools=[], warmup=False)

    assert llm._context_window == 131072
    assert len(calls) == 1  # called once at construction, not per stream() call


def test_stream_populates_usage_dict_for_plain_text_reply(monkeypatch):
    monkeypatch.delenv("OPENAI_PRICE_INPUT_PER_1K", raising=False)
    monkeypatch.delenv("OPENAI_PRICE_OUTPUT_PER_1K", raising=False)
    runnable = _FakeRunnable([[_text_chunk("hi"), _usage_chunk(100, 20)]])
    llm = _make_llm(monkeypatch, runnable, tools=[])

    usage = {}
    result = list(llm.stream([{"role": "user", "content": "hi"}], threading.Event(), usage))

    assert result == ["hi"]
    assert usage["provider"] == "local"
    assert usage["model"] == "lfm2.5-230m"
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 20
    assert usage["total_tokens"] == 120
    assert usage["cost_usd"] == 0.0  # local is always free
    assert usage["tool_calls"] == []
    assert usage["context_window"] is None  # _make_llm's LangChainLlm has no LM Studio to query in tests


def test_stream_usage_none_by_default_does_not_error(monkeypatch):
    runnable = _FakeRunnable([[_text_chunk("hi"), _usage_chunk(10, 5)]])
    llm = _make_llm(monkeypatch, runnable, tools=[])

    result = list(llm.stream([{"role": "user", "content": "hi"}], threading.Event()))

    assert result == ["hi"]  # no exception with usage left as default None


def test_stream_usage_left_empty_when_backend_reports_no_metadata(monkeypatch):
    runnable = _FakeRunnable([[_text_chunk("hi")]])  # no usage_metadata attached
    llm = _make_llm(monkeypatch, runnable, tools=[])

    usage = {}
    list(llm.stream([{"role": "user", "content": "hi"}], threading.Event(), usage))

    assert usage == {}


def test_stream_usage_includes_tool_calls_made(monkeypatch):
    fake_tool = MagicMock()
    fake_tool.name = "get_current_time"
    fake_tool.invoke.return_value = "Monday, 2026-09-10 12:00 UTC"

    runnable = _FakeRunnable(
        [
            [_tool_call_chunk("get_current_time", {}, "call_1")],
            [_text_chunk("It's Monday."), _usage_chunk(50, 10)],
        ]
    )
    llm = _make_llm(monkeypatch, runnable, tools=[fake_tool])

    usage = {}
    result = list(llm.stream([{"role": "user", "content": "what day is it"}], threading.Event(), usage))

    assert result == ["It's Monday."]
    assert usage["tool_calls"] == [{"name": "get_current_time", "args": {}}]
    assert usage["input_tokens"] == 50
    assert usage["output_tokens"] == 10


def test_stream_usage_includes_resolved_context_window(monkeypatch):
    runnable = _FakeRunnable([[_text_chunk("hi"), _usage_chunk(10, 5)]])
    llm = _make_llm(monkeypatch, runnable, tools=[])
    llm._context_window = 131072  # simulate what Task 4's __init__ would have resolved

    usage = {}
    list(llm.stream([{"role": "user", "content": "hi"}], threading.Event(), usage))

    assert usage["context_window"] == 131072


def test_stream_retries_context_window_lookup_on_next_turn_when_still_none(monkeypatch):
    # Simulates: LM Studio wasn't reachable yet when LangChainLlm was
    # constructed (context_window stays None), then comes up before the
    # next turn -- the lookup should self-heal without recreating the LLM.
    runnable = _FakeRunnable([
        [_text_chunk("hi"), _usage_chunk(10, 5)],
        [_text_chunk("hi"), _usage_chunk(10, 5)],
    ])
    llm = _make_llm(monkeypatch, runnable, tools=[])
    assert llm._context_window is None  # autouse fixture makes construction return None

    calls = []

    def fake_get_context_window(provider):
        calls.append(provider)
        return 131072

    monkeypatch.setattr("asr_test.llm.langchain_llm.get_context_window", fake_get_context_window)

    usage1 = {}
    list(llm.stream([{"role": "user", "content": "hi"}], threading.Event(), usage1))
    assert usage1["context_window"] == 131072
    assert len(calls) == 1

    usage2 = {}
    list(llm.stream([{"role": "user", "content": "hi again"}], threading.Event(), usage2))
    assert usage2["context_window"] == 131072
    assert len(calls) == 1  # not retried again once resolved


def test_generate_title_returns_the_models_response(monkeypatch):
    runnable = _FakeRunnable([])
    llm = _make_llm(monkeypatch, runnable, tools=[])
    llm._model.invoke.return_value = MagicMock(content="Weekend trip planning")

    title = llm.generate_title("where should I go this weekend", "Try the coast, it's lovely this time of year.")

    assert title == "Weekend trip planning"


def test_generate_title_strips_surrounding_quotes(monkeypatch):
    runnable = _FakeRunnable([])
    llm = _make_llm(monkeypatch, runnable, tools=[])
    llm._model.invoke.return_value = MagicMock(content='"Weekend trip planning"')

    title = llm.generate_title("where should I go this weekend", "Try the coast.")

    assert title == "Weekend trip planning"


def test_generate_title_returns_none_on_failure(monkeypatch):
    runnable = _FakeRunnable([])
    llm = _make_llm(monkeypatch, runnable, tools=[])
    llm._model.invoke.side_effect = ConnectionError("boom")

    assert llm.generate_title("hi", "hello") is None


def test_generate_title_returns_none_for_empty_response(monkeypatch):
    runnable = _FakeRunnable([])
    llm = _make_llm(monkeypatch, runnable, tools=[])
    llm._model.invoke.return_value = MagicMock(content="   ")

    assert llm.generate_title("hi", "hello") is None

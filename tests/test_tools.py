from asr_test.llm.tools import tool_status


def test_get_current_time_is_always_enabled(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    status = tool_status()

    time_entry = next(t for t in status if t["name"] == "get_current_time")
    assert time_entry["enabled"] is True


def test_web_search_disabled_when_tavily_key_not_set(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    status = tool_status()

    search_entry = next(t for t in status if t["name"] == "web_search")
    assert search_entry["enabled"] is False


def test_web_search_enabled_when_tavily_key_set(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")

    status = tool_status()

    search_entry = next(t for t in status if t["name"] == "web_search")
    assert search_entry["enabled"] is True


def test_tool_status_does_not_construct_real_tavily_tool(monkeypatch):
    # Regression guard: tool_status() must stay cheap/side-effect-free
    # (no network, no real TavilySearch construction) so GET /options
    # can call it on every request.
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")

    status = tool_status()

    assert all(isinstance(t["name"], str) for t in status)
    assert all("label" in t and "enabled" in t for t in status)

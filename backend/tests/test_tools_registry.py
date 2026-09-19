import os
import sys

import pytest

from pos.tools import all_tools, build_tools, credential_fields, get_tool, resolve_enabled, tool_status
from pos.tools.base import ToolSpec
from pos.tools.registry import register


def test_both_builtin_tools_are_registered_with_their_own_specs():
    ids = [s.id for s in all_tools()]
    assert "get_current_time" in ids and "web_search" in ids
    assert get_tool("web_search").requires_key is True
    assert get_tool("get_current_time").requires_key is False


def test_credential_fields_come_from_the_registry_not_a_hardcoded_table():
    assert credential_fields()["tavily"] == {"tavily_api_key": "TAVILY_API_KEY"}


def test_a_tool_needing_a_key_is_only_available_when_the_key_is_present():
    spec = get_tool("web_search")
    assert spec.available({}) is False
    assert spec.available({"TAVILY_API_KEY": "k"}) is True


def test_resolve_enabled_honours_user_switch_default_and_availability():
    env = {"TAVILY_API_KEY": "k"}
    assert resolve_enabled({}, env) == ["get_current_time", "web_search"]          # defaults
    assert resolve_enabled({"web_search": False}, env) == ["get_current_time"]     # user switched off
    assert resolve_enabled({"get_current_time": False}, {}) == []                   # web_search lacks a key


def test_build_tools_skips_unknown_and_unavailable_ids_without_raising():
    tools = build_tools(["get_current_time", "web_search", "nope"], env={})
    assert [t.name for t in tools] == ["get_current_time"]


def test_build_tools_none_means_every_default_tool_that_is_available():
    assert [t.name for t in build_tools(None, env={})] == ["get_current_time"]
    assert [t.name for t in build_tools(None, env={"TAVILY_API_KEY": "k"})] == ["get_current_time", "tavily_search"]


def test_build_tools_passes_the_key_explicitly_and_never_touches_os_environ(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    tools = build_tools(["web_search"], env={"TAVILY_API_KEY": "tvly-override"})
    assert [t.name for t in tools] == ["tavily_search"]
    assert "TAVILY_API_KEY" not in os.environ


def test_tool_status_never_constructs_a_tool(monkeypatch):
    # If status constructed Tavily it would import langchain_tavily; make that impossible.
    monkeypatch.setitem(sys.modules, "langchain_tavily", None)

    status = tool_status({"TAVILY_API_KEY": "k"}, {})

    assert {s["name"]: s["enabled"] for s in status} == {"get_current_time": True, "web_search": True}


def test_tool_status_defaults_env_to_os_environ(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert {s["name"]: s["enabled"] for s in tool_status()}["web_search"] is False


def test_registering_a_duplicate_id_is_rejected():
    dup = ToolSpec(id="get_current_time", label="x", description="x", build=lambda env: None)
    with pytest.raises(ValueError, match="duplicate"):
        register(dup)

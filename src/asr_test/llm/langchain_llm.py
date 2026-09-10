from __future__ import annotations

import threading
import time
from collections.abc import Iterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from ..interfaces.llm import LlmBase
from .context_window import get_context_window
from .provider import resolve_provider
from .tools import default_tools


class LangChainLlm(LlmBase):
    """Same LlmBase contract as OpenAiCompatibleLlm, plus tool calling
    via LangChain's bind_tools() and multi-backend support (local LM
    Studio / OpenAI / Azure OpenAI, picked by resolve_provider() from
    env vars — see docs/superpowers/specs/2026-09-10-llm-provider-cost-tracking-design.md).
    The tool-execution loop (invoke, check tool_calls, run tool, append
    ToolMessage, invoke again) is hand-rolled per LangChain's own "Tool
    Execution Loop" pattern rather than using create_agent — that owns
    its own conversation memory (AgentState + checkpointer), which
    would duplicate Agent's self.conversation."""

    def __init__(
        self,
        model: str | None = None,
        system_prompt: str = (
            "You are a concise voice assistant. Answer in one or two short sentences. "
            "Plain text only — no markdown, lists, or emoji. Your words are spoken aloud. "
            "Use a tool when you need current information (today's date/time, or facts "
            "you're not sure of) instead of guessing."
        ),
        max_tokens: int = 120,
        timeout: float = 30,
        tools: list[BaseTool] | None = None,
        max_tool_rounds: int = 3,   # bounded — avoid an infinite tool-call loop
        warmup: bool = True,
    ):
        self.provider = resolve_provider(model_override=model)
        self._context_window = get_context_window(self.provider)
        self.system_prompt = system_prompt
        self.max_tool_rounds = max_tool_rounds
        self.tools = default_tools() if tools is None else tools
        self._tools_by_name = {t.name: t for t in self.tools}

        self._model = self._build_model(max_tokens=max_tokens, timeout=timeout)
        self._runnable = self._model.bind_tools(self.tools) if self.tools else self._model

        if warmup:
            t0 = time.perf_counter()
            try:
                self._model.invoke([HumanMessage("hi")], max_tokens=1)
                print(f"  llm warm-up: {time.perf_counter() - t0:.2f}s")
            except Exception as e:
                hint = " — is LM Studio running?" if self.provider.name == "local" else ""
                print(f"  llm warm-up failed ({e}){hint}")

    def _build_model(self, max_tokens: int, timeout: float):
        common = dict(max_tokens=max_tokens, temperature=0.7, timeout=timeout, stream_usage=True)
        if self.provider.name == "azure":
            return AzureChatOpenAI(
                azure_endpoint=self.provider.azure_endpoint,
                azure_deployment=self.provider.azure_deployment,
                api_version=self.provider.api_version,
                api_key=self.provider.api_key,
                **common,
            )
        return ChatOpenAI(
            base_url=self.provider.base_url,
            api_key=self.provider.api_key,
            model=self.provider.model,
            **common,
        )

    def stream(self, messages: list[dict], cancel: threading.Event) -> Iterator[str]:
        full: list[BaseMessage] = [SystemMessage(self.system_prompt)]
        for m in messages:
            full.append(
                HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
            )

        for _ in range(self.max_tool_rounds):
            if cancel.is_set():
                return

            accumulated = None
            for chunk in self._runnable.stream(full):
                if cancel.is_set():
                    return
                accumulated = chunk if accumulated is None else accumulated + chunk
                if chunk.content:
                    yield chunk.content

            if accumulated is None or not accumulated.tool_calls:
                return

            full.append(accumulated)
            for call in accumulated.tool_calls:
                if cancel.is_set():
                    return
                tool_ = self._tools_by_name.get(call["name"])
                result = tool_.invoke(call["args"]) if tool_ is not None else f"unknown tool: {call['name']}"
                full.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from ..interfaces.llm import LlmBase
from .context_window import get_context_window
from .pricing import estimate_cost
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
        warmup_attempts: int = 3,        # bounded backoff, not a long block — see _warmup()
        warmup_backoff_base: float = 1.0,
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
            self._warmup(warmup_attempts, warmup_backoff_base)

    def _warmup(self, attempts: int, backoff_base: float) -> None:
        """A few quick retries (short bounded backoff, not a long block —
        this runs at server startup and must not stall it for long)
        rather than one attempt, since the common failure mode observed
        in practice is a race: the server starts before LM Studio's own
        local server has finished coming up, not LM Studio being
        genuinely absent. If every attempt fails, context_window still
        gets a lazy per-turn retry later (see _fill_usage) as a further
        safety net for anything slower than this covers."""
        last_error: Exception | None = None
        for attempt in range(attempts):
            t0 = time.perf_counter()
            try:
                self._model.invoke([HumanMessage("hi")], max_tokens=1)
                retried = f" (attempt {attempt + 1}/{attempts})" if attempt else ""
                print(f"  llm warm-up: {time.perf_counter() - t0:.2f}s{retried}")
                if self._context_window is None:
                    self._context_window = get_context_window(self.provider)
                return
            except Exception as e:
                last_error = e
                if attempt < attempts - 1:
                    time.sleep(backoff_base * (2 ** attempt))
        hint = " — is LM Studio running?" if self.provider.name == "local" else ""
        print(f"  llm warm-up failed after {attempts} attempts ({last_error}){hint}")

    def generate_title(self, first_user_message: str, first_bot_message: str) -> str | None:
        """A short (3-5 word) title summarizing a conversation's topic,
        the way ChatGPT names each chat after its first exchange. Calls
        self._model directly (not self._runnable/the tool loop, and not
        self.system_prompt's voice-assistant persona) -- this is a
        one-off, non-streaming, unrelated request, not a conversation
        turn. Returns None on any failure (never raises) so a caller
        can skip storing a title without special-casing this method."""
        try:
            response = self._model.invoke([
                SystemMessage(
                    "Summarize the topic of this exchange in 3 to 5 words, as a short "
                    "chat title. Sentence case, no punctuation, no quotes, no emoji."
                ),
                HumanMessage(f"User: {first_user_message}\nAssistant: {first_bot_message}"),
            ], max_tokens=16)
            title = (response.content or "").strip().strip('"').strip("'")
            return title or None
        except Exception:
            return None

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

    def stream(
        self, messages: list[dict], cancel: threading.Event, usage: dict | None = None
    ) -> Iterator[str]:
        full: list[BaseMessage] = [SystemMessage(self.system_prompt)]
        for m in messages:
            full.append(
                HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
            )

        tool_calls_made: list[dict] = []

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
                if usage is not None and accumulated is not None:
                    self._fill_usage(usage, accumulated, tool_calls_made)
                return

            full.append(accumulated)
            for call in accumulated.tool_calls:
                if cancel.is_set():
                    return
                tool_calls_made.append({"name": call["name"], "args": call["args"]})
                tool_ = self._tools_by_name.get(call["name"])
                result = tool_.invoke(call["args"]) if tool_ is not None else f"unknown tool: {call['name']}"
                full.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    def _fill_usage(self, usage: dict, accumulated, tool_calls_made: list[dict]) -> None:
        meta = getattr(accumulated, "usage_metadata", None)
        if not meta:
            return  # backend didn't report it — leave usage as {}, not zeroed
        if self._context_window is None:
            # Retried per-turn (not just once at __init__) so a transient
            # failure self-heals -- e.g. the LLM was constructed at server
            # startup before LM Studio itself had finished starting, which
            # would otherwise leave context_window permanently None for the
            # life of this (cached, shared) instance.
            self._context_window = get_context_window(self.provider)
        input_tokens = meta.get("input_tokens")
        output_tokens = meta.get("output_tokens")
        cost = None
        if input_tokens is not None and output_tokens is not None:
            cost = estimate_cost(self.provider.name, self.provider.model, input_tokens, output_tokens)
        usage.update({
            "provider": self.provider.name,
            "model": self.provider.model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": meta.get("total_tokens"),
            "cost_usd": cost,
            "tool_calls": tool_calls_made,
            "context_window": self._context_window,
        })

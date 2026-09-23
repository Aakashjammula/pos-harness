"""Builds the trace payload for one turn: usage, cost, model, and tool
calls, in the shape sent as the `done` SSE event and stored per turn.

Verified against a real tool-calling turn (see the backend design
discussion): a turn can span multiple AIMessages (one per tool-call
round), each carrying its own `usage_metadata` -- so cost is summed
across every new message in the turn, not read off just the last one.

Per-tool cost follows the OpenTelemetry GenAI convention: tokens belong
to model calls, never to tools. A tool has no token count of its own --
what it costs you is the growth it causes in the *next* round's input.
So a tool's cost here is a delta between two provider-reported numbers,
not something re-counted client-side.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.messages import AIMessage, ToolMessage

from pos import config, pricing

logger = logging.getLogger(__name__)

_counter_model: Any | None = None


def _count_tokens(message: ToolMessage) -> int:
    """Counts one tool result's tokens, for splitting a shared delta.

    Uses the model's own encoding via `get_num_tokens_from_messages`,
    which runs locally (no API call) and accounts for the role and
    tool_call_id overhead that a bare tokenizer would miss. Falls back to
    a character estimate if the model can't be built -- this only ever
    decides the *ratio* within a round, never a total.

    Args:
        message: The tool result to measure.

    Returns:
        A token count, at least 1 so a round of empty results still
        splits evenly rather than dividing by zero.
    """
    global _counter_model
    try:
        if _counter_model is None:
            from langchain.chat_models import init_chat_model

            _counter_model = init_chat_model(f"azure_openai:{config.MODEL_NAME}")
        return max(1, _counter_model.get_num_tokens_from_messages([message]))
    except Exception:  # noqa: BLE001 -- counting must never break a reply
        logger.debug("token counter unavailable; falling back to a character estimate")
        return max(1, len(str(message.content)) // 4)


def _split_delta(delta: int, results: list[ToolMessage | None]) -> list[int]:
    """Divides one round's growth between the tools that caused it.

    Args:
        delta: The real, provider-reported growth to divide up.
        results: Each tool call's result, in order. A missing result is
            weighted as 1.

    Returns:
        One count per tool, summing to exactly `delta` -- the remainder
        goes to the last tool so the parts always tie back to the total.
    """
    if len(results) == 1:
        return [delta]
    weights = [_count_tokens(r) if r is not None else 1 for r in results]
    total = sum(weights)
    shares = [delta * w // total for w in weights]
    shares[-1] += delta - sum(shares)
    return shares


def build_trace(new_messages: list[Any], reasoning_effort: str, model_spec: str | None = None) -> dict:
    """Builds the trace payload for the messages a single turn produced.

    Args:
        new_messages: The messages this turn added -- i.e. the tail of
            `stream.output["messages"]` beyond however many messages the
            thread had before this call. Includes every tool-call round,
            not just the final answer.
        reasoning_effort: The `reasoning_effort` value this turn was
            called with.
        model_spec: The model as configured, e.g. "openai:gpt-5". Prices
            differ between providers, so the rates come from this rather
            than from the model id the reply reported.

    Returns:
        A dict with `model`, `finish_reason`, `reasoning_effort`, `usage`
        (token counts, cache counts, `cost_usd`, and a per-round
        breakdown), and `tool_calls` (each tagged with its round and what
        it cost).
    """
    ai_messages = [m for m in new_messages if isinstance(m, AIMessage)]
    tool_messages = {m.tool_call_id: m for m in new_messages if isinstance(m, ToolMessage)}

    input_tokens = output_tokens = total_tokens = 0
    cache_read = cache_creation = reasoning_tokens = 0
    rounds: list[dict] = []
    for index, msg in enumerate(ai_messages):
        usage = msg.usage_metadata or {}
        details = usage.get("input_token_details") or {}
        out_details = usage.get("output_token_details") or {}

        round_input = usage.get("input_tokens") or 0
        input_tokens += round_input
        output_tokens += usage.get("output_tokens") or 0
        total_tokens += usage.get("total_tokens") or 0
        cache_read += details.get("cache_read") or 0
        cache_creation += details.get("cache_creation") or 0
        # Reasoning tokens are a *breakdown of* output_tokens, not an extra
        # charge on top: the model thinks, we're billed for it at the output
        # rate, and the thinking itself is stripped from the reply. So this
        # is recorded for visibility only -- cost_usd below must not add it.
        reasoning_tokens += out_details.get("reasoning") or 0

        # How much bigger this round's prompt was than the last one's. On the
        # first round there is nothing to compare against: its input is the
        # thread as it already stood, not growth.
        previous = rounds[-1]["input_tokens"] if rounds else None
        rounds.append({
            "index": index,
            "input_tokens": round_input,
            "output_tokens": usage.get("output_tokens") or 0,
            "reasoning_tokens": out_details.get("reasoning") or 0,
            "cache_read": details.get("cache_read") or 0,
            "cache_creation": details.get("cache_creation") or 0,
            "delta": None if previous is None else round_input - previous,
            "tool_calls": len(msg.tool_calls),
            # What the model said in this round, before its tool calls ran.
            # The transcript keeps one blob of text per turn, so this is the
            # only place the commentary between tool batches survives.
            "text": msg.text or "",
        })

    final = ai_messages[-1] if ai_messages else None
    model = final.response_metadata.get("model_name") if final else None
    finish_reason = final.response_metadata.get("finish_reason") if final else None

    # `input_tokens` above is the sum over every round, which is what the turn
    # cost -- but it is not how big the conversation is. Each round resends the
    # whole thread, so the *last* round's input is the context as it now
    # stands. A turn with 11 rounds bills ~121k while the thread is ~17k.
    context_tokens = (final.usage_metadata or {}).get("input_tokens") or 0 if final else 0

    # Prices are per provider, so they follow the configured spec, not the
    # model id the reply reported -- the same model costs differently on
    # Azure and on OpenAI.
    spec = model_spec or config.MODEL_NAME

    # A round's tool calls are paid for by the *next* round, which is the one
    # that has to carry their results. One tool in a round gets the whole
    # delta exactly; several share it, split by result size.
    tool_calls = []
    for index, msg in enumerate(ai_messages):
        if not msg.tool_calls:
            continue
        results = [tool_messages.get(call["id"]) for call in msg.tool_calls]
        next_delta = rounds[index + 1]["delta"] if index + 1 < len(rounds) else None
        if next_delta is None:
            costs: list[int | None] = [None] * len(msg.tool_calls)
        else:
            costs = list(_split_delta(next_delta, results))
        for call, result_msg, cost in zip(msg.tool_calls, results, costs, strict=True):
            tool_calls.append({
                "name": call["name"],
                "args": call["args"],
                "result": result_msg.content if result_msg else None,
                "round": index,
                "cost_tokens": cost,
                # True when the round made several calls, so this number is a
                # share of a real delta rather than the delta itself.
                "cost_estimated": len(msg.tool_calls) > 1,
            })

    return {
        "model": model,
        "finish_reason": finish_reason,
        "reasoning_effort": reasoning_effort,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cache_read": cache_read,
            "cache_creation": cache_creation,
            "reasoning_tokens": reasoning_tokens,
            "context_tokens": context_tokens,
            "cost_usd": pricing.cost_usd(
                spec, input_tokens, output_tokens, cache_read, cache_creation
            ),
            # The model's own limit, to compare `context_tokens` against.
            # `model` here is the versioned id the reply reported; the
            # catalogue lookup strips the date to find it.
            "context_window": pricing.plan_for(spec, model).context_window,
            "rounds": rounds,
        },
        "tool_calls": tool_calls,
    }

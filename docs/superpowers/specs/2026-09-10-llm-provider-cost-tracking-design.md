# LLM provider selection + usage/cost tracking

Date: 2026-09-10
Status: draft, awaiting review

## Context

This is sub-project A of a four-part plan (the other three: session
persistence, a text-only chat mode, and a token/trace UI panel — each
gets its own spec once this one lands). `LangChainLlm`
(`src/asr_test/llm/langchain_llm.py`) currently talks to exactly one
backend: a local LM Studio server via `ChatOpenAI(base_url="http://localhost:1234/v1", ...)`,
with `warmup=True` assuming that's always reachable. It exposes no
information about how many tokens a turn used or what it cost — the
`LlmBase.stream()` contract yields plain text pieces and nothing else,
by design (see its docstring: implementations are stateless, so the
existing `OpenAiCompatibleLlm` deliberately owns no `last_ttft`/`last_total`
instance state, because one instance is shared across concurrent server
sessions per `server.py`'s provider cache).

This spec adds: (1) three selectable LLM backends — local LM Studio,
OpenAI's real API, and Azure OpenAI — chosen once per server/CLI run via
environment variables, and (2) per-turn token usage + a cost estimate,
threaded back to the caller without breaking the existing
shared-instance statelessness guarantee. The later trace-UI sub-project
will *display* this data; this spec only makes it exist and reach
`Agent`/the console/the websocket payload.

## Decisions from brainstorming

- **Backend selection is env-var only**, decided once at process
  startup — no per-session UI picker. Precedence: `AZURE_OPENAI_API_KEY`
  set → Azure; else `OPENAI_API_KEY` set → OpenAI; else → local (today's
  behavior, unchanged default).
- **One `LangChainLlm` class**, not three subclasses — the tool-calling
  loop in `stream()` is backend-agnostic; only model construction and
  pricing lookup differ per backend. Matches this codebase's existing
  bias toward one implementation per concern (`SileroVad`, `OnnxAsrEngine`)
  rather than a class hierarchy.
- **Cost is `$0` for local**, always. For OpenAI/Azure, a small built-in
  price table covers a few common OpenAI models as a best-effort
  default (explicitly commented as something that will drift — LLM
  pricing changes over time and isn't something to hardcode confidently);
  env vars override it. Azure pricing is contract/region-specific, so it
  has **no built-in default** — cost is `null`/"n/a" until the user sets
  its price env vars themselves.
- **Usage data flows out via a caller-owned mutable dict**, the same
  shape as the existing `cancel: threading.Event` parameter — not
  instance state — so sharing one `LangChainLlm` across concurrent
  sessions stays race-free.
- Scope: this spec does not build any UI. It ends at "the data exists
  and reaches `Agent`'s console output and the `bot_text` websocket
  event." The dedicated trace-UI sub-project (D) consumes it.

## Backend selection (`src/asr_test/llm/provider.py`)

New module, one function:

```python
@dataclass
class ProviderConfig:
    name: str                 # "local" | "openai" | "azure"
    model: str
    # local/openai (ChatOpenAI):
    base_url: str | None = None
    api_key: str | None = None
    # azure (AzureChatOpenAI):
    azure_endpoint: str | None = None
    azure_deployment: str | None = None
    api_version: str | None = None

def resolve_provider(model_override: str | None = None) -> ProviderConfig: ...
```

`resolve_provider()` reads `os.environ` directly (no config-file layer —
matches how `TAVILY_API_KEY` already works for the search tool). Env
vars:

| Var | Backend | Required | Notes |
|---|---|---|---|
| `AZURE_OPENAI_API_KEY` | azure | yes (selects this backend) | |
| `AZURE_OPENAI_ENDPOINT` | azure | yes | e.g. `https://<resource>.openai.azure.com/` |
| `AZURE_OPENAI_DEPLOYMENT` | azure | yes | deployment name, used as "model" everywhere |
| `AZURE_OPENAI_API_VERSION` | azure | no | default `"2026-01-01-preview"` |
| `OPENAI_API_KEY` | openai | yes (selects this backend) | |
| `OPENAI_MODEL` | openai | no | default `"gpt-4o-mini"` |

`model_override` (from `LangChainLlm(model=...)`, itself driven by the
existing UI model dropdown / `--llm-model` CLI flag in local mode) wins
over `OPENAI_MODEL`/`AZURE_OPENAI_DEPLOYMENT` when given — matches how
`model` already works today. Missing-required-var errors raise
`RuntimeError` with a message naming exactly which var is missing —
fail fast at startup, not on the first turn.

`LangChainLlm._build_model(provider: ProviderConfig)` constructs
`ChatOpenAI` (local/openai) or `AzureChatOpenAI` (azure) accordingly,
and skips the constructor's `warmup` probe's specific error message
tuning for "is LM Studio running?" when the backend isn't local (still
warms up, just doesn't imply the wrong troubleshooting step).

## Usage + cost (`src/asr_test/llm/pricing.py` + `langchain_llm.py`)

```python
PRICING: dict[tuple[str, str], tuple[float, float]] = {
    # (provider, model) -> (input $/1K tokens, output $/1K tokens)
    # Best-effort snapshot, not guaranteed current — override via
    # OPENAI_PRICE_INPUT_PER_1K / OPENAI_PRICE_OUTPUT_PER_1K if these
    # have changed (they will, eventually).
    ("openai", "gpt-4o-mini"): (0.15, 0.60),
    ("openai", "gpt-4o"): (2.50, 10.00),
}

def price_for(provider: str, model: str) -> tuple[float, float] | None: ...
def estimate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float | None: ...
```

`price_for()` checks env overrides first
(`{PROVIDER}_PRICE_INPUT_PER_1K`/`..._OUTPUT_PER_1K`, uppercased
provider name) before falling back to `PRICING`. Returns `None` (→ cost
`null` downstream) if neither an override nor a table entry exists.
Local always short-circuits to `(0.0, 0.0)` without consulting either.

`LlmBase.stream()`'s signature (`src/asr_test/interfaces/llm.py`) gains
one optional parameter:

```python
def stream(self, messages: list[dict], cancel: threading.Event, usage: dict | None = None) -> Iterator[str]: ...
```

Existing implementations (`OpenAiCompatibleLlm`, `FakeLlm`) get the same
`usage: dict | None = None` parameter added to their own `stream()`
signatures (for interface conformance with the updated `LlmBase`) but
never read or write it — behavior is unchanged and no existing test
breaks. `LangChainLlm.stream()`,
when `usage` is not `None`, populates it in place once the final round
of its tool-loop finishes (see langchain_llm.py's existing loop —
LangChain's `ChatOpenAI`/`AzureChatOpenAI` attach `usage_metadata`
— `{"input_tokens", "output_tokens", "total_tokens"}` — to the
accumulated `AIMessageChunk` when `stream_usage=True` is passed at
construction, which this change adds):

```python
usage.update({
    "provider": self.provider.name,
    "model": self.provider.model,
    "input_tokens": ...,
    "output_tokens": ...,
    "total_tokens": ...,
    "cost_usd": estimate_cost(...),   # None if unpriced
    "tool_calls": [{"name": ..., "args": ...} for call in ...],  # every tool call across all rounds this turn
})
```

If the model never returns `usage_metadata` (some backends omit it),
`usage` is left as `{}` — callers must treat a missing key as "unknown,"
not assume zero.

## Wiring into `Agent`

`Agent.respond()` (`src/asr_test/agent.py`, around the existing
`self.llm.stream(messages, self.cancel)` call) changes to:

```python
usage: dict = {}
for piece in self.llm.stream(messages, self.cancel, usage):
    ...  # unchanged
```

After the stream completes: the existing per-turn timing print (local
mode's `report()`-style output) gains one more line when `usage` is
non-empty, e.g.:

```
  tokens: 812 in / 47 out (859 total)   cost: $0.0002   tools: get_current_time
```

For server mode, `usage` (if non-empty) is merged into the existing
`bot_text` event's payload: `emit("bot_text", {"text": joined, "usage": usage})`.
No new websocket event type — this stays additive to what already
exists, so the current browser UI (which ignores unknown JSON fields)
keeps working unmodified. The dedicated trace-UI sub-project reads this
same field.

## Error handling

- Missing required env var for the selected backend → `RuntimeError` at
  `LangChainLlm.__init__` (i.e. at server/CLI startup, or at first
  websocket connection needing that model in server mode's lazy
  `get_llm()` cache) — not silently falling back to local, since that
  would silently send data to the wrong place if someone thought they'd
  configured Azure.
- Warmup failure (backend unreachable) behaves as today: caught,
  logged, doesn't stop startup.
- `estimate_cost()`/`price_for()` never raise — an unpriced model is
  `None`/"n/a", not an error.

## Testing

- `tests/test_provider.py` (new): `resolve_provider()` precedence
  (azure beats openai beats local), required-var errors, `model_override`
  winning over env-derived model name.
- `tests/test_pricing.py` (new): table lookup, env override precedence,
  `None` for unknown model, local always `(0.0, 0.0)`.
- `tests/test_langchain_llm.py` (extend): `usage` dict populated from a
  fake accumulated message's `usage_metadata`; `usage=None` (default)
  behaves exactly as today (no regression on the four existing tests);
  tool calls across multiple rounds all appear in `usage["tool_calls"]`.
- No test talks to a real OpenAI/Azure endpoint — `ChatOpenAI`/`AzureChatOpenAI`
  construction is monkeypatched exactly like today's `ChatOpenAI` mock.

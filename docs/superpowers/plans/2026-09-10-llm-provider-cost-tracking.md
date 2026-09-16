# LLM Provider Selection + Usage/Cost Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `LangChainLlm` run against local LM Studio, OpenAI, or Azure OpenAI (chosen via env vars at startup), and surface per-turn token usage, context-window size, and a cost estimate to `Agent`'s console output and the `bot_text` websocket event, without breaking the existing shared-instance statelessness guarantee.

**Architecture:** A new `resolve_provider()` picks a backend from env vars and returns a small `ProviderConfig`; `LangChainLlm` uses it to build either `ChatOpenAI` or `AzureChatOpenAI`. Usage data flows out of `stream()` via a caller-owned mutable dict (same pattern as the existing `cancel: threading.Event` parameter) instead of instance state, so one `LangChainLlm` can still be shared safely across concurrent server sessions. `Agent` passes that dict in and does something with it once the turn completes.

**Tech Stack:** Python 3.14, `langchain-openai` (`ChatOpenAI`, `AzureChatOpenAI`), pytest + `unittest.mock`.

**Spec:** `docs/superpowers/specs/2026-09-10-llm-provider-cost-tracking-design.md`

## Global Constraints

- Backend selection is env-var only, resolved once at construction — no per-session picker.
- Precedence: `AZURE_OPENAI_API_KEY` set → azure; else `OPENAI_API_KEY` set → openai; else → local (default, matches today's behavior).
- One `LangChainLlm` class handles all three backends — no per-backend subclasses.
- Cost is always `$0.0` for local. OpenAI/Azure cost comes from `PRICING` table entries or `{PROVIDER}_PRICE_INPUT_PER_1K`/`..._OUTPUT_PER_1K` env overrides; `None` ("n/a") if neither exists.
- Context window: local resolves it live from LM Studio's REST API v0 (`GET {host}/api/v0/models`, a different base path than `/v1/models`); OpenAI/Azure have no such API at all (confirmed via research) and use a small table + `{PROVIDER}_CONTEXT_WINDOW` env override instead, same pattern as pricing. Resolved once at `LangChainLlm.__init__`, never per-turn.
- `LlmBase.stream()` gains one new optional parameter: `usage: dict | None = None`. Existing implementations accept it for interface conformance but ignore it — no behavior change, no existing test may break.
- Missing required env var for the selected backend raises `RuntimeError` naming the missing var — fail fast at construction, never silently fall back to local.
- The full existing suite (`uv run pytest tests/ -q`, 48 tests as of this plan) must stay green after every task.

---

### Task 1: `ProviderConfig` + `resolve_provider()`

**Files:**
- Create: `src/pos/llm/provider.py`
- Test: `tests/test_provider.py`

**Interfaces:**
- Produces: `ProviderConfig` dataclass with fields `name: str`, `model: str`, `base_url: str | None = None`, `api_key: str | None = None`, `azure_endpoint: str | None = None`, `azure_deployment: str | None = None`, `api_version: str | None = None`.
- Produces: `resolve_provider(model_override: str | None = None) -> ProviderConfig`.
- Consumes: nothing from other tasks (this is the first task).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_provider.py`:

```python
import pytest

from pos.llm.provider import resolve_provider


def test_defaults_to_local_when_no_env_vars_set(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    provider = resolve_provider()

    assert provider.name == "local"
    assert provider.base_url == "http://localhost:1234/v1"
    assert provider.api_key == "lm-studio"
    assert provider.model == "lfm2.5-230m"


def test_model_override_wins_for_local(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    provider = resolve_provider(model_override="custom-model")

    assert provider.name == "local"
    assert provider.model == "custom-model"


def test_openai_backend_selected_when_key_set(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    provider = resolve_provider()

    assert provider.name == "openai"
    assert provider.api_key == "sk-test"
    assert provider.model == "gpt-4o-mini"
    assert provider.base_url is None


def test_openai_model_env_var_used_when_no_override(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")

    provider = resolve_provider()

    assert provider.model == "gpt-4o"


def test_model_override_wins_over_openai_model_env_var(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")

    provider = resolve_provider(model_override="gpt-4o-mini")

    assert provider.model == "gpt-4o-mini"


def test_azure_backend_takes_precedence_over_openai(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "my-deployment")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")  # both set — azure must win

    provider = resolve_provider()

    assert provider.name == "azure"
    assert provider.api_key == "azure-key"
    assert provider.azure_endpoint == "https://example.openai.azure.com/"
    assert provider.azure_deployment == "my-deployment"
    assert provider.model == "my-deployment"
    assert provider.api_version == "2026-01-01-preview"


def test_azure_api_version_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "my-deployment")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-05-01")

    provider = resolve_provider()

    assert provider.api_version == "2025-05-01"


def test_azure_model_override_wins_over_deployment_env_var(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "my-deployment")

    provider = resolve_provider(model_override="other-deployment")

    assert provider.model == "other-deployment"
    assert provider.azure_deployment == "other-deployment"


def test_azure_missing_endpoint_raises(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "my-deployment")

    with pytest.raises(RuntimeError, match="AZURE_OPENAI_ENDPOINT"):
        resolve_provider()


def test_azure_missing_deployment_raises_when_no_override(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)

    with pytest.raises(RuntimeError, match="AZURE_OPENAI_DEPLOYMENT"):
        resolve_provider()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_provider.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pos.llm.provider'`

- [ ] **Step 3: Write the implementation**

Create `src/pos/llm/provider.py`:

```python
"""Resolves which LLM backend to talk to from environment variables —
env-var-only, decided once per process, no per-session picker (see
docs/superpowers/specs/2026-09-10-llm-provider-cost-tracking-design.md)."""

from __future__ import annotations

import os
from dataclasses import dataclass

_DEFAULT_LOCAL_MODEL = "lfm2.5-230m"
_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
_DEFAULT_AZURE_API_VERSION = "2026-01-01-preview"


@dataclass
class ProviderConfig:
    name: str                        # "local" | "openai" | "azure"
    model: str
    base_url: str | None = None
    api_key: str | None = None
    azure_endpoint: str | None = None
    azure_deployment: str | None = None
    api_version: str | None = None


def resolve_provider(model_override: str | None = None) -> ProviderConfig:
    """Precedence: AZURE_OPENAI_API_KEY set -> azure; else OPENAI_API_KEY
    set -> openai; else -> local. Raises RuntimeError naming the missing
    var if a selected backend's other required vars aren't set — fails
    at construction time, never silently falls back to local."""

    azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if azure_key:
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise RuntimeError(
                "AZURE_OPENAI_ENDPOINT is required when AZURE_OPENAI_API_KEY is set"
            )
        deployment = model_override or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        if not deployment:
            raise RuntimeError(
                "AZURE_OPENAI_DEPLOYMENT is required when AZURE_OPENAI_API_KEY is "
                "set (or pass model=... to override it)"
            )
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", _DEFAULT_AZURE_API_VERSION)
        return ProviderConfig(
            name="azure",
            model=deployment,
            api_key=azure_key,
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            api_version=api_version,
        )

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        model = model_override or os.environ.get("OPENAI_MODEL", _DEFAULT_OPENAI_MODEL)
        return ProviderConfig(name="openai", model=model, api_key=openai_key)

    return ProviderConfig(
        name="local",
        model=model_override or _DEFAULT_LOCAL_MODEL,
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_provider.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add src/pos/llm/provider.py tests/test_provider.py
git commit -m "feat: add resolve_provider() for env-var LLM backend selection"
```

---

### Task 2: Pricing table + cost estimation

**Files:**
- Create: `src/pos/llm/pricing.py`
- Test: `tests/test_pricing.py`

**Interfaces:**
- Consumes: nothing from Task 1 (independent module).
- Produces: `PRICING: dict[tuple[str, str], tuple[float, float]]`, `price_for(provider: str, model: str) -> tuple[float, float] | None`, `estimate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float | None`. Task 4 (LangChainLlm usage population) calls `estimate_cost(...)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pricing.py`:

```python
from pos.llm.pricing import estimate_cost, price_for


def test_local_is_always_free(monkeypatch):
    monkeypatch.delenv("LOCAL_PRICE_INPUT_PER_1K", raising=False)
    monkeypatch.delenv("LOCAL_PRICE_OUTPUT_PER_1K", raising=False)

    assert price_for("local", "anything") == (0.0, 0.0)
    assert estimate_cost("local", "anything", input_tokens=999999, output_tokens=999999) == 0.0


def test_known_openai_model_uses_built_in_table(monkeypatch):
    monkeypatch.delenv("OPENAI_PRICE_INPUT_PER_1K", raising=False)
    monkeypatch.delenv("OPENAI_PRICE_OUTPUT_PER_1K", raising=False)

    prices = price_for("openai", "gpt-4o-mini")

    assert prices == (0.15, 0.60)


def test_unknown_model_with_no_override_returns_none(monkeypatch):
    monkeypatch.delenv("OPENAI_PRICE_INPUT_PER_1K", raising=False)
    monkeypatch.delenv("OPENAI_PRICE_OUTPUT_PER_1K", raising=False)

    assert price_for("openai", "some-unlisted-model") is None
    assert estimate_cost("openai", "some-unlisted-model", 1000, 1000) is None


def test_azure_has_no_built_in_default(monkeypatch):
    monkeypatch.delenv("AZURE_PRICE_INPUT_PER_1K", raising=False)
    monkeypatch.delenv("AZURE_PRICE_OUTPUT_PER_1K", raising=False)

    assert price_for("azure", "my-deployment") is None


def test_env_override_wins_over_built_in_table(monkeypatch):
    monkeypatch.setenv("OPENAI_PRICE_INPUT_PER_1K", "1.0")
    monkeypatch.setenv("OPENAI_PRICE_OUTPUT_PER_1K", "2.0")

    assert price_for("openai", "gpt-4o-mini") == (1.0, 2.0)


def test_env_override_enables_pricing_for_azure(monkeypatch):
    monkeypatch.setenv("AZURE_PRICE_INPUT_PER_1K", "3.0")
    monkeypatch.setenv("AZURE_PRICE_OUTPUT_PER_1K", "4.0")

    assert price_for("azure", "my-deployment") == (3.0, 4.0)


def test_estimate_cost_computes_weighted_sum(monkeypatch):
    monkeypatch.setenv("OPENAI_PRICE_INPUT_PER_1K", "1.0")
    monkeypatch.setenv("OPENAI_PRICE_OUTPUT_PER_1K", "2.0")

    # 2000 input tokens @ $1.0/1K = $2.0; 500 output tokens @ $2.0/1K = $1.0
    cost = estimate_cost("openai", "gpt-4o-mini", input_tokens=2000, output_tokens=500)

    assert cost == 3.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pricing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pos.llm.pricing'`

- [ ] **Step 3: Write the implementation**

Create `src/pos/llm/pricing.py`:

```python
"""Best-effort LLM pricing lookup. LLM prices change over time and this
table is a point-in-time snapshot, not a guarantee — override any entry
with {PROVIDER}_PRICE_INPUT_PER_1K / {PROVIDER}_PRICE_OUTPUT_PER_1K env
vars (provider name upper-cased) if it's gone stale. Local is always
free and never consults either the table or an override."""

from __future__ import annotations

import os

PRICING: dict[tuple[str, str], tuple[float, float]] = {
    # (provider, model) -> (input $/1K tokens, output $/1K tokens)
    ("openai", "gpt-4o-mini"): (0.15, 0.60),
    ("openai", "gpt-4o"): (2.50, 10.00),
}


def price_for(provider: str, model: str) -> tuple[float, float] | None:
    if provider == "local":
        return (0.0, 0.0)

    env_in = os.environ.get(f"{provider.upper()}_PRICE_INPUT_PER_1K")
    env_out = os.environ.get(f"{provider.upper()}_PRICE_OUTPUT_PER_1K")
    if env_in is not None and env_out is not None:
        return (float(env_in), float(env_out))

    return PRICING.get((provider, model))


def estimate_cost(
    provider: str, model: str, input_tokens: int, output_tokens: int
) -> float | None:
    prices = price_for(provider, model)
    if prices is None:
        return None
    input_price, output_price = prices
    return (input_tokens / 1000) * input_price + (output_tokens / 1000) * output_price
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pricing.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/pos/llm/pricing.py tests/test_pricing.py
git commit -m "feat: add LLM pricing table and cost estimation"
```

---

### Task 2b: Context-window size lookup

**Files:**
- Create: `src/pos/llm/context_window.py`
- Test: `tests/test_context_window.py`

**Interfaces:**
- Consumes: `ProviderConfig` from Task 1 (`src/pos/llm/provider.py`) — only its `.name`, `.model`, `.base_url` fields.
- Produces: `CONTEXT_WINDOWS: dict[tuple[str, str], int]`, `get_context_window(provider: ProviderConfig) -> int | None`. Task 4 calls this once at `LangChainLlm.__init__` time (never per-turn — a live HTTP call to LM Studio on every turn would add latency to the voice loop) and stores the result as `self._context_window`; Task 5 reads that stored value into `usage["context_window"]`.

Local (LM Studio) exposes context length live via its REST API v0 —
`GET {host}/api/v0/models` returns each model's `max_context_length`.
This is a **different base path** than the OpenAI-compatible
`/v1/models` this project already calls elsewhere — `host` is
`provider.base_url` with its trailing `/v1` removed (e.g.
`http://localhost:1234/v1` -> `http://localhost:1234`). OpenAI has no
API for this at all (confirmed via research — long-standing gap, not
an oversight in this plan) and Azure doesn't either, so both fall back
to a small built-in table + env override, same pattern as `pricing.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_context_window.py`:

```python
from unittest.mock import MagicMock

from pos.llm.context_window import get_context_window
from pos.llm.provider import ProviderConfig


def _local_provider(model="meta-llama-3.1-8b-instruct"):
    return ProviderConfig(
        name="local", model=model,
        base_url="http://localhost:1234/v1", api_key="lm-studio",
    )


def test_local_queries_lm_studio_v0_models_endpoint(monkeypatch):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "object": "list",
        "data": [
            {"id": "qwen2-vl-7b-instruct", "max_context_length": 32768},
            {"id": "meta-llama-3.1-8b-instruct", "max_context_length": 131072},
        ],
    }
    mock_response.raise_for_status.return_value = None
    captured_url = {}

    def fake_get(url, timeout):
        captured_url["url"] = url
        return mock_response

    monkeypatch.setattr("pos.llm.context_window.requests.get", fake_get)

    window = get_context_window(_local_provider())

    assert window == 131072
    assert captured_url["url"] == "http://localhost:1234/api/v0/models"


def test_local_returns_none_when_model_not_found_in_response(monkeypatch):
    mock_response = MagicMock()
    mock_response.json.return_value = {"object": "list", "data": [{"id": "other-model", "max_context_length": 4096}]}
    mock_response.raise_for_status.return_value = None
    monkeypatch.setattr("pos.llm.context_window.requests.get", lambda url, timeout: mock_response)

    assert get_context_window(_local_provider(model="not-listed")) is None


def test_local_returns_none_on_request_failure(monkeypatch):
    def fake_get(url, timeout):
        raise ConnectionError("LM Studio not running")

    monkeypatch.setattr("pos.llm.context_window.requests.get", fake_get)

    assert get_context_window(_local_provider()) is None


def test_openai_known_model_uses_built_in_table(monkeypatch):
    monkeypatch.delenv("OPENAI_CONTEXT_WINDOW", raising=False)
    provider = ProviderConfig(name="openai", model="gpt-4o-mini", api_key="sk-test")

    assert get_context_window(provider) == 128_000


def test_openai_unknown_model_with_no_override_returns_none(monkeypatch):
    monkeypatch.delenv("OPENAI_CONTEXT_WINDOW", raising=False)
    provider = ProviderConfig(name="openai", model="some-unlisted-model", api_key="sk-test")

    assert get_context_window(provider) is None


def test_azure_has_no_built_in_default(monkeypatch):
    monkeypatch.delenv("AZURE_CONTEXT_WINDOW", raising=False)
    provider = ProviderConfig(name="azure", model="my-deployment", api_key="azure-key")

    assert get_context_window(provider) is None


def test_env_override_wins_for_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_CONTEXT_WINDOW", "200000")
    provider = ProviderConfig(name="openai", model="gpt-4o-mini", api_key="sk-test")

    assert get_context_window(provider) == 200000


def test_env_override_enables_azure(monkeypatch):
    monkeypatch.setenv("AZURE_CONTEXT_WINDOW", "128000")
    provider = ProviderConfig(name="azure", model="my-deployment", api_key="azure-key")

    assert get_context_window(provider) == 128000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_context_window.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pos.llm.context_window'`

- [ ] **Step 3: Write the implementation**

Create `src/pos/llm/context_window.py`:

```python
"""Context-window size lookup, paired with pricing.py's tables. Local
(LM Studio) exposes this live via its REST API v0 -- see
https://lmstudio.ai/docs/developer/rest/endpoints -- a DIFFERENT base
path (/api/v0/...) than the OpenAI-compatible /v1/models this project
calls elsewhere for the model-name dropdown. OpenAI's API has no
endpoint that returns a model's context window at all (confirmed via
research, not an oversight); Azure doesn't either, since it also
depends on which base model was deployed. Both fall back to a small
built-in table, same env-override pattern as pricing.py."""

from __future__ import annotations

import os

import requests

from .provider import ProviderConfig

CONTEXT_WINDOWS: dict[tuple[str, str], int] = {
    # (provider, model) -> max context window, in tokens.
    # Best-effort snapshot, same caveat as pricing.py's PRICING table --
    # override via {PROVIDER}_CONTEXT_WINDOW if this has changed or a
    # new model needs one.
    ("openai", "gpt-4o-mini"): 128_000,
    ("openai", "gpt-4o"): 128_000,
}


def _local_context_window(base_url: str | None, model: str) -> int | None:
    host = (base_url or "").removesuffix("/v1")
    try:
        resp = requests.get(f"{host}/api/v0/models", timeout=3)
        resp.raise_for_status()
        for entry in resp.json().get("data", []):
            if entry.get("id") == model:
                return entry.get("max_context_length")
    except Exception:
        return None
    return None


def get_context_window(provider: ProviderConfig) -> int | None:
    if provider.name == "local":
        return _local_context_window(provider.base_url, provider.model)

    env_val = os.environ.get(f"{provider.name.upper()}_CONTEXT_WINDOW")
    if env_val is not None:
        return int(env_val)

    return CONTEXT_WINDOWS.get((provider.name, provider.model))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_context_window.py -v`
Expected: PASS (9 tests)

Then run: `uv run pytest tests/ -q`
Expected: PASS, all tests green

- [ ] **Step 5: Commit**

```bash
git add src/pos/llm/context_window.py tests/test_context_window.py
git commit -m "feat: add context-window size lookup (live for local, table for cloud)"
```

---

### Task 3: `LlmBase.stream()` gains an optional `usage` parameter (interface conformance)

**Files:**
- Modify: `src/pos/interfaces/llm.py` (whole file, currently 21 lines)
- Modify: `src/pos/llm/openai_compatible.py:45` (the `stream` method signature)
- Modify: `tests/fakes.py` (the `FakeLlm` class, currently lines 42-49)
- Test: `tests/test_openai_compatible_llm.py` (extend), `tests/fakes.py` doubles as the "test" for `FakeLlm` itself via the new test below

**Interfaces:**
- Consumes: nothing from Tasks 1-2.
- Produces: `LlmBase.stream(self, messages: list[dict], cancel: threading.Event, usage: dict | None = None) -> Iterator[str]` — the contract Task 4 (`LangChainLlm`) implements for real and Task 6 (`Agent`) calls. `FakeLlm.__init__` gains `fake_usage: dict | None = None`; when set, `FakeLlm.stream()` copies its contents into the `usage` dict passed by the caller — this is what Task 6's `Agent` test uses to simulate a real usage-reporting LLM.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_openai_compatible_llm.py`:

```python
def test_stream_accepts_and_ignores_usage_param(monkeypatch):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = [_fake_chunk("hi"), _fake_chunk(None)]
    monkeypatch.setattr("pos.llm.openai_compatible.OpenAI", lambda **kw: mock_client)

    llm = OpenAiCompatibleLlm(warmup=False)
    cancel = threading.Event()
    usage = {}
    result = list(llm.stream([{"role": "user", "content": "hi"}], cancel, usage))

    assert result == ["hi"]
    assert usage == {}  # OpenAiCompatibleLlm doesn't populate it — accepted for interface conformance only
```

Create `tests/test_fakes.py`:

```python
import threading

from fakes import FakeLlm


def test_fake_llm_ignores_usage_by_default():
    llm = FakeLlm("hi there")
    usage = {}

    list(llm.stream([{"role": "user", "content": "hi"}], threading.Event(), usage))

    assert usage == {}


def test_fake_llm_reports_configured_fake_usage():
    fake_usage = {"provider": "local", "model": "x", "input_tokens": 5, "output_tokens": 3}
    llm = FakeLlm("hi there", fake_usage=fake_usage)
    usage = {}

    list(llm.stream([{"role": "user", "content": "hi"}], threading.Event(), usage))

    assert usage == fake_usage


def test_fake_llm_works_with_usage_none():
    llm = FakeLlm("hi there")

    result = list(llm.stream([{"role": "user", "content": "hi"}], threading.Event(), None))

    assert result == ["hi ", "there "]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_openai_compatible_llm.py tests/test_fakes.py -v`
Expected: FAIL — `TypeError: stream() takes 3 positional arguments but 4 were given` (both `OpenAiCompatibleLlm.stream` and `FakeLlm.stream` don't accept a third positional arg yet)

- [ ] **Step 3: Write the implementation**

Replace the full contents of `src/pos/interfaces/llm.py`:

```python
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator


class LlmBase(ABC):
    """Chat LLM: streams a response to one turn as text pieces.

    Fully stateless — callers pass the full prior-turns message list
    each call, and implementations own no conversation history *and no
    per-call metrics as instance state*. Timing (ttft, total) is the
    caller's job to measure around the stream() call, not something
    read back from the engine afterward — one instance can be shared
    across concurrent sessions (see server.py's provider cache), and a
    shared mutable last_ttft/last_total would race between them.

    `usage`, if passed, is a caller-owned dict (same pattern as
    `cancel`) that an implementation may fill in place with per-call
    usage/cost data once the stream finishes — not stored on self, so
    sharing one instance across concurrent sessions stays race-free.
    An implementation that doesn't track usage (or a call where the
    backend never reported it) simply leaves it untouched — callers
    must treat a missing key as "unknown," never assume zero.
    """

    @abstractmethod
    def stream(
        self, messages: list[dict], cancel: threading.Event, usage: dict | None = None
    ) -> Iterator[str]: ...
```

In `src/pos/llm/openai_compatible.py`, change the `stream` method signature (line 45) from:

```python
    def stream(self, messages: list[dict], cancel: threading.Event) -> Iterator[str]:
```

to:

```python
    def stream(
        self, messages: list[dict], cancel: threading.Event, usage: dict | None = None
    ) -> Iterator[str]:
```

(no other change to that method's body — it never reads or writes `usage`)

In `tests/fakes.py`, replace the `FakeLlm` class (lines 42-49) with:

```python
class FakeLlm(LlmBase):
    def __init__(self, reply: str = "hi there", fake_usage: dict | None = None):
        self.reply = reply
        self.calls: list[list[dict]] = []
        self.fake_usage = fake_usage

    def stream(
        self, messages: list[dict], cancel: threading.Event, usage: dict | None = None
    ) -> Iterator[str]:
        self.calls.append(messages)
        if usage is not None and self.fake_usage is not None:
            usage.update(self.fake_usage)
        for word in self.reply.split():
            yield word + " "
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_openai_compatible_llm.py tests/test_fakes.py -v`
Expected: PASS (5 new tests)

Then run: `uv run pytest tests/ -q`
Expected: PASS, 79 tests (48 existing + 11 from Task 1 + 7 from Task 2 + 9 from Task 2b + 4 new this task [1 in test_openai_compatible_llm.py + 3 in test_fakes.py] — this full-suite run is the regression check for everything so far)

- [ ] **Step 5: Commit**

```bash
git add src/pos/interfaces/llm.py src/pos/llm/openai_compatible.py tests/fakes.py tests/test_openai_compatible_llm.py tests/test_fakes.py
git commit -m "feat: add optional usage dict param to LlmBase.stream() contract"
```

---

### Task 4: `LangChainLlm` builds the model from `resolve_provider()`

**Files:**
- Modify: `src/pos/llm/langchain_llm.py` (constructor and imports; full current content is 92 lines, shown above)
- Test: `tests/test_langchain_llm.py` (extend existing file)

**Interfaces:**
- Consumes: `resolve_provider(model_override) -> ProviderConfig` from Task 1 (`src/pos/llm/provider.py`); `get_context_window(provider) -> int | None` from Task 2b (`src/pos/llm/context_window.py`).
- Produces: `LangChainLlm.provider: ProviderConfig` and `LangChainLlm._context_window: int | None` (instance attributes, both read by Task 5 when populating usage). `LangChainLlm.__init__` signature becomes `(self, model: str | None = None, system_prompt=..., max_tokens=120, timeout=30, tools=None, max_tool_rounds=3, warmup=True)` — note `base_url`/`api_key` are **removed** as direct constructor params (now derived from the resolved provider); no existing caller in this codebase passes them (`agent.py` calls `LangChainLlm()`, `server.py` calls `LangChainLlm(model=model)`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_langchain_llm.py` (keep all existing tests and helpers unchanged):

```python
def test_local_provider_builds_chat_openai_with_lm_studio_defaults(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured_kwargs = {}

    def fake_chat_openai(**kwargs):
        captured_kwargs.update(kwargs)
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = _FakeRunnable([[_text_chunk("hi")]])
        return mock_model

    monkeypatch.setattr("pos.llm.langchain_llm.ChatOpenAI", fake_chat_openai)

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

    monkeypatch.setattr("pos.llm.langchain_llm.ChatOpenAI", fake_chat_openai)

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

    monkeypatch.setattr("pos.llm.langchain_llm.AzureChatOpenAI", fake_azure_chat_openai)

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

    monkeypatch.setattr("pos.llm.langchain_llm.ChatOpenAI", fake_chat_openai)

    llm = LangChainLlm(model="gpt-4o-mini", tools=[], warmup=False)

    assert llm.provider.model == "gpt-4o-mini"
    assert captured_kwargs["model"] == "gpt-4o-mini"


def test_warmup_failure_message_omits_lm_studio_hint_for_non_local(monkeypatch, capsys):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mock_model = MagicMock()
    mock_model.invoke.side_effect = RuntimeError("boom")
    mock_model.bind_tools.return_value = _FakeRunnable([])
    monkeypatch.setattr("pos.llm.langchain_llm.ChatOpenAI", lambda **kw: mock_model)

    LangChainLlm(tools=[], warmup=True)

    out = capsys.readouterr().out
    assert "llm warm-up failed" in out
    assert "LM Studio" not in out


def test_context_window_resolved_once_at_construction(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    mock_model = MagicMock()
    mock_model.bind_tools.return_value = _FakeRunnable([[_text_chunk("hi")]])
    monkeypatch.setattr("pos.llm.langchain_llm.ChatOpenAI", lambda **kw: mock_model)
    calls = []

    def fake_get_context_window(provider):
        calls.append(provider)
        return 131072

    monkeypatch.setattr("pos.llm.langchain_llm.get_context_window", fake_get_context_window)

    llm = LangChainLlm(tools=[], warmup=False)

    assert llm._context_window == 131072
    assert len(calls) == 1  # called once at construction, not per stream() call
```

Also update every **existing** test in `tests/test_langchain_llm.py` that constructs `LangChainLlm(...)` to first clear both provider env vars, since a leftover `OPENAI_API_KEY`/`AZURE_OPENAI_API_KEY` in the test-running shell would otherwise silently switch which branch they exercise. Add this fixture at the top of the file (runs automatically for every test in this file, no per-test change needed):

```python
import pytest


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Default every test in this file to no real network call for the
    # context-window lookup (Task 2b's local branch hits LM Studio's
    # REST API) — test_context_window_resolved_once_at_construction
    # below overrides this per-test with its own monkeypatch.setattr.
    monkeypatch.setattr("pos.llm.langchain_llm.get_context_window", lambda provider: None)
```

(place this fixture and the `import pytest` near the top of the file, after the existing imports — the four pre-existing tests and `_make_llm` helper need no other change; they'll now reliably resolve to the local branch, with no network call for context window either)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_langchain_llm.py -v`
Expected: FAIL — the new tests fail because `LangChainLlm` doesn't have a `.provider` attribute yet and still takes `base_url`/`api_key` as its first two positional-or-keyword params instead of resolving them internally; `AzureChatOpenAI` isn't imported/referenced in `langchain_llm.py` yet either.

- [ ] **Step 3: Write the implementation**

Replace `src/pos/llm/langchain_llm.py` lines 1-53 (everything from the imports through the end of `__init__`, i.e. up to but not including the `stream` method) with:

```python
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
```

Leave the rest of the file (the `stream` method) untouched for this task — Task 5 modifies it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_langchain_llm.py -v`
Expected: PASS (10 tests: 4 pre-existing + 6 new)

Then run: `uv run pytest tests/ -q`
Expected: PASS, all tests green

- [ ] **Step 5: Commit**

```bash
git add src/pos/llm/langchain_llm.py tests/test_langchain_llm.py
git commit -m "feat: LangChainLlm builds its model from resolve_provider() (local/openai/azure)"
```

---

### Task 5: `LangChainLlm.stream()` populates the `usage` dict

**Files:**
- Modify: `src/pos/llm/langchain_llm.py` (the `stream` method, and its imports)
- Test: `tests/test_langchain_llm.py` (extend)

**Interfaces:**
- Consumes: `estimate_cost(provider, model, input_tokens, output_tokens) -> float | None` from Task 2 (`src/pos/llm/pricing.py`); `self.provider` and `self._context_window` from Task 4.
- Produces: `LangChainLlm.stream(self, messages, cancel, usage=None)` now fills `usage` in place with keys `provider`, `model`, `input_tokens`, `output_tokens`, `total_tokens`, `cost_usd`, `tool_calls` (a list of `{"name": str, "args": dict}`), `context_window` once the final tool-loop round finishes — this is what Task 6's `Agent` reads.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_langchain_llm.py`. First, extend the `_text_chunk`/`_tool_call_chunk` helpers' neighbor with a usage-carrying chunk helper and update `_FakeRunnable` is unchanged (it already just replays chunk lists) — add:

```python
def _usage_chunk(input_tokens, output_tokens):
    chunk = AIMessageChunk(content="")
    chunk.usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    return chunk
```

Then add the tests:

```python
def test_stream_populates_usage_dict_for_plain_text_reply(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_langchain_llm.py -v`
Expected: FAIL — `usage` stays `{}` for `test_stream_populates_usage_dict_for_plain_text_reply` and `test_stream_usage_includes_tool_calls_made` (current `stream()` ignores the parameter entirely; it doesn't even accept a third argument yet, so this actually fails with `TypeError: stream() takes 3 positional arguments but 4 were given` first)

- [ ] **Step 3: Write the implementation**

Add this import to `src/pos/llm/langchain_llm.py`, alongside the existing `from .provider import resolve_provider` line:

```python
from .pricing import estimate_cost
```

Replace the `stream` method (everything from `def stream(self, messages: list[dict], cancel: threading.Event) -> Iterator[str]:` to the end of the file) with:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_langchain_llm.py -v`
Expected: PASS (15 tests: 10 from Task 4 + 5 new)

Then run: `uv run pytest tests/ -q`
Expected: PASS, all tests green

- [ ] **Step 5: Commit**

```bash
git add src/pos/llm/langchain_llm.py tests/test_langchain_llm.py
git commit -m "feat: LangChainLlm.stream() reports token usage, cost, and tool calls"
```

---

### Task 6: Wire usage into `Agent` (console output + `bot_text` event)

**Files:**
- Modify: `src/pos/agent.py:337-371` (shown in full above)
- Test: `tests/test_agent.py` (extend)

**Interfaces:**
- Consumes: `LlmBase.stream(messages, cancel, usage)` from Task 3; `FakeLlm(reply, fake_usage=...)` from Task 3.
- Produces: nothing further downstream in this plan — this is the last task. (The future trace-UI sub-project reads the `usage` key this task adds to the `bot_text` event payload.)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent.py`:

```python
def test_bot_text_event_includes_usage_when_llm_reports_it(monkeypatch):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    events: list[tuple[str, dict]] = []
    fake_usage = {
        "provider": "local", "model": "lfm2.5-230m",
        "input_tokens": 12, "output_tokens": 4, "total_tokens": 16,
        "cost_usd": 0.0, "tool_calls": [], "context_window": 131072,
    }
    agent = _build_agent(
        vad=FakeVad(start_at=1, end_at=3), stt=FakeStt("hello"),
        llm=FakeLlm("hi there", fake_usage=fake_usage),
        on_event=lambda name, data: events.append((name, data)),
    )
    threads = agent.start()
    try:
        frame = np.zeros(config.FRAME, dtype=np.float32)
        for _ in range(3):
            agent.feed_audio(frame)
        deadline = time.time() + 2.0
        while not any(name == "bot_text" for name, _ in events) and time.time() < deadline:
            time.sleep(0.02)

        bot_events = [data for name, data in events if name == "bot_text"]
        assert bot_events == [{"text": "hi there ", "usage": fake_usage}]
    finally:
        agent.shutdown(threads)
```

No second test is needed for the "no usage" case — the pre-existing
`test_on_event_fires_user_text_and_bot_text` already asserts the bare
`{"text": "hi there "}` payload shape with no `"usage"` key (since
`FakeLlm`'s default `fake_usage=None` leaves `usage` empty). Re-running
the full suite in Step 4 is that regression check; nothing new to write.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent.py::test_bot_text_event_includes_usage_when_llm_reports_it -v`
Expected: FAIL — `bot_events == [{"text": "hi there "}]`, missing the `"usage"` key (assertion mismatch)

- [ ] **Step 3: Write the implementation**

In `src/pos/agent.py`, change the block from (current lines 341-371):

```python
        t_start = time.perf_counter()
        try:
            for piece in self.llm.stream(messages, self.cancel):
                if self.cancel.is_set() or turn != self.current_turn():
                    return
                if ttft is None:
                    ttft = time.perf_counter() - t_start
                full_response.append(piece)
                buf += piece
                buf = drain(buf)
                if buf is None:
                    return
        except Exception as e:
            print(f"   [llm failed: {e}]")
            return

        if buf.strip():
            enqueue(buf)

        if ttft is not None:
            total = time.perf_counter() - t_start
            self.m_ttft.append(ttft)
            self.m_llm.append(total)
            if config.VERBOSE_TIMING:
                print(f"      llm: ttft {ttft:.2f}s / total {total:.2f}s")

        if full_response:
            joined = "".join(full_response)
            self.conversation.append({"role": "user", "content": text})
            self.conversation.append({"role": "assistant", "content": joined})
            self._on_event("bot_text", {"text": joined})
```

to:

```python
        t_start = time.perf_counter()
        usage: dict = {}
        try:
            for piece in self.llm.stream(messages, self.cancel, usage):
                if self.cancel.is_set() or turn != self.current_turn():
                    return
                if ttft is None:
                    ttft = time.perf_counter() - t_start
                full_response.append(piece)
                buf += piece
                buf = drain(buf)
                if buf is None:
                    return
        except Exception as e:
            print(f"   [llm failed: {e}]")
            return

        if buf.strip():
            enqueue(buf)

        if ttft is not None:
            total = time.perf_counter() - t_start
            self.m_ttft.append(ttft)
            self.m_llm.append(total)
            if config.VERBOSE_TIMING:
                print(f"      llm: ttft {ttft:.2f}s / total {total:.2f}s")
            if usage:
                cost = usage.get("cost_usd")
                cost_str = f"${cost:.4f}" if cost is not None else "n/a"
                tools_str = ", ".join(c["name"] for c in usage.get("tool_calls", [])) or "none"
                context_window = usage.get("context_window")
                total_str = (
                    f"{usage.get('total_tokens')} total / {context_window} context"
                    if context_window is not None
                    else f"{usage.get('total_tokens')} total"
                )
                print(
                    f"      tokens: {usage.get('input_tokens')} in / "
                    f"{usage.get('output_tokens')} out ({total_str})"
                    f"   cost: {cost_str}   tools: {tools_str}"
                )

        if full_response:
            joined = "".join(full_response)
            self.conversation.append({"role": "user", "content": text})
            self.conversation.append({"role": "assistant", "content": joined})
            payload = {"text": joined}
            if usage:
                payload["usage"] = usage
            self._on_event("bot_text", payload)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_agent.py -v`
Expected: PASS, including the pre-existing `test_on_event_fires_user_text_and_bot_text` (still asserts the bare `{"text": "hi there "}` shape, which still holds since `FakeLlm`'s default `fake_usage=None` leaves `usage` empty) and the new `test_bot_text_event_includes_usage_when_llm_reports_it`

Then run the full suite: `uv run pytest tests/ -q`
Expected: PASS, all tests green (48 original + all tests added across Tasks 1-6 in this plan; Task 6 adds exactly one new test)

- [ ] **Step 5: Commit**

```bash
git add src/pos/agent.py tests/test_agent.py
git commit -m "feat: Agent surfaces LLM token usage/cost via console output and bot_text event"
```

---

### Task 7: Document the new env vars in README

**Files:**
- Modify: `README.md` (the "LLM: LangChain + tool calling" section added in an earlier session — search for that heading)

**Interfaces:**
- Consumes: nothing (documentation only).
- Produces: nothing consumed by other tasks — this is the last task in the plan.

- [ ] **Step 1: Add the provider/pricing env vars to the README**

Find the "### LLM: LangChain + tool calling" section in `README.md` (added when `LangChainLlm` was first introduced) and add a new subsection immediately after its existing `TAVILY_API_KEY` example block:

```markdown
**Backend selection** (env vars, decided once per run — no in-session picker):

| Backend | Selected by | Other vars |
|---|---|---|
| Local (default) | *(none of the below set)* | — talks to LM Studio, same as before |
| OpenAI | `OPENAI_API_KEY` | `OPENAI_MODEL` (default `gpt-4o-mini`) |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` (wins if both are set) | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION` (default `2026-01-01-preview`) |

Every turn's token usage and an estimated cost are printed alongside the
existing `llm: ttft ...` timing line, and included as a `usage` field on
the server's `bot_text` websocket event. Cost is always `$0` on the
local backend. For OpenAI, a small built-in price table covers a couple
of common models (a point-in-time snapshot — likely to drift; override
with `OPENAI_PRICE_INPUT_PER_1K`/`OPENAI_PRICE_OUTPUT_PER_1K`, both in
$/1K tokens). Azure pricing varies per contract/region and has no
built-in default — set `AZURE_PRICE_INPUT_PER_1K`/`AZURE_PRICE_OUTPUT_PER_1K`
yourself, or cost shows as unavailable.

Context-window size is also reported (`usage["context_window"]`, and in
the console line as `.../<n> context`). For the local backend it's read
live from LM Studio's own REST API v0 (no setup needed). OpenAI and
Azure have no API that exposes this at all, so it comes from the same
kind of small built-in table + env override as pricing —
`OPENAI_CONTEXT_WINDOW`/`AZURE_CONTEXT_WINDOW` (plain integer, tokens)
— and is `None`/omitted from the console line if unset and the model
isn't in the built-in table.
```

- [ ] **Step 2: Verify the full suite is still green (documentation-only change, but confirms nothing else regressed)**

Run: `uv run pytest tests/ -q`
Expected: PASS, all tests green

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document LLM backend selection and pricing env vars"
```

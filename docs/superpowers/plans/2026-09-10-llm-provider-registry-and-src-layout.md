# LLM Provider Registry + src-layout Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the LLM provider if/elif branching (spread across `llm/provider.py`, `llm/pricing.py`, `llm/context_window.py`, `langchain_llm.py`) with a self-registering provider registry, then move every root-level Python entry point and `static/` into `src/pos/` with `[project.scripts]` console entries.

**Architecture:** Each backend (local/openai/azure) becomes one class implementing `LlmProviderBase` (`detect`, `resolve`, `build_model`, `price_for`, `context_window_for`), registered via a `@register` class decorator. `LangChainLlm` and everything else calls five small dispatch functions in `llm/providers/registry.py` and never branches on provider name again. Phase 2 is a pure location move — `main.py`/`server.py`/`ws_client.py` become `src/pos/cli/{local,server,relay_client}.py`, `static/` becomes `src/pos/static/`, invoked via `uv run pos-agent`/`pos-server`/`pos-client`.

**Tech Stack:** Python 3.14, `langchain-openai` (`ChatOpenAI`, `AzureChatOpenAI`), pytest, `hatchling` build backend.

**Spec:** `docs/superpowers/specs/2026-09-10-llm-provider-registry-and-src-layout-design.md`

## Global Constraints

- Every provider's env-var names, defaults, and error messages must stay byte-identical to today (`AZURE_OPENAI_API_KEY`/`AZURE_OPENAI_ENDPOINT`/`AZURE_OPENAI_DEPLOYMENT`/`AZURE_OPENAI_API_VERSION`, `OPENAI_API_KEY`/`OPENAI_MODEL`, `{PROVIDER}_PRICE_INPUT_PER_1K`/`..._OUTPUT_PER_1K`/`..._CONTEXT_WINDOW`) — this is a refactor, not a behavior change.
- `LangChainLlm` ends Phase 1 with **zero** `if self.provider.name == ...` branches anywhere in it (the one exception: the warmup-failure hint string check, which the spec explicitly keeps as a plain string comparison, not a registry lookup).
- Phase 2 changes no logic in any moved file — only file location, import paths, and (for `server.py`) wrapping the `if __name__ == "__main__":` body in a `run()` function.
- The full existing suite (`uv run pytest tests/ -q`) must stay green after every task. It has 134 tests as of this plan; expect the count to shift as Phase 1 redistributes tests across files (not a bug if the total changes, as long as nothing is silently dropped — the self-review below maps every old test to its new home).
- `uv run ruff check .` must stay clean (this project added it as its lint gate; see `pyproject.toml`'s `[tool.ruff]`).

---

### Task 1: Provider registry mechanism (`base.py` + `registry.py`)

**Files:**
- Create: `src/pos/llm/providers/__init__.py` (empty for now — populated in Task 5)
- Create: `src/pos/llm/providers/base.py`
- Create: `src/pos/llm/providers/registry.py`
- Test: `tests/test_provider_registry_mechanism.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `ProviderConfig` dataclass, `LlmProviderBase` ABC (`name: str`, `priority: int`, abstract `detect()`, `resolve(model_override)`, `build_model(provider, **kwargs)`, `price_for(provider)`, `context_window_for(provider)`), and `register`, `resolve_provider`, `build_model`, `price_for`, `context_window_for`, `estimate_cost` functions. Tasks 2-4 (local/openai/azure) each define a class implementing `LlmProviderBase` and decorate it with `@register`; Task 6 (`langchain_llm.py`) calls the five dispatch functions.

This task tests the registry *mechanism* in isolation, using fake
providers registered directly in the test — before any real provider
exists. `tests/test_providers_registry.py` (Task 5) later tests the
real local/openai/azure precedence once they're registered; this file
never touches those.

- [ ] **Step 1: Write the failing tests**

Create `src/pos/llm/providers/__init__.py` (empty file, so the
package can be imported before Task 5 fills it in):

```python
```

Create `tests/test_provider_registry_mechanism.py`:

```python
import pytest

from pos.llm.providers.base import LlmProviderBase, ProviderConfig
from pos.llm.providers.registry import (
    _REGISTRY,
    build_model,
    context_window_for,
    estimate_cost,
    price_for,
    register,
    resolve_provider,
)


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    # The real registry (Task 5 onward) is populated once at import time
    # and shared process-wide -- these tests register throwaway fakes
    # and must not leak them into (or see) that real registry.
    monkeypatch.setattr("pos.llm.providers.registry._REGISTRY", {})


class _FakeProvider(LlmProviderBase):
    name = "fake"
    priority = 100
    detected = True

    def detect(self):
        return self.detected

    def resolve(self, model_override):
        return ProviderConfig(name="fake", model=model_override or "fake-default-model")

    def build_model(self, provider, **model_kwargs):
        self.build_calls.append((provider, model_kwargs))
        return "fake-model-instance"

    def price_for(self, provider):
        return (1.0, 2.0)

    def context_window_for(self, provider):
        return 4096


def test_register_and_resolve_dispatches_to_the_only_registered_provider():
    register(_FakeProvider)

    provider = resolve_provider()

    assert provider.name == "fake"
    assert provider.model == "fake-default-model"


def test_resolve_provider_checks_providers_in_priority_order():
    class _HighPriority(_FakeProvider):
        name = "high"
        priority = 0

        def resolve(self, model_override):
            return ProviderConfig(name="high", model="high-model")

    class _LowPriority(_FakeProvider):
        name = "low"
        priority = 50

        def resolve(self, model_override):
            return ProviderConfig(name="low", model="low-model")

    register(_LowPriority)
    register(_HighPriority)

    provider = resolve_provider()

    assert provider.name == "high"  # lower priority number checked first


def test_resolve_provider_skips_providers_that_dont_detect():
    class _NotDetected(_FakeProvider):
        name = "not-detected"
        priority = 0
        detected = False

    class _Fallback(_FakeProvider):
        name = "fallback"
        priority = 100

    register(_NotDetected)
    register(_Fallback)

    provider = resolve_provider()

    assert provider.name == "fallback"


def test_build_model_dispatches_to_the_matching_registered_provider():
    _FakeProvider.build_calls = []
    register(_FakeProvider)
    fake_instance = _REGISTRY["fake"]
    provider_config = ProviderConfig(name="fake", model="m")

    result = build_model(provider_config, some_kwarg=1)

    assert result == "fake-model-instance"
    assert fake_instance.build_calls == [(provider_config, {"some_kwarg": 1})]


def test_price_for_and_context_window_for_dispatch_to_matching_provider():
    register(_FakeProvider)
    provider_config = ProviderConfig(name="fake", model="m")

    assert price_for(provider_config) == (1.0, 2.0)
    assert context_window_for(provider_config) == 4096


def test_estimate_cost_computes_weighted_sum_from_price_for():
    register(_FakeProvider)
    provider_config = ProviderConfig(name="fake", model="m")

    # price_for returns (1.0, 2.0): 2000 input @ $1/1K = $2, 500 output @ $2/1K = $1
    cost = estimate_cost(provider_config, input_tokens=2000, output_tokens=500)

    assert cost == 3.0


def test_estimate_cost_returns_none_when_unpriced():
    class _Unpriced(_FakeProvider):
        name = "unpriced"
        priority = 100

        def price_for(self, provider):
            return None

    register(_Unpriced)
    provider_config = ProviderConfig(name="unpriced", model="m")

    assert estimate_cost(provider_config, input_tokens=1000, output_tokens=1000) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_provider_registry_mechanism.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pos.llm.providers.base'`

- [ ] **Step 3: Write the implementation**

Create `src/pos/llm/providers/base.py`:

```python
"""The provider-registry contract. See registry.py for the dispatch
functions, and local.py/openai.py/azure.py for the concrete providers
-- see docs/superpowers/specs/2026-09-10-llm-provider-registry-and-src-layout-design.md
for why this replaced a scattered if/elif branching approach."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ProviderConfig:
    name: str                        # "local" | "openai" | "azure" | ...
    model: str
    base_url: str | None = None
    api_key: str | None = None
    azure_endpoint: str | None = None
    azure_deployment: str | None = None
    api_version: str | None = None


class LlmProviderBase(ABC):
    """One backend LangChainLlm can talk to. A new provider is one new
    file implementing this and decorated with @register -- no other
    file needs to change (see registry.py's resolve_provider(), which
    iterates whatever's registered rather than branching by name)."""

    name: str
    priority: int   # lower = checked first; the fallback provider uses the highest number

    @abstractmethod
    def detect(self) -> bool:
        """True if this provider's required env var(s) are present."""

    @abstractmethod
    def resolve(self, model_override: str | None) -> ProviderConfig:
        """Build this provider's config. May raise RuntimeError if
        detect() returned True but other required config is missing."""

    @abstractmethod
    def build_model(self, provider: ProviderConfig, **model_kwargs):
        """Construct and return this provider's langchain chat-model instance."""

    @abstractmethod
    def price_for(self, provider: ProviderConfig) -> tuple[float, float] | None:
        """(input $/1K tokens, output $/1K tokens), or None if unpriced."""

    @abstractmethod
    def context_window_for(self, provider: ProviderConfig) -> int | None:
        """Max context window in tokens, or None if unknown."""
```

Create `src/pos/llm/providers/registry.py`:

```python
"""Dispatch layer over whatever providers are registered -- see base.py
for the contract each provider implements. Providers self-register via
@register when their module is imported (see providers/__init__.py,
which imports local.py/openai.py/azure.py for exactly this side effect)."""

from __future__ import annotations

from .base import LlmProviderBase, ProviderConfig

_REGISTRY: dict[str, LlmProviderBase] = {}


def register(cls: type[LlmProviderBase]) -> type[LlmProviderBase]:
    _REGISTRY[cls.name] = cls()
    return cls


def resolve_provider(model_override: str | None = None) -> ProviderConfig:
    for provider in sorted(_REGISTRY.values(), key=lambda p: p.priority):
        if provider.detect():
            return provider.resolve(model_override)
    raise RuntimeError(
        "no LLM provider available -- this should be unreachable if a "
        "fallback provider (priority high enough, detect() always True) is registered"
    )


def build_model(provider: ProviderConfig, **model_kwargs):
    return _REGISTRY[provider.name].build_model(provider, **model_kwargs)


def price_for(provider: ProviderConfig) -> tuple[float, float] | None:
    return _REGISTRY[provider.name].price_for(provider)


def context_window_for(provider: ProviderConfig) -> int | None:
    return _REGISTRY[provider.name].context_window_for(provider)


def estimate_cost(provider: ProviderConfig, input_tokens: int, output_tokens: int) -> float | None:
    prices = price_for(provider)
    if prices is None:
        return None
    input_price, output_price = prices
    return (input_tokens / 1000) * input_price + (output_tokens / 1000) * output_price
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_provider_registry_mechanism.py -v`
Expected: PASS (7 tests)

Then run: `uv run pytest tests/ -q`
Expected: PASS, all previously-passing tests still pass (this task adds a new, self-contained package + test file; nothing existing imports it yet)

- [ ] **Step 5: Commit**

```bash
git add src/pos/llm/providers/__init__.py src/pos/llm/providers/base.py src/pos/llm/providers/registry.py tests/test_provider_registry_mechanism.py
git commit -m "feat: add LLM provider registry mechanism (base + dispatch)"
```

---

### Task 2: `LocalProvider`

**Files:**
- Create: `src/pos/llm/providers/local.py`
- Test: `tests/test_providers_local.py`

**Interfaces:**
- Consumes: `LlmProviderBase`, `ProviderConfig`, `register` from Task 1.
- Produces: `LocalProvider` class (registered as `"local"`, `priority=100`). Task 5 imports this module (for its registration side effect) into `providers/__init__.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_providers_local.py`:

```python
from unittest.mock import MagicMock

from pos.llm.providers.base import ProviderConfig
from pos.llm.providers.local import LocalProvider


def _local_provider(model="meta-llama-3.1-8b-instruct"):
    return ProviderConfig(name="local", model=model, base_url="http://localhost:1234/v1", api_key="lm-studio")


def test_detect_is_always_true():
    assert LocalProvider().detect() is True


def test_resolve_defaults_when_no_model_override():
    provider = LocalProvider().resolve(model_override=None)

    assert provider.name == "local"
    assert provider.model == "lfm2.5-230m"
    assert provider.base_url == "http://localhost:1234/v1"
    assert provider.api_key == "lm-studio"


def test_resolve_model_override_wins():
    provider = LocalProvider().resolve(model_override="custom-model")

    assert provider.model == "custom-model"


def test_price_for_is_always_free():
    assert LocalProvider().price_for(_local_provider()) == (0.0, 0.0)


def test_build_model_passes_expected_kwargs(monkeypatch):
    captured_kwargs = {}

    def fake_chat_openai(**kwargs):
        captured_kwargs.update(kwargs)
        return "the-model"

    monkeypatch.setattr("pos.llm.providers.local.ChatOpenAI", fake_chat_openai)

    result = LocalProvider().build_model(
        _local_provider(model="lfm2.5-230m"), max_tokens=120, temperature=0.7, timeout=30, stream_usage=True
    )

    assert result == "the-model"
    assert captured_kwargs["base_url"] == "http://localhost:1234/v1"
    assert captured_kwargs["api_key"] == "lm-studio"
    assert captured_kwargs["model"] == "lfm2.5-230m"
    assert captured_kwargs["stream_usage"] is True


def test_context_window_queries_lm_studio_v0_models_endpoint(monkeypatch):
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

    monkeypatch.setattr("pos.llm.providers.local.requests.get", fake_get)

    window = LocalProvider().context_window_for(_local_provider())

    assert window == 131072
    assert captured_url["url"] == "http://localhost:1234/api/v0/models"


def test_context_window_prefers_loaded_context_length_over_max_context_length(monkeypatch):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "object": "list",
        "data": [{"id": "lfm2.5-230m", "max_context_length": 128000, "loaded_context_length": 8192}],
    }
    mock_response.raise_for_status.return_value = None
    monkeypatch.setattr("pos.llm.providers.local.requests.get", lambda url, timeout: mock_response)

    assert LocalProvider().context_window_for(_local_provider(model="lfm2.5-230m")) == 8192


def test_context_window_falls_back_to_max_context_length_when_not_loaded(monkeypatch):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "object": "list",
        "data": [{"id": "lfm2.5-230m", "state": "not-loaded", "max_context_length": 128000}],
    }
    mock_response.raise_for_status.return_value = None
    monkeypatch.setattr("pos.llm.providers.local.requests.get", lambda url, timeout: mock_response)

    assert LocalProvider().context_window_for(_local_provider(model="lfm2.5-230m")) == 128000


def test_context_window_returns_none_when_model_not_found_in_response(monkeypatch):
    mock_response = MagicMock()
    mock_response.json.return_value = {"object": "list", "data": [{"id": "other-model", "max_context_length": 4096}]}
    mock_response.raise_for_status.return_value = None
    monkeypatch.setattr("pos.llm.providers.local.requests.get", lambda url, timeout: mock_response)

    assert LocalProvider().context_window_for(_local_provider(model="not-listed")) is None


def test_context_window_returns_none_on_request_failure(monkeypatch):
    def fake_get(url, timeout):
        raise ConnectionError("LM Studio not running")

    monkeypatch.setattr("pos.llm.providers.local.requests.get", fake_get)

    assert LocalProvider().context_window_for(_local_provider()) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_providers_local.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pos.llm.providers.local'`

- [ ] **Step 3: Write the implementation**

Create `src/pos/llm/providers/local.py`:

```python
"""LM Studio (or any OpenAI-compatible local server). Always detected
(the fallback provider -- lowest priority in resolve_provider()'s
checking order, i.e. the highest priority number). See
docs/superpowers/specs/2026-09-10-llm-provider-registry-and-src-layout-design.md."""

from __future__ import annotations

import requests
from langchain_openai import ChatOpenAI

from .base import LlmProviderBase, ProviderConfig
from .registry import register

_DEFAULT_MODEL = "lfm2.5-230m"


@register
class LocalProvider(LlmProviderBase):
    name = "local"
    priority = 100   # fallback -- always matches, checked last

    def detect(self) -> bool:
        return True

    def resolve(self, model_override: str | None) -> ProviderConfig:
        return ProviderConfig(
            name=self.name,
            model=model_override or _DEFAULT_MODEL,
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
        )

    def build_model(self, provider: ProviderConfig, **model_kwargs):
        return ChatOpenAI(
            base_url=provider.base_url,
            api_key=provider.api_key,
            model=provider.model,
            **model_kwargs,
        )

    def price_for(self, provider: ProviderConfig) -> tuple[float, float] | None:
        return (0.0, 0.0)

    def context_window_for(self, provider: ProviderConfig) -> int | None:
        host = (provider.base_url or "").removesuffix("/v1")
        try:
            resp = requests.get(f"{host}/api/v0/models", timeout=3)
            resp.raise_for_status()
            for entry in resp.json().get("data", []):
                if entry.get("id") == provider.model:
                    # loaded_context_length is what LM Studio actually
                    # configured for the running instance (e.g. 8192, set
                    # in its UI) -- max_context_length is the model's
                    # architectural ceiling (e.g. 128000) and can be much
                    # larger than what's really available. Prefer the
                    # real, currently-in-effect value; only fall back to
                    # the ceiling if the model isn't loaded
                    # (loaded_context_length absent/None in that state).
                    return entry.get("loaded_context_length") or entry.get("max_context_length")
        except Exception:
            return None
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_providers_local.py -v`
Expected: PASS (9 tests)

Then run: `uv run pytest tests/ -q`
Expected: PASS, all tests green

- [ ] **Step 5: Commit**

```bash
git add src/pos/llm/providers/local.py tests/test_providers_local.py
git commit -m "feat: add LocalProvider (LM Studio)"
```

---

### Task 3: `OpenAIProvider`

**Files:**
- Create: `src/pos/llm/providers/openai.py`
- Test: `tests/test_providers_openai.py`

**Interfaces:**
- Consumes: `LlmProviderBase`, `ProviderConfig`, `register` from Task 1.
- Produces: `OpenAIProvider` class (registered as `"openai"`, `priority=10`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_providers_openai.py`:

```python
from pos.llm.providers.base import ProviderConfig
from pos.llm.providers.openai import OpenAIProvider


def _openai_provider(model="gpt-4o-mini"):
    return ProviderConfig(name="openai", model=model, api_key="sk-test")


def test_detect_true_when_api_key_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    assert OpenAIProvider().detect() is True


def test_detect_false_when_api_key_not_set(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert OpenAIProvider().detect() is False


def test_resolve_uses_default_model_when_no_override_or_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    provider = OpenAIProvider().resolve(model_override=None)

    assert provider.name == "openai"
    assert provider.api_key == "sk-test"
    assert provider.model == "gpt-4o-mini"
    assert provider.base_url is None


def test_resolve_model_env_var_used_when_no_override(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")

    provider = OpenAIProvider().resolve(model_override=None)

    assert provider.model == "gpt-4o"


def test_resolve_model_override_wins_over_env_var(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")

    provider = OpenAIProvider().resolve(model_override="gpt-4o-mini")

    assert provider.model == "gpt-4o-mini"


def test_price_for_known_model_uses_built_in_table(monkeypatch):
    monkeypatch.delenv("OPENAI_PRICE_INPUT_PER_1K", raising=False)
    monkeypatch.delenv("OPENAI_PRICE_OUTPUT_PER_1K", raising=False)

    assert OpenAIProvider().price_for(_openai_provider()) == (0.15, 0.60)


def test_price_for_unknown_model_with_no_override_returns_none(monkeypatch):
    monkeypatch.delenv("OPENAI_PRICE_INPUT_PER_1K", raising=False)
    monkeypatch.delenv("OPENAI_PRICE_OUTPUT_PER_1K", raising=False)

    assert OpenAIProvider().price_for(_openai_provider(model="some-unlisted-model")) is None


def test_price_for_env_override_wins_over_built_in_table(monkeypatch):
    monkeypatch.setenv("OPENAI_PRICE_INPUT_PER_1K", "1.0")
    monkeypatch.setenv("OPENAI_PRICE_OUTPUT_PER_1K", "2.0")

    assert OpenAIProvider().price_for(_openai_provider()) == (1.0, 2.0)


def test_context_window_known_model_uses_built_in_table(monkeypatch):
    monkeypatch.delenv("OPENAI_CONTEXT_WINDOW", raising=False)

    assert OpenAIProvider().context_window_for(_openai_provider()) == 128_000


def test_context_window_unknown_model_with_no_override_returns_none(monkeypatch):
    monkeypatch.delenv("OPENAI_CONTEXT_WINDOW", raising=False)

    assert OpenAIProvider().context_window_for(_openai_provider(model="some-unlisted-model")) is None


def test_context_window_env_override_wins(monkeypatch):
    monkeypatch.setenv("OPENAI_CONTEXT_WINDOW", "200000")

    assert OpenAIProvider().context_window_for(_openai_provider()) == 200000


def test_build_model_passes_expected_kwargs(monkeypatch):
    captured_kwargs = {}

    def fake_chat_openai(**kwargs):
        captured_kwargs.update(kwargs)
        return "the-model"

    monkeypatch.setattr("pos.llm.providers.openai.ChatOpenAI", fake_chat_openai)

    result = OpenAIProvider().build_model(
        _openai_provider(), max_tokens=120, temperature=0.7, timeout=30, stream_usage=True
    )

    assert result == "the-model"
    assert captured_kwargs["base_url"] is None
    assert captured_kwargs["api_key"] == "sk-test"
    assert captured_kwargs["model"] == "gpt-4o-mini"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_providers_openai.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pos.llm.providers.openai'`

- [ ] **Step 3: Write the implementation**

Create `src/pos/llm/providers/openai.py`:

```python
"""OpenAI's own API. See
docs/superpowers/specs/2026-09-10-llm-provider-registry-and-src-layout-design.md."""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

from .base import LlmProviderBase, ProviderConfig
from .registry import register

_DEFAULT_MODEL = "gpt-4o-mini"

# Best-effort snapshot, not guaranteed current -- override via
# OPENAI_PRICE_INPUT_PER_1K/OPENAI_PRICE_OUTPUT_PER_1K if these have
# changed (they will, eventually).
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}

# Same caveat as PRICING -- override via OPENAI_CONTEXT_WINDOW.
CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
}


@register
class OpenAIProvider(LlmProviderBase):
    name = "openai"
    priority = 10

    def detect(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))

    def resolve(self, model_override: str | None) -> ProviderConfig:
        api_key = os.environ["OPENAI_API_KEY"]
        model = model_override or os.environ.get("OPENAI_MODEL", _DEFAULT_MODEL)
        return ProviderConfig(name=self.name, model=model, api_key=api_key)

    def build_model(self, provider: ProviderConfig, **model_kwargs):
        return ChatOpenAI(
            base_url=None,
            api_key=provider.api_key,
            model=provider.model,
            **model_kwargs,
        )

    def price_for(self, provider: ProviderConfig) -> tuple[float, float] | None:
        env_in = os.environ.get("OPENAI_PRICE_INPUT_PER_1K")
        env_out = os.environ.get("OPENAI_PRICE_OUTPUT_PER_1K")
        if env_in is not None and env_out is not None:
            return (float(env_in), float(env_out))
        return PRICING.get(provider.model)

    def context_window_for(self, provider: ProviderConfig) -> int | None:
        env_val = os.environ.get("OPENAI_CONTEXT_WINDOW")
        if env_val is not None:
            return int(env_val)
        return CONTEXT_WINDOWS.get(provider.model)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_providers_openai.py -v`
Expected: PASS (11 tests)

Then run: `uv run pytest tests/ -q`
Expected: PASS, all tests green

- [ ] **Step 5: Commit**

```bash
git add src/pos/llm/providers/openai.py tests/test_providers_openai.py
git commit -m "feat: add OpenAIProvider"
```

---

### Task 4: `AzureProvider`

**Files:**
- Create: `src/pos/llm/providers/azure.py`
- Test: `tests/test_providers_azure.py`

**Interfaces:**
- Consumes: `LlmProviderBase`, `ProviderConfig`, `register` from Task 1.
- Produces: `AzureProvider` class (registered as `"azure"`, `priority=0` — checked first).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_providers_azure.py`:

```python
import pytest

from pos.llm.providers.azure import AzureProvider
from pos.llm.providers.base import ProviderConfig


def _azure_provider(model="my-deployment"):
    return ProviderConfig(name="azure", model=model, api_key="azure-key")


def test_detect_true_when_api_key_set(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")

    assert AzureProvider().detect() is True


def test_detect_false_when_api_key_not_set(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    assert AzureProvider().detect() is False


def test_resolve_raises_when_endpoint_missing(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "my-deployment")

    with pytest.raises(RuntimeError, match="AZURE_OPENAI_ENDPOINT"):
        AzureProvider().resolve(model_override=None)


def test_resolve_raises_when_deployment_missing_and_no_override(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)

    with pytest.raises(RuntimeError, match="AZURE_OPENAI_DEPLOYMENT"):
        AzureProvider().resolve(model_override=None)


def test_resolve_model_override_wins_over_deployment_env_var(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "my-deployment")

    provider = AzureProvider().resolve(model_override="other-deployment")

    assert provider.model == "other-deployment"
    assert provider.azure_deployment == "other-deployment"


def test_resolve_api_version_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "my-deployment")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-05-01")

    provider = AzureProvider().resolve(model_override=None)

    assert provider.api_version == "2025-05-01"


def test_resolve_api_version_defaults_when_not_set(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "my-deployment")
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)

    provider = AzureProvider().resolve(model_override=None)

    assert provider.api_version == "2026-01-01-preview"


def test_price_for_has_no_built_in_default(monkeypatch):
    monkeypatch.delenv("AZURE_PRICE_INPUT_PER_1K", raising=False)
    monkeypatch.delenv("AZURE_PRICE_OUTPUT_PER_1K", raising=False)

    assert AzureProvider().price_for(_azure_provider()) is None


def test_price_for_env_override_enables_pricing(monkeypatch):
    monkeypatch.setenv("AZURE_PRICE_INPUT_PER_1K", "3.0")
    monkeypatch.setenv("AZURE_PRICE_OUTPUT_PER_1K", "4.0")

    assert AzureProvider().price_for(_azure_provider()) == (3.0, 4.0)


def test_context_window_has_no_built_in_default(monkeypatch):
    monkeypatch.delenv("AZURE_CONTEXT_WINDOW", raising=False)

    assert AzureProvider().context_window_for(_azure_provider()) is None


def test_context_window_env_override_enables_it(monkeypatch):
    monkeypatch.setenv("AZURE_CONTEXT_WINDOW", "128000")

    assert AzureProvider().context_window_for(_azure_provider()) == 128000


def test_build_model_passes_expected_kwargs(monkeypatch):
    captured_kwargs = {}

    def fake_azure_chat_openai(**kwargs):
        captured_kwargs.update(kwargs)
        return "the-model"

    monkeypatch.setattr("pos.llm.providers.azure.AzureChatOpenAI", fake_azure_chat_openai)
    provider = ProviderConfig(
        name="azure", model="my-deployment", api_key="azure-key",
        azure_endpoint="https://example.openai.azure.com/", azure_deployment="my-deployment",
        api_version="2026-01-01-preview",
    )

    result = AzureProvider().build_model(provider, max_tokens=120, temperature=0.7, timeout=30, stream_usage=True)

    assert result == "the-model"
    assert captured_kwargs["azure_endpoint"] == "https://example.openai.azure.com/"
    assert captured_kwargs["azure_deployment"] == "my-deployment"
    assert captured_kwargs["api_key"] == "azure-key"
    assert captured_kwargs["api_version"] == "2026-01-01-preview"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_providers_azure.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pos.llm.providers.azure'`

- [ ] **Step 3: Write the implementation**

Create `src/pos/llm/providers/azure.py`:

```python
"""Azure OpenAI. See
docs/superpowers/specs/2026-09-10-llm-provider-registry-and-src-layout-design.md."""

from __future__ import annotations

import os

from langchain_openai import AzureChatOpenAI

from .base import LlmProviderBase, ProviderConfig
from .registry import register

_DEFAULT_API_VERSION = "2026-01-01-preview"


@register
class AzureProvider(LlmProviderBase):
    name = "azure"
    priority = 0   # checked first

    def detect(self) -> bool:
        return bool(os.environ.get("AZURE_OPENAI_API_KEY"))

    def resolve(self, model_override: str | None) -> ProviderConfig:
        api_key = os.environ["AZURE_OPENAI_API_KEY"]
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
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", _DEFAULT_API_VERSION)
        return ProviderConfig(
            name=self.name,
            model=deployment,
            api_key=api_key,
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            api_version=api_version,
        )

    def build_model(self, provider: ProviderConfig, **model_kwargs):
        return AzureChatOpenAI(
            azure_endpoint=provider.azure_endpoint,
            azure_deployment=provider.azure_deployment,
            api_version=provider.api_version,
            api_key=provider.api_key,
            **model_kwargs,
        )

    def price_for(self, provider: ProviderConfig) -> tuple[float, float] | None:
        env_in = os.environ.get("AZURE_PRICE_INPUT_PER_1K")
        env_out = os.environ.get("AZURE_PRICE_OUTPUT_PER_1K")
        if env_in is not None and env_out is not None:
            return (float(env_in), float(env_out))
        return None

    def context_window_for(self, provider: ProviderConfig) -> int | None:
        env_val = os.environ.get("AZURE_CONTEXT_WINDOW")
        if env_val is not None:
            return int(env_val)
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_providers_azure.py -v`
Expected: PASS (11 tests)

Then run: `uv run pytest tests/ -q`
Expected: PASS, all tests green

- [ ] **Step 5: Commit**

```bash
git add src/pos/llm/providers/azure.py tests/test_providers_azure.py
git commit -m "feat: add AzureProvider"
```

---

### Task 5: Wire up `providers/__init__.py` + real end-to-end precedence tests

**Files:**
- Modify: `src/pos/llm/providers/__init__.py`
- Test: `tests/test_providers_registry.py`

**Interfaces:**
- Consumes: `AzureProvider`, `LocalProvider`, `OpenAIProvider` (Tasks 2-4); `ProviderConfig`, `resolve_provider`, `build_model`, `price_for`, `context_window_for`, `estimate_cost` (Task 1).
- Produces: `from pos.llm.providers import ...` as the one public import surface for everything above. Task 6 (`langchain_llm.py`) imports from here.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_providers_registry.py` (this is an *integration* test
of the real registry, once all three real providers are registered —
distinct from Task 1's `test_provider_registry_mechanism.py`, which
tests the mechanism with throwaway fakes and resets the registry
between tests):

```python
import pytest

from pos.llm.providers import ProviderConfig, estimate_cost, resolve_provider


def test_defaults_to_local_when_no_env_vars_set(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    provider = resolve_provider()

    assert provider.name == "local"
    assert provider.base_url == "http://localhost:1234/v1"
    assert provider.api_key == "lm-studio"
    assert provider.model == "lfm2.5-230m"


def test_openai_backend_selected_when_key_set(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    provider = resolve_provider()

    assert provider.name == "openai"
    assert provider.api_key == "sk-test"
    assert provider.model == "gpt-4o-mini"


def test_azure_backend_takes_precedence_over_openai(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "my-deployment")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")  # both set — azure must win

    provider = resolve_provider()

    assert provider.name == "azure"
    assert provider.model == "my-deployment"


def test_model_override_wins_regardless_of_which_provider_is_selected(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    provider = resolve_provider(model_override="gpt-4o")

    assert provider.model == "gpt-4o"


def test_azure_missing_endpoint_raises(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "my-deployment")

    with pytest.raises(RuntimeError, match="AZURE_OPENAI_ENDPOINT"):
        resolve_provider()


def test_estimate_cost_computes_weighted_sum(monkeypatch):
    monkeypatch.setenv("OPENAI_PRICE_INPUT_PER_1K", "1.0")
    monkeypatch.setenv("OPENAI_PRICE_OUTPUT_PER_1K", "2.0")
    provider = ProviderConfig(name="openai", model="gpt-4o-mini", api_key="sk-test")

    # 2000 input tokens @ $1.0/1K = $2.0; 500 output tokens @ $2.0/1K = $1.0
    cost = estimate_cost(provider, input_tokens=2000, output_tokens=500)

    assert cost == 3.0


def test_estimate_cost_local_always_zero():
    provider = ProviderConfig(name="local", model="anything", base_url="http://localhost:1234/v1", api_key="lm-studio")

    assert estimate_cost(provider, input_tokens=999999, output_tokens=999999) == 0.0


def test_estimate_cost_returns_none_when_unpriced(monkeypatch):
    monkeypatch.delenv("OPENAI_PRICE_INPUT_PER_1K", raising=False)
    monkeypatch.delenv("OPENAI_PRICE_OUTPUT_PER_1K", raising=False)
    provider = ProviderConfig(name="openai", model="some-unlisted-model", api_key="sk-test")

    assert estimate_cost(provider, input_tokens=1000, output_tokens=1000) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_providers_registry.py -v`
Expected: FAIL — `resolve_provider()` raises `RuntimeError: no LLM provider available` for every test (nothing is registered yet — `providers/__init__.py` is still empty from Task 1)

- [ ] **Step 3: Write the implementation**

Replace `src/pos/llm/providers/__init__.py`:

```python
"""Public surface for the LLM provider registry. Importing this module
imports azure.py/local.py/openai.py purely for their @register side
effect -- each self-registers into registry.py's _REGISTRY when its
module loads. Adding a new provider means adding one file and one
import line here; resolve_provider() and friends never need to change."""

from .azure import AzureProvider
from .base import ProviderConfig
from .local import LocalProvider
from .openai import OpenAIProvider
from .registry import build_model, context_window_for, estimate_cost, price_for, resolve_provider

__all__ = [
    "ProviderConfig",
    "resolve_provider",
    "build_model",
    "price_for",
    "context_window_for",
    "estimate_cost",
    "AzureProvider",
    "LocalProvider",
    "OpenAIProvider",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_providers_registry.py -v`
Expected: PASS (8 tests)

Then run: `uv run pytest tests/ -q`
Expected: PASS, all tests green

- [ ] **Step 5: Commit**

```bash
git add src/pos/llm/providers/__init__.py tests/test_providers_registry.py
git commit -m "feat: wire up the LLM provider registry (local+openai+azure registered)"
```

---

### Task 6: Switch `LangChainLlm` to the registry, delete the old provider files

**Files:**
- Modify: `src/pos/llm/langchain_llm.py`
- Delete: `src/pos/llm/provider.py`, `src/pos/llm/pricing.py`, `src/pos/llm/context_window.py`
- Delete: `tests/test_provider.py`, `tests/test_pricing.py`, `tests/test_context_window.py`
- Modify: `tests/test_langchain_llm.py`

**Interfaces:**
- Consumes: `build_model`, `context_window_for`, `estimate_cost`, `resolve_provider` from `pos.llm.providers` (Task 5).
- Produces: nothing further downstream in this plan — this is the last Phase 1 task.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_langchain_llm.py` in full (this removes the four
tests that tested *provider construction* through the full
`LangChainLlm` — `test_local_provider_builds_chat_openai_with_lm_studio_defaults`,
`test_openai_provider_builds_chat_openai_without_base_url`,
`test_azure_provider_builds_azure_chat_openai`,
`test_model_kwarg_overrides_env_derived_model` — since that's exactly
what Tasks 2-4's `test_build_model_passes_expected_kwargs` tests now
cover directly against each provider; adds one new test confirming
`LangChainLlm.__init__` itself passes the right kwargs *into*
`build_model`; and renames every mock target from
`ChatOpenAI`/`AzureChatOpenAI`/`get_context_window` to `build_model`/`context_window_for`,
since those are the names `langchain_llm.py` imports after this task):

```python
import threading
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessageChunk

from pos.llm.langchain_llm import LangChainLlm


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Default every test in this file to no real network call for the
    # context-window lookup (local's provider hits LM Studio's REST
    # API) — test_context_window_resolved_once_at_construction below
    # overrides this per-test with its own monkeypatch.setattr.
    monkeypatch.setattr("pos.llm.langchain_llm.context_window_for", lambda provider: None)


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
    """Stands in for `build_model(...).bind_tools([...])`. `rounds` is a
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
    monkeypatch.setattr("pos.llm.langchain_llm.build_model", lambda provider, **kw: mock_model)
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


def test_init_passes_expected_kwargs_to_build_model(monkeypatch):
    captured = {}

    def fake_build_model(provider, **kwargs):
        captured.update(kwargs)
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = _FakeRunnable([])
        return mock_model

    monkeypatch.setattr("pos.llm.langchain_llm.build_model", fake_build_model)

    LangChainLlm(max_tokens=99, timeout=12, tools=[], warmup=False)

    assert captured["max_tokens"] == 99
    assert captured["temperature"] == 0.7
    assert captured["timeout"] == 12
    assert captured["stream_usage"] is True


def test_warmup_failure_message_omits_lm_studio_hint_for_non_local(monkeypatch, capsys):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mock_model = MagicMock()
    mock_model.invoke.side_effect = RuntimeError("boom")
    mock_model.bind_tools.return_value = _FakeRunnable([])
    monkeypatch.setattr("pos.llm.langchain_llm.build_model", lambda provider, **kw: mock_model)

    LangChainLlm(tools=[], warmup=True, warmup_attempts=1)

    out = capsys.readouterr().out
    assert "llm warm-up failed" in out
    assert "LM Studio" not in out


def test_warmup_retries_on_failure_and_succeeds_before_attempts_exhausted(monkeypatch, capsys):
    monkeypatch.setattr("pos.llm.langchain_llm.time.sleep", lambda seconds: None)
    mock_model = MagicMock()
    mock_model.invoke.side_effect = [ConnectionError("not up yet"), MagicMock()]
    mock_model.bind_tools.return_value = _FakeRunnable([])
    monkeypatch.setattr("pos.llm.langchain_llm.build_model", lambda provider, **kw: mock_model)

    LangChainLlm(tools=[], warmup=True, warmup_attempts=3, warmup_backoff_base=1.0)

    out = capsys.readouterr().out
    assert mock_model.invoke.call_count == 2
    assert "llm warm-up:" in out
    assert "llm warm-up failed" not in out


def test_warmup_gives_up_and_logs_after_exhausting_attempts(monkeypatch, capsys):
    monkeypatch.setattr("pos.llm.langchain_llm.time.sleep", lambda seconds: None)
    mock_model = MagicMock()
    mock_model.invoke.side_effect = ConnectionError("still not up")
    mock_model.bind_tools.return_value = _FakeRunnable([])
    monkeypatch.setattr("pos.llm.langchain_llm.build_model", lambda provider, **kw: mock_model)

    LangChainLlm(tools=[], warmup=True, warmup_attempts=3, warmup_backoff_base=1.0)

    out = capsys.readouterr().out
    assert mock_model.invoke.call_count == 3
    assert "llm warm-up failed after 3 attempts" in out


def test_context_window_resolved_once_at_construction(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    mock_model = MagicMock()
    mock_model.bind_tools.return_value = _FakeRunnable([[_text_chunk("hi")]])
    monkeypatch.setattr("pos.llm.langchain_llm.build_model", lambda provider, **kw: mock_model)
    calls = []

    def fake_context_window_for(provider):
        calls.append(provider)
        return 131072

    monkeypatch.setattr("pos.llm.langchain_llm.context_window_for", fake_context_window_for)

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
    llm._context_window = 131072  # simulate what __init__ would have resolved

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

    def fake_context_window_for(provider):
        calls.append(provider)
        return 131072

    monkeypatch.setattr("pos.llm.langchain_llm.context_window_for", fake_context_window_for)

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_langchain_llm.py -v`
Expected: FAIL — `AttributeError: <module 'pos.llm.langchain_llm'> does not have the attribute 'build_model'` (langchain_llm.py hasn't been updated yet)

- [ ] **Step 3: Write the implementation**

In `src/pos/llm/langchain_llm.py`, replace the imports:

```python
from __future__ import annotations

import threading
import time
from collections.abc import Iterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from ..interfaces.llm import LlmBase
from .providers import build_model, context_window_for, estimate_cost, resolve_provider
from .tools import default_tools
```

(removes `from langchain_openai import AzureChatOpenAI, ChatOpenAI` and
the three old `.context_window`/`.pricing`/`.provider` imports)

Replace `__init__`'s body from `self._model = self._build_model(...)`
onward, and delete the `_build_model` method entirely:

```python
        self.provider = resolve_provider(model_override=model)
        self._context_window = context_window_for(self.provider)
        self.system_prompt = system_prompt
        self.max_tool_rounds = max_tool_rounds
        self.tools = default_tools() if tools is None else tools
        self._tools_by_name = {t.name: t for t in self.tools}

        self._model = build_model(
            self.provider, max_tokens=max_tokens, temperature=0.7, timeout=timeout, stream_usage=True
        )
        self._runnable = self._model.bind_tools(self.tools) if self.tools else self._model

        if warmup:
            self._warmup(warmup_attempts, warmup_backoff_base)
```

(the `_build_model` method that used to follow `_warmup` is deleted —
there is no replacement method, `build_model` is now the imported
registry function called directly above)

In `_warmup`, change both `get_context_window(self.provider)` calls to
`context_window_for(self.provider)` (there is exactly one such call, in
the success branch after a successful `invoke`).

In `_fill_usage`, change the lazy-retry line:

```python
        if self._context_window is None:
            self._context_window = context_window_for(self.provider)
```

and the cost line:

```python
        cost = None
        if input_tokens is not None and output_tokens is not None:
            cost = estimate_cost(self.provider, input_tokens, output_tokens)
```

Everything else in the file (`generate_title`, `stream`, the rest of
`_fill_usage`) is unchanged.

Delete the three old files:

```bash
git rm src/pos/llm/provider.py src/pos/llm/pricing.py src/pos/llm/context_window.py
git rm tests/test_provider.py tests/test_pricing.py tests/test_context_window.py
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_langchain_llm.py -v`
Expected: PASS (21 tests)

Then run: `uv run pytest tests/ -q`
Expected: PASS, all tests green. Then run `uv run ruff check .` — expect
no complaints (no leftover unused imports from the deleted files).

- [ ] **Step 5: Commit**

```bash
git add src/pos/llm/langchain_llm.py tests/test_langchain_llm.py
git commit -m "refactor: switch LangChainLlm to the provider registry, delete old provider files"
```

---

### Task 7: `src/pos/cli/` — move `main.py`

**Files:**
- Create: `src/pos/cli/__init__.py` (empty)
- Create: `src/pos/cli/local.py` (moved from root `main.py`, unchanged content)
- Delete: `main.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: nothing new — this file's own existing imports (`Agent`, `SileroVad`, `SessionStore`, etc.) are unaffected, since it already imports everything via `from pos... import ...` (absolute imports, not relative to its own location).
- Produces: `pos.cli.local.main()`, wired to the `pos-agent` console script.

- [ ] **Step 1: Move the file**

```bash
mkdir -p src/pos/cli
touch src/pos/cli/__init__.py
git mv main.py src/pos/cli/local.py
```

- [ ] **Step 2: Add the console-script entry**

In `pyproject.toml`, add a new top-level table (after `[dependency-groups]`,
before `[tool.ruff]`, or anywhere at the top level — TOML table order
doesn't matter):

```toml
[project.scripts]
pos-agent = "pos.cli.local:main"
```

- [ ] **Step 3: Verify**

Run: `uv sync` (re-installs the package so the new console script is
registered)
Expected: completes without error

Run: `uv run pos-agent --help`
Expected: prints the same `--tts`/`--voice`/`--trigger-word`/`--vad-*`/`--mic`/`--list-mics`
help text this used to print as `uv run main.py --help`

Run: `uv run pytest tests/ -q`
Expected: PASS, all tests green (nothing imports `main.py`/`pos.cli.local` today, so this is a pure sanity check)

- [ ] **Step 4: Commit**

```bash
git add src/pos/cli/__init__.py src/pos/cli/local.py pyproject.toml
git commit -m "refactor: move main.py into src/pos/cli/local.py, add pos-agent script"
```

---

### Task 8: `src/pos/cli/server.py` + `static/` move

**Files:**
- Create: `src/pos/cli/server.py` (moved from root `server.py`, one path fix + `__main__` wrapped in `run()`)
- Create: `src/pos/static/index.html` (moved from root `static/index.html`)
- Delete: `server.py`, `static/` (the now-empty root directory)
- Modify: `tests/test_server.py` (import path)
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: nothing new.
- Produces: `pos.cli.server.create_app(...)` (same signature as
  today's `server.create_app`), `pos.cli.server.run()` (new — the
  former `if __name__ == "__main__":` body), wired to the `pos-server`
  console script.

- [ ] **Step 1: Move the files**

```bash
git mv server.py src/pos/cli/server.py
mkdir -p src/pos/static
git mv static/index.html src/pos/static/index.html
rmdir static 2>/dev/null || true
```

- [ ] **Step 2: Fix the static-file path and wrap `__main__` in `run()`**

In `src/pos/cli/server.py`, change:

```python
        return FileResponse(Path(__file__).parent / "static" / "index.html")
```

to (one more `.parent` — `cli/server.py` is now one directory deeper
than the repo root `server.py` was, and `static/` moved from being a
sibling of the old `server.py` to being a sibling of `cli/`, i.e.
`pos/static/`):

```python
        return FileResponse(Path(__file__).parent.parent / "static" / "index.html")
```

Replace the file's existing tail:

```python
if __name__ == "__main__":
    import uvicorn

    from pos.stt import OnnxAsrEngine

    app = create_app(stt=OnnxAsrEngine())
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

with:

```python
def run() -> None:
    import uvicorn

    from pos.stt import OnnxAsrEngine

    app = create_app(stt=OnnxAsrEngine())
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()
```

- [ ] **Step 3: Update the test import**

In `tests/test_server.py`, change:

```python
from server import _default_llm_models, create_app
```

to:

```python
from pos.cli.server import _default_llm_models, create_app
```

- [ ] **Step 4: Add a `GET /` smoke test**

Add to `tests/test_server.py` (this catches a wrong static-file path
immediately, rather than only on manual browser testing):

```python
def test_index_page_is_served():
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm(),
        default_tts_engine="kokoro",
        session_store=SessionStore(":memory:"),
    )
    client = TestClient(app)

    resp = client.get("/")

    assert resp.status_code == 200
    assert "Voice Agent" in resp.text
```

- [ ] **Step 5: Add the console-script entry**

In `pyproject.toml`'s `[project.scripts]` table (created in Task 7):

```toml
[project.scripts]
pos-agent = "pos.cli.local:main"
pos-server = "pos.cli.server:run"
```

- [ ] **Step 6: Verify**

Run: `uv sync`
Expected: completes without error

Run: `uv run pytest tests/test_server.py -v`
Expected: PASS, including the new `test_index_page_is_served`

Run: `uv run pytest tests/ -q`
Expected: PASS, all tests green

Run: `uv run pos-server` in one terminal, then in another: `curl http://localhost:8000/` (or open it in a browser)
Expected: the same page that `uv run server.py` used to serve. Stop the
server (Ctrl+C) once confirmed.

- [ ] **Step 7: Commit**

```bash
git add src/pos/cli/server.py src/pos/static/index.html tests/test_server.py pyproject.toml
git commit -m "refactor: move server.py into src/pos/cli/, static/ into src/pos/static/, add pos-server script"
```

---

### Task 9: `src/pos/cli/relay_client.py`

**Files:**
- Create: `src/pos/cli/relay_client.py` (moved from root `ws_client.py`, unchanged content)
- Delete: `ws_client.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: nothing new.
- Produces: `pos.cli.relay_client.main()`, wired to the `pos-client` console script.

- [ ] **Step 1: Move the file**

```bash
git mv ws_client.py src/pos/cli/relay_client.py
```

- [ ] **Step 2: Add the console-script entry**

In `pyproject.toml`'s `[project.scripts]` table:

```toml
[project.scripts]
pos-agent = "pos.cli.local:main"
pos-server = "pos.cli.server:run"
pos-client = "pos.cli.relay_client:main"
```

- [ ] **Step 3: Verify**

Run: `uv sync`
Expected: completes without error

Run: `uv run pos-client --help`
Expected: prints the same help text this used to print as `uv run ws_client.py --help`

Run: `uv run pytest tests/ -q`
Expected: PASS, all tests green

- [ ] **Step 4: Commit**

```bash
git add src/pos/cli/relay_client.py pyproject.toml
git commit -m "refactor: move ws_client.py into src/pos/cli/relay_client.py, add pos-client script"
```

---

### Task 10: Delete `conftest.py`, update the README

**Files:**
- Delete: `conftest.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing — this is the last task in the plan.

- [ ] **Step 1: Delete `conftest.py`**

```bash
git rm conftest.py
```

`conftest.py`'s only purpose (per its own docstring) was letting tests
`from server import ...` without packaging the root scripts. Every test
now imports `pos.cli.server`/etc. as a normal installed package
module — no `sys.path` trick needed.

- [ ] **Step 2: Verify the suite still passes without it**

Run: `uv run pytest tests/ -q`
Expected: PASS, all tests green (confirms no test was relying on
`conftest.py`'s `sys.path` insertion for something other than the
now-gone `from server import ...` pattern)

- [ ] **Step 3: Update the README**

In `README.md`, replace every occurrence of:
- `uv run main.py` → `uv run pos-agent`
- `uv run server.py` → `uv run pos-server`
- `uv run ws_client.py` → `uv run pos-client`

(these appear in the "Usage", "Running: local vs. server", "Mic
selection and mute", "VAD tuning", "LLM: LangChain + tool calling", and
"CLI client provider selection" sections)

Replace the Architecture section's file tree — change:

```
main.py                          local-mode CLI entrypoint (--tts, --voice, --trigger-word, --vad-*)
server.py                        FastAPI multi-session websocket server (see "Running" above)
ws_client.py                     Python CLI relay client for server.py
static/index.html                browser relay client for server.py — voice/text toggle, live
                                  transcript, settings, session History sidebar
conftest.py                      empty — puts the repo root on sys.path so tests can
                                  `from server import ...` without packaging the root scripts
src/pos/
```

to:

```
src/pos/
  cli/
    local.py                     local-mode CLI entrypoint (--tts, --voice, --trigger-word, --vad-*)
                                  — console script: `uv run pos-agent`
    server.py                    FastAPI multi-session websocket server (see "Running" above)
                                  — console script: `uv run pos-server`
    relay_client.py               Python CLI relay client for server.py
                                  — console script: `uv run pos-client`
  static/index.html               browser relay client for server.py — voice/text toggle, live
                                  transcript, settings, session History sidebar
```

(the rest of the `src/pos/` tree entries — `config.py`, `agent.py`,
`storage.py`, `interfaces/`, `audio/`, `vad/`, `stt/`, `tts/`, `llm/` —
are unchanged by this plan's moves, except `llm/` gains the new
`providers/` subpackage from Phase 1; add one line for it:)

```
  llm/langchain_llm.py           LangChainLlm(LlmBase) — ChatOpenAI/AzureChatOpenAI + bound
                                  tools, hand-rolled tool loop, usage/cost/context-window
                                  reporting, generate_title()
  llm/providers/                 provider registry — base.py (LlmProviderBase ABC + ProviderConfig),
                                  registry.py (dispatch), local.py/openai.py/azure.py (one
                                  self-registering provider each; add a new backend by adding
                                  one file here, no other file needs to change)
  llm/tools.py                   get_current_time, web search (Tavily, needs TAVILY_API_KEY)
  llm/openai_compatible.py       OpenAiCompatibleLlm(LlmBase) — plain OpenAI SDK, no tool calling, kept for reference/tests
```

(remove the now-stale `llm/provider.py`, `llm/pricing.py`,
`llm/context_window.py` lines from the tree — those files are deleted)

- [ ] **Step 4: Final full verification**

Run: `uv run pytest tests/ -q`
Expected: PASS, all tests green

Run: `uv run ruff check .`
Expected: `All checks passed!`

Run: `git status --porcelain`
Expected: clean (everything committed) once Step 5 below runs

- [ ] **Step 5: Commit**

```bash
git add conftest.py README.md
git commit -m "docs: update README for pos-agent/pos-server/pos-client, remove conftest.py"
```

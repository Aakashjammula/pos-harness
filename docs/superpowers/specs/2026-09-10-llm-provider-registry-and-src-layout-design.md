# LLM provider registry + src-layout reorganization

Date: 2026-09-10
Status: draft, awaiting review

## Context

Two related pieces of cleanup, done in this order because the second
would otherwise need redoing once the first lands:

**1. Provider registry.** `LangChainLlm` currently selects its backend
via manual branching spread across three files: `llm/provider.py`'s
`resolve_provider()` (`if azure_key: ... elif openai_key: ... else:
local`), `llm/langchain_llm.py`'s `_build_model()` (`if
self.provider.name == "azure": AzureChatOpenAI(...) else:
ChatOpenAI(...)`), plus per-provider `if`/env-var branches duplicated
again in `llm/pricing.py` and `llm/context_window.py`. Researched
against how this actually breaks down at scale (LiteLLM's own
provider-abstraction approach, the general "registry pattern vs.
if/elif chains" literature): every new provider today means editing
four functions across three files, none of which live near each other.
The **Registry pattern** — each provider is a self-contained class that
registers itself; core code does a dict lookup instead of a branch
chain — is the standard fix, and is directly justified here since more
providers are planned.

**2. src-layout reorganization.** `main.py`, `server.py`, `ws_client.py`
and `static/` currently sit at the repo root as loose files/scripts,
while all real package code already lives under `src/asr_test/`.
Researched current best practice (pyOpenSci's packaging guide, Real
Python's project-layout reference, and confirming this is what `uv
init` itself now produces by default): entry-point scripts belong
*inside* the package too, invoked via `[project.scripts]` console-script
entries in `pyproject.toml` (`command-name = "package.module:function"`)
rather than as loose root files or thin wrapper scripts. This is a pure
reorganization — no class, interface, or behavior changes; every moved
function keeps its exact signature and logic.

## Decisions from brainstorming

- Registry: each provider is a class implementing a new
  `LlmProviderBase` ABC (`detect`, `resolve`, `build_model`, `price_for`,
  `context_window_for`) and self-registers via a `@register` class
  decorator when its module is imported. `llm/provider.py`,
  `llm/pricing.py`, and `llm/context_window.py` are deleted — their
  logic is fully redistributed into `llm/providers/{local,openai,azure}.py`.
- `LangChainLlm` ends up with **zero** provider-name branching anywhere
  in it — every provider-specific decision (env-var detection, which
  LangChain chat-model class to construct, pricing, context-window
  lookup) lives inside that provider's own file.
- Registry dispatch functions (`resolve_provider`, `build_model`,
  `price_for`, `context_window_for`, `estimate_cost`) all take/return
  the same `ProviderConfig` shape as today — this is an internal
  refactor, not a behavior or public-contract change. Every existing
  env var (`OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`,
  `{PROVIDER}_PRICE_INPUT_PER_1K`, etc.) keeps working identically.
- File moves: `main.py` → `src/asr_test/cli/local.py`, `server.py` →
  `src/asr_test/cli/server.py`, `ws_client.py` →
  `src/asr_test/cli/relay_client.py`, `static/` →
  `src/asr_test/static/`. No root-level Python files remain except
  `conftest.py`, which becomes unnecessary and is deleted (its whole
  purpose was letting tests `from server import ...` without a proper
  package import; once `server.py` is `asr_test.cli.server`, tests
  import it the normal way).
- New `[project.scripts]`: `asr-agent`, `asr-server`, `asr-client` —
  confirmed with the user. `uv run server.py` becomes `uv run
  asr-server` (and similarly for the other two); this is a deliberate,
  documented change to the README, not an oversight.
- Each moved file's actual code (functions, classes, logic) is
  unchanged — this phase is a location/import-path change only. The
  provider registry refactor (phase 1) is a real behavior-preserving
  code change; the file move (phase 2) is not.

## Phase 1 — Provider registry

### `llm/providers/base.py`

```python
@dataclass
class ProviderConfig:
    name: str
    model: str
    base_url: str | None = None
    api_key: str | None = None
    azure_endpoint: str | None = None
    azure_deployment: str | None = None
    api_version: str | None = None


class LlmProviderBase(ABC):
    name: str
    priority: int   # lower = checked first; the fallback provider uses the highest number

    @abstractmethod
    def detect(self) -> bool: ...                                    # True if this provider's env vars are present
    @abstractmethod
    def resolve(self, model_override: str | None) -> ProviderConfig: ...
    @abstractmethod
    def build_model(self, provider: ProviderConfig, **model_kwargs): ...   # returns a langchain chat model instance
    @abstractmethod
    def price_for(self, provider: ProviderConfig) -> tuple[float, float] | None: ...
    @abstractmethod
    def context_window_for(self, provider: ProviderConfig) -> int | None: ...
```

Matches this codebase's existing ABC convention (`interfaces/*.py`) —
kept inside `llm/providers/`, not the top-level `interfaces/` package,
since this is an internal detail of one engine (`LangChainLlm`'s
backend selection), not a pipeline-stage contract like `VadBase`/`TtsBase`.

### `llm/providers/registry.py`

```python
_REGISTRY: dict[str, LlmProviderBase] = {}

def register(cls: type[LlmProviderBase]) -> type[LlmProviderBase]:
    _REGISTRY[cls().name] = cls()
    return cls

def resolve_provider(model_override: str | None = None) -> ProviderConfig:
    for provider in sorted(_REGISTRY.values(), key=lambda p: p.priority):
        if provider.detect():
            return provider.resolve(model_override)
    raise RuntimeError("no LLM provider available (unreachable -- local always detects True)")

def build_model(provider: ProviderConfig, **model_kwargs): ...        # dispatches to _REGISTRY[provider.name]
def price_for(provider: ProviderConfig) -> tuple[float, float] | None: ...
def context_window_for(provider: ProviderConfig) -> int | None: ...
def estimate_cost(provider: ProviderConfig, input_tokens: int, output_tokens: int) -> float | None: ...
```

`estimate_cost` replaces `pricing.py`'s standalone function, computed
from `price_for()`'s result — kept in the registry module (not
per-provider) since the arithmetic is identical regardless of provider,
only the price *table* varies.

### `llm/providers/local.py`, `openai.py`, `azure.py`

One `@register`-decorated class each, holding exactly the logic that's
today scattered across `provider.py`/`pricing.py`/`context_window.py`
for that one backend:

- **local** (`priority = 100`, the fallback — `detect()` always
  `True`): `resolve()` returns the `http://localhost:1234/v1` /
  `lm-studio` defaults (env-overridable model via `model_override`
  only, no env var — matches today, local has never had a
  `LOCAL_MODEL` env var). `price_for()` always `(0.0, 0.0)`.
  `context_window_for()` does the live `GET
  {base_url-without-/v1}/api/v0/models` call, preferring
  `loaded_context_length` over `max_context_length` — identical logic
  to today's `context_window.py`, just relocated.
- **openai** (`priority = 10`, `detect()` checks `OPENAI_API_KEY`):
  `resolve()` reads `OPENAI_MODEL` (default `gpt-4o-mini`).
  `price_for()`/`context_window_for()` check
  `OPENAI_PRICE_INPUT_PER_1K`/`OPENAI_PRICE_OUTPUT_PER_1K`/`OPENAI_CONTEXT_WINDOW`
  env overrides first, then this provider's own built-in
  `PRICING`/`CONTEXT_WINDOWS` dicts (same two entries as today:
  `gpt-4o-mini`, `gpt-4o`).
- **azure** (`priority = 0`, checked first, `detect()` checks
  `AZURE_OPENAI_API_KEY`): `resolve()` raises `RuntimeError` naming the
  missing var if `AZURE_OPENAI_ENDPOINT`/`AZURE_OPENAI_DEPLOYMENT`
  aren't set (same error-message text as today).
  `price_for()`/`context_window_for()` only check
  `AZURE_PRICE_INPUT_PER_1K`/etc. env vars — no built-in table (Azure
  pricing is contract-specific, same reasoning as today).

### `llm/langchain_llm.py` changes

Removes `_build_model()` entirely, and the `ChatOpenAI`/`AzureChatOpenAI`
imports (no longer referenced in this file at all). `__init__` becomes:

```python
self.provider = resolve_provider(model_override=model)
self._context_window = context_window_for(self.provider)
...
self._model = build_model(
    self.provider, max_tokens=max_tokens, temperature=0.7, timeout=timeout, stream_usage=True
)
```

`_fill_usage()`'s cost line becomes `cost = estimate_cost(self.provider,
input_tokens, output_tokens)`; its lazy context-window retry becomes
`self._context_window = context_window_for(self.provider)`. The
warmup-failure hint (`"is LM Studio running?"` only for local) keeps
checking `self.provider.name == "local"` — a plain string comparison,
not a registry lookup; harmless, `"local"` is a stable identifier, not
worth adding a `hint_for_failure()` method to the ABC for one string.

### Testing

Old `tests/test_provider.py`, `tests/test_pricing.py`,
`tests/test_context_window.py` are deleted; their coverage is
redistributed:

- `tests/test_providers_registry.py`: `resolve_provider()` precedence
  (azure beats openai beats local — same cases as today's
  `test_provider.py`), `build_model`/`price_for`/`context_window_for`
  dispatch to the right registered provider, `estimate_cost` arithmetic.
- `tests/test_providers_local.py`: local's `resolve`/`price_for`
  (always free)/`context_window_for` (live API call, mocked
  `requests.get` — same cases as today's `test_context_window.py`'s
  local-branch tests).
- `tests/test_providers_openai.py`: env-var precedence, built-in table,
  unknown-model-returns-`None` (same cases as today's
  `test_pricing.py`/`test_context_window.py`'s openai-branch tests).
- `tests/test_providers_azure.py`: missing-var errors, no built-in
  default, env-override-only pricing/context-window.
- `tests/test_langchain_llm.py`: mocks move from
  `asr_test.llm.langchain_llm.ChatOpenAI`/`AzureChatOpenAI` to
  `asr_test.llm.providers.local.ChatOpenAI` /
  `asr_test.llm.providers.openai.ChatOpenAI` /
  `asr_test.llm.providers.azure.AzureChatOpenAI` (each provider module
  imports its own langchain class now); `get_context_window` mock
  becomes `asr_test.llm.providers.local.LocalProvider.context_window_for`
  or simpler, monkeypatch `asr_test.llm.langchain_llm.context_window_for`
  (the name `langchain_llm.py` imports) — same pattern as today, just
  the dotted path changes since the function moved modules.

## Phase 2 — src-layout reorganization

Pure file moves + import-path updates, no logic changes:

| From | To |
|---|---|
| `main.py` | `src/asr_test/cli/local.py` |
| `server.py` | `src/asr_test/cli/server.py` |
| `ws_client.py` | `src/asr_test/cli/relay_client.py` |
| `static/index.html` | `src/asr_test/static/index.html` |
| `conftest.py` | deleted (no longer needed) |

`src/asr_test/cli/__init__.py` — empty (marker file only).

`server.py`'s `FileResponse(Path(__file__).parent / "static" / "index.html")`
becomes `Path(__file__).parent.parent / "static" / "index.html"` (one
more `.parent` — `cli/server.py` is now one level deeper than the repo
root was).

`pyproject.toml` gains:

```toml
[project.scripts]
asr-agent = "asr_test.cli.local:main"
asr-server = "asr_test.cli.server:run"
asr-client = "asr_test.cli.relay_client:main"
```

`server.py`'s existing `if __name__ == "__main__":` block (constructs
`OnnxAsrEngine`, calls `create_app()`, calls `uvicorn.run()`) becomes a
`def run() -> None:` function, called from both `if __name__ ==
"__main__":` (so `uv run python -m asr_test.cli.server` still works
during development) and the console-script entry point.

Every test's import changes: `from server import create_app,
_default_llm_models` → `from asr_test.cli.server import create_app,
_default_llm_models` (and similarly anywhere `main`/`ws_client` internals
are tested — currently neither has direct test coverage beyond
`py_compile`, so this only affects `tests/test_server.py`).

`README.md`'s every `uv run server.py`/`uv run main.py`/`uv run
ws_client.py` becomes `uv run asr-server`/`uv run asr-agent`/`uv run
asr-client`; the Architecture file tree is updated to show the new
`cli/` and `static/` locations under `src/asr_test/`.

## Error handling

- Registry: `resolve_provider()` raising `RuntimeError` for a truly
  unreachable state (no provider detected, including local) is a
  defensive assertion, not an expected runtime path — local's
  `detect()` always returns `True`, so this can only happen if a future
  change removes that guarantee.
- File move: `server.py`'s static-file path change is the one place a
  wrong relative path would silently 404 instead of erroring at import
  time — covered by a smoke-style test that `GET /` returns 200 with
  the actual moved file's content.

## Testing (phase 2)

- `tests/test_server.py`: update the import; add one test asserting
  `GET /` returns 200 (catches a wrong static path immediately rather
  than only on manual browser testing).
- Manual verification (no automated coverage for CLI scripts today,
  consistent with the existing pattern): `uv run asr-agent --help`,
  `uv run asr-server` then `GET /`, `uv run asr-client --help` all work
  after the move.

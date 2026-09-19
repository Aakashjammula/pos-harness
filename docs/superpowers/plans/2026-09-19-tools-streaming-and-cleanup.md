# Tools, Streaming Chat and Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make tools a first-class, user-selectable feature (each tool in its own module, key-requiring tools get a key form in a separate Tools UI), stream text chat over HTTP (SSE) using the same generator voice mode uses, and close the remaining correctness/ops leftovers.

**Architecture:** A `pos.tools` registry (one `ToolSpec` per tool module) drives credential fields, the `/tools` API, per-user on/off switches (new `user_tool_settings` table) and the tools bound to `LangChainLlm`. Text chat moves from `/ws?mode=text` to `POST /chat/stream` (Server-Sent Events, Anthropic-style event flow); both it and voice's `Agent.respond` call one shared `run_turn()` around `LlmBase.stream()`. `/ws` becomes voice-only.

**Tech Stack:** Python 3.14, FastAPI, psycopg, LangChain, pytest; Next.js 16 / React / TypeScript; Docker Compose.

**Spec:** No separate spec file: the requirements are the "Requirements" section below, taken from the user's instructions in the working session (tool requirements verbatim: *"tavily is supposed to be tool ... option for user to select tool and tools will have separate UI where if that tool require api then they need to keep ... store all tools separately in [code] in function"*; *"we need it streaming api which is same function used in ws when in voice mode"*; *"remove ws text mode"*).

## Global Constraints

- **Never run backend tests against a non-`*_test` database.** `tests/conftest.py` refuses; the default is `postgresql://pos:pos@localhost:5433/pos_test`. Never point `TEST_DATABASE_URL` at `pos` (a past run wiped the dev DB twice).
- Do not create accounts in the real `pos` database for manual checks; if unavoidable, use one throwaway `@example.com` user and `DELETE` it by email afterwards.
- No LLM endpoint, model name or API key is assumed anywhere (no `localhost:1234`, no `lfm2.5-230m`).
- Secrets travel in headers or the credential store, never in URLs or error text.
- Backend line length 120 (ruff). Frontend must pass `npx tsc --noEmit` and `npx eslint src`; the repo's lint forbids `setState` synchronously inside effects.
- Commit trailer on every commit: `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.
- Run backend tests with: `cd backend && uv run pytest -q -p no:cacheprovider`.

## Requirements (the 20 tasks, grouped)

| # | Requirement | Task |
|---|---|---|
| 1 | Tools registry; each tool in its own module/function | 1 |
| 2 | Per-user tool selection stored server-side | 2 |
| 3 | Tool credential fields come from the registry; `GET/PUT /tools` | 3 |
| 4 | Server binds each user's selected tools; per-user cloud LLMs skip warm-up | 4 |
| 5 | Separate Tools UI; Tavily key moves there; blank-key check | 5 |
| 6 | Fix list-content (`str + list`) bug in `LlmBase.stream` and titles | 6 |
| 7 | Shared one-turn helper used by voice and text | 7 |
| 8 | `POST /chat/stream` SSE | 8 |
| 9 | Remove WS text mode; stop text-mode log noise | 9 |
| 10 | Frontend text mode streams over SSE | 10 |
| 11 | Surface magic-link mail failures | 11 |
| 12 | Investigate `GET /sessions → 401` | 12 |
| 13 | Neutral LM Studio URL placeholder | 13 |
| 14 | Bedrock listing: verify or mark untested | 14 |
| 15 | List models for server-env provider keys | 15 |
| 16 | Backend runs as non-root | 16 |
| 17 | Rebuild + live verification (incl. Gemini 400 fix, SSE via `curl -N`) | 17 |
| 18 | Browser check of model dropdown, Tools panel and streaming | 18 |
| 19 | README/docs | 19 |
| 20 | Final full-suite/lint gate and commit | 20 |

## File Structure

- `backend/src/pos/tools/{base,registry,current_time,web_search,__init__}.py` — tool registry (files already created; Task 1 tests/wires them).
- `backend/src/pos/turn.py` — `run_turn()` / `TurnStats` (Task 7).
- `backend/src/pos/llm/content.py` — `content_text()` (Task 6).
- `backend/src/pos/db.py` — `user_tool_settings` table (Task 2).
- `backend/src/pos/auth/store.py` — tool-setting accessors (Task 2).
- `backend/src/pos/auth/routes.py` — credential fields from registry, `/tools` routes (Task 3).
- `backend/src/pos/cli/server.py` — per-user tools, `/chat/stream`, voice-only `/ws` (Tasks 4, 8, 9).
- `backend/src/pos/agent.py` — use `run_turn`, drop text-only paths (Tasks 7, 9).
- `frontend/src/components/ToolsPanel.tsx`, `frontend/src/hooks/useTools.ts`, `frontend/src/lib/tools.ts` — Tools UI (Task 5).
- `frontend/src/hooks/useVoiceSession.ts`, `frontend/src/lib/chatStream.ts` — SSE text chat (Task 10).
- Delete `backend/src/pos/llm/tools.py`, `backend/tests/test_tools.py` (replaced).

---

### Task 1: Tool registry wired into LangChainLlm

**Files:**
- Create (already on disk): `backend/src/pos/tools/*.py`
- Modify: `backend/src/pos/llm/langchain_llm.py` (import + `__init__`), `backend/src/pos/cli/server.py:101` (import `tool_status`)
- Delete: `backend/src/pos/llm/tools.py`, `backend/tests/test_tools.py`
- Test: `backend/tests/test_tools_registry.py`

**Interfaces:**
- Produces: `pos.tools.build_tools(enabled: Iterable[str] | None, env: Mapping | None) -> list[BaseTool]`, `resolve_enabled(settings: Mapping[str,bool], env) -> list[str]`, `tool_status(env, settings) -> list[dict]`, `credential_fields() -> dict[str, dict[str,str]]`, `all_tools() -> list[ToolSpec]`, `get_tool(id) -> ToolSpec | None`.
- `LangChainLlm.__init__` gains `enabled_tools: Iterable[str] | None = None`.

- [ ] **Step 1: Write the failing test** `backend/tests/test_tools_registry.py`

```python
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


def test_build_tools_passes_the_key_explicitly_and_never_touches_os_environ(monkeypatch):
    import os

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    tools = build_tools(["web_search"], env={"TAVILY_API_KEY": "tvly-override"})
    assert [t.name for t in tools] == ["tavily_search"]
    assert "TAVILY_API_KEY" not in os.environ


def test_tool_status_never_constructs_a_tool(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("constructed a tool")

    monkeypatch.setattr(get_tool("web_search"), "build", boom, raising=False) if False else None
    status = tool_status({"TAVILY_API_KEY": "k"}, {})
    assert {s["name"]: s["enabled"] for s in status} == {"get_current_time": True, "web_search": True}


def test_registering_a_duplicate_id_is_rejected():
    dup = ToolSpec(id="get_current_time", label="x", description="x", build=lambda env: None)
    with pytest.raises(ValueError, match="duplicate"):
        register(dup)
```

- [ ] **Step 2: Run to see it fail** — `uv run pytest tests/test_tools_registry.py -q` → import errors until Step 3 completes (package exists but `langchain_llm` still imports the old module; the registry tests themselves should already pass — if any fail, fix the registry files, not the tests).

- [ ] **Step 3: Wire `LangChainLlm` and delete the old module**

In `backend/src/pos/llm/langchain_llm.py` replace `from .tools import default_tools` with `from ..tools import build_tools`; add `enabled_tools: Iterable[str] | None = None,` to `__init__` (import `Iterable` from `collections.abc`); replace
`self.tools = default_tools(env=env) if tools is None else tools` with
`self.tools = build_tools(enabled_tools, env) if tools is None else tools`.
In `backend/src/pos/cli/server.py` replace `from pos.llm.tools import tool_status` with `from pos.tools import tool_status`. Then `git rm backend/src/pos/llm/tools.py backend/tests/test_tools.py`. In `tests/test_langchain_llm.py` change the monkeypatch target `pos.llm.langchain_llm.default_tools` to `pos.llm.langchain_llm.build_tools` and its fake signature to `def fake_build_tools(enabled=None, env=None)`; update the `seen["env"]` assertion accordingly.

- [ ] **Step 4: Run the whole suite** — `uv run pytest -q -p no:cacheprovider` → all pass.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: tool registry, one module per tool"`.

---

### Task 2: Per-user tool settings (DB + store)

**Files:**
- Modify: `backend/src/pos/db.py` (append to `_SCHEMA` before the closing `"""`), `backend/src/pos/auth/store.py` (append methods)
- Test: `backend/tests/test_auth_store.py` (append)

**Interfaces:**
- Produces: `UserStore.get_tool_settings(user_id: str) -> dict[str, bool]`, `UserStore.set_tool_enabled(user_id: str, tool_id: str, enabled: bool) -> None`.

- [ ] **Step 1: Failing test** (append to `tests/test_auth_store.py`, using that file's existing store/user fixtures — copy the exact fixture usage from a neighbouring test such as the credential tests in the same file)

```python
def test_tool_settings_default_empty_and_upsert_per_user(store):
    a = store.create_user("ta@test.com", None, name="A", username="user_ta")
    b = store.create_user("tb@test.com", None, name="B", username="user_tb")

    assert store.get_tool_settings(a["id"]) == {}

    store.set_tool_enabled(a["id"], "web_search", False)
    store.set_tool_enabled(a["id"], "web_search", True)      # upsert, not a duplicate row
    store.set_tool_enabled(a["id"], "get_current_time", False)

    assert store.get_tool_settings(a["id"]) == {"web_search": True, "get_current_time": False}
    assert store.get_tool_settings(b["id"]) == {}            # per user
```

- [ ] **Step 2: Run** `uv run pytest tests/test_auth_store.py -q -k tool_settings` → FAIL (`AttributeError`).

- [ ] **Step 3: Implement.** Schema:

```sql
CREATE TABLE IF NOT EXISTS user_tool_settings (
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    tool_id TEXT NOT NULL,
    enabled BOOLEAN NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, tool_id)
);
```
Store methods (same style as `save_credential`):

```python
    def get_tool_settings(self, user_id: str) -> dict[str, bool]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT tool_id, enabled FROM user_tool_settings WHERE user_id = %s", (user_id,)
            ).fetchall()
        return {r["tool_id"]: r["enabled"] for r in rows}

    def set_tool_enabled(self, user_id: str, tool_id: str, enabled: bool) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO user_tool_settings (user_id, tool_id, enabled) VALUES (%s, %s, %s) "
                "ON CONFLICT (user_id, tool_id) DO UPDATE SET enabled = EXCLUDED.enabled, updated_at = now()",
                (user_id, tool_id, enabled),
            )
```
Also add `user_tool_settings` to the `TRUNCATE` lists in `tests/test_auth_routes.py::_client` and `tests/test_server.py::_fresh_stores` (they truncate `users ... CASCADE`, which already cascades — verify, no change needed if so).

- [ ] **Step 4: Run** the test → PASS. **Step 5: Commit** `feat: per-user tool settings table`.

---

### Task 3: Tools API and registry-driven credential fields

**Files:**
- Modify: `backend/src/pos/auth/routes.py` (`PROVIDER_FIELDS`, new routes)
- Test: `backend/tests/test_auth_routes.py` (append)

**Interfaces:**
- Consumes: Task 1 `all_tools`, `get_tool`, `credential_fields`, `resolve_enabled`; Task 2 store methods.
- Produces: `GET /tools -> {"tools": [{"id","label","description","requires_key","credential_provider","credential_fields":[str],"configured":bool,"enabled":bool,"active":bool}]}`; `PUT /tools/{tool_id}` body `{"enabled": bool}` → `{"id","enabled"}`; 404 unknown id; 409 `"add its API key first"` when enabling a key-requiring tool that is not configured.

- [ ] **Step 1: Failing tests** (append; `_signup` and `_client` already exist in the file)

```python
def test_tools_list_reports_key_requirement_and_configured_state():
    client = _client()
    _signup(client, "a@test.com")

    tools = {t["id"]: t for t in client.get("/tools").json()["tools"]}

    assert tools["get_current_time"]["requires_key"] is False
    assert tools["get_current_time"]["active"] is True
    ws = tools["web_search"]
    assert ws["requires_key"] is True and ws["credential_provider"] == "tavily"
    assert ws["credential_fields"] == ["tavily_api_key"]
    assert ws["configured"] is False and ws["active"] is False

    client.put("/credentials/tavily", json={"tavily_api_key": "tvly-x"})
    ws = {t["id"]: t for t in client.get("/tools").json()["tools"]}["web_search"]
    assert ws["configured"] is True and ws["active"] is True


def test_a_tool_can_be_switched_off_and_on_per_user():
    client = _client()
    _signup(client, "a@test.com")

    assert client.put("/tools/get_current_time", json={"enabled": False}).json() == {
        "id": "get_current_time", "enabled": False}
    assert {t["id"]: t for t in client.get("/tools").json()["tools"]}["get_current_time"]["active"] is False

    client.put("/tools/get_current_time", json={"enabled": True})
    assert {t["id"]: t for t in client.get("/tools").json()["tools"]}["get_current_time"]["active"] is True


def test_enabling_a_key_tool_without_its_key_is_refused_and_unknown_tools_404():
    client = _client()
    _signup(client, "a@test.com")

    assert client.put("/tools/web_search", json={"enabled": True}).status_code == 409
    assert client.put("/tools/nope", json={"enabled": True}).status_code == 404
    assert client.put("/tools/web_search", json={"enabled": False}).status_code == 200   # disabling always ok


def test_tools_routes_require_auth():
    client = _client()
    assert client.get("/tools").status_code == 401
    assert client.put("/tools/web_search", json={"enabled": True}).status_code == 401
```

- [ ] **Step 2: Run** → FAIL (404s). 

- [ ] **Step 3: Implement.** In `routes.py`: rename the current LLM entries to `_LLM_PROVIDER_FIELDS` (delete the `"tavily"` entry) and set
`PROVIDER_FIELDS = {**_LLM_PROVIDER_FIELDS, **tool_credential_fields()}` (import `credential_fields as tool_credential_fields`, `all_tools`, `get_tool`, `resolve_enabled` from `pos.tools`; `import os`). Add inside `build_auth_router` (after the credentials routes):

```python
    def _tool_env(user_id: str, spec) -> dict:
        creds = users.get_credential(user_id, spec.credential_provider) if spec.credential_provider else None
        return {**os.environ, **(creds or {})}

    @router.get("/tools")
    async def list_tools(user_id: str = Depends(require_user_id)):
        settings = users.get_tool_settings(user_id)
        out = []
        for spec in all_tools():
            configured = spec.available(_tool_env(user_id, spec))
            wanted = settings.get(spec.id, spec.default_enabled)
            out.append({
                "id": spec.id, "label": spec.label, "description": spec.description,
                "requires_key": spec.requires_key, "credential_provider": spec.credential_provider,
                "credential_fields": list(spec.credential_fields), "configured": configured,
                "enabled": wanted, "active": wanted and configured,
            })
        return {"tools": out}

    @router.put("/tools/{tool_id}")
    async def set_tool(tool_id: str, body: dict, user_id: str = Depends(require_user_id)):
        spec = get_tool(tool_id)
        if spec is None:
            raise HTTPException(status_code=404, detail=f"unknown tool {tool_id!r}")
        enabled = bool(body.get("enabled"))
        if enabled and not spec.available(_tool_env(user_id, spec)):
            raise HTTPException(status_code=409, detail="add its API key first")
        users.set_tool_enabled(user_id, tool_id, enabled)
        return {"id": tool_id, "enabled": enabled}
```
(`resolve_enabled` import is not needed here; drop it.) The existing `PUT /credentials/tavily` keeps working because `credential_fields()` supplies it; keep `tests` that use `"tavily"`.

- [ ] **Step 4: Run** the four tests + the whole `test_auth_routes.py` → PASS. **Step 5: Commit** `feat: /tools API, credential fields driven by the tool registry`.

---

### Task 4: Server binds each user's tools; per-user cloud LLMs skip warm-up

**Files:**
- Modify: `backend/src/pos/cli/server.py` (`create_app`: factories, `/options`, `/ws` LLM resolution)
- Test: `backend/tests/test_server.py`

**Interfaces:**
- Consumes: `resolve_enabled`, `tool_status`, `UserStore.get_tool_settings`.
- Produces: `llm_env_factory(model: str | None, env: Mapping[str,str], enabled_tools: list[str]) -> LlmBase`; helper `_resolve_user_llm(user_id, provider, model) -> LlmBase` (raises `_NoLlmConfigured`) reused by Task 8.

- [ ] **Step 1: Failing tests** (append to `test_server.py`; reuse `_fresh_stores`, `_sign_in`, `FakeStt/FakeTts/FakeVad/FakeLlm`)

```python
def test_ws_binds_exactly_the_tools_the_user_enabled(monkeypatch):
    captured = {}

    def spy(model, env, enabled_tools):
        captured["enabled"] = list(enabled_tools)
        return FakeLlm("hi")

    session_store, user_store = _fresh_stores()
    app = create_app(stt=FakeStt("x"), tts_engines={"kokoro": FakeTts},
                     llm_factory=lambda m: FakeLlm(), llm_env_factory=spy,
                     vad_factory=lambda **kw: FakeVad(1, 3), default_tts_engine="kokoro",
                     session_store=session_store, user_store=user_store)
    client = TestClient(app)
    user_id = _sign_in(client)
    user_store.save_credential(user_id, "tavily", {"TAVILY_API_KEY": "k"})
    user_store.set_tool_enabled(user_id, "get_current_time", False)

    with client.websocket_connect("/ws?mode=voice") as ws:
        ws.receive_json()

    assert captured["enabled"] == ["web_search"]


def test_real_llm_env_factory_never_warms_up(monkeypatch):
    built = []
    monkeypatch.setattr("pos.llm.LangChainLlm", lambda **kw: built.append(kw) or FakeLlm())
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    session_store, user_store = _fresh_stores()
    app = create_app(stt=FakeStt("x"), tts_engines={"kokoro": FakeTts}, default_tts_engine="kokoro",
                     session_store=session_store, user_store=user_store)
    client = TestClient(app)
    uid = _sign_in(client)
    user_store.save_credential(uid, "local", {"LOCAL_BASE_URL": "http://h/v1", "LOCAL_MODEL": "m"})

    with client.websocket_connect("/ws?mode=voice") as ws:
        ws.receive_json()

    assert built and built[0]["warmup"] is False
```
Also update the existing spies in `test_server.py` (`spy_llm_env_factory(model, env)`) to accept `enabled_tools`, and the existing `test_ws_credentials_are_loaded_via_llm_env_factory` to keep asserting model None / env contents.

- [ ] **Step 2: Run** → FAIL. 

- [ ] **Step 3: Implement.** Real factories in `create_app`:

```python
        if llm_env_factory is None:
            llm_env_factory = lambda model, env, enabled: LangChainLlm(  # noqa: E731
                model=model, env=env, enabled_tools=enabled, warmup=False
            )
```
(warm-up is pointless for a per-user cloud LLM: it costs a paid call and blocked one connect for 89s). Extract from the `/ws` handler a closure:

```python
    class _NoLlmConfigured(Exception):
        pass

    async def _resolve_user_llm(user_id: str, provider: str, model: str | None) -> LlmBase:
        loop = asyncio.get_running_loop()
        stored = users.get_credential(user_id, provider)
        tool_env: dict[str, str] = {}
        for spec in all_tools():                         # from pos.tools
            if spec.credential_provider:
                tool_env.update(users.get_credential(user_id, spec.credential_provider) or {})
        overrides = {**(stored or {}), **tool_env}
        tool_settings = users.get_tool_settings(user_id)
        merged = {**os.environ, **overrides}
        if not _llm_configured({**os.environ, **(stored or {})}):
            raise _NoLlmConfigured
        if overrides or tool_settings:
            enabled = resolve_enabled(tool_settings, merged)
            return await loop.run_in_executor(None, llm_env_factory, model, merged, enabled)
        return await get_llm(model)
```
and make `/ws` call it (`try/except _NoLlmConfigured` → the existing "no LLM configured" error event + close 1008; remove the old inline block). Note `_llm_configured` must be evaluated on os.environ + the *LLM* credential only (tool keys must not count as an LLM). `/options`: `"tools": tool_status()`.

- [ ] **Step 4: Run** full `test_server.py` → PASS; then whole suite. **Step 5: Commit** `feat: bind each user's selected tools; skip warm-up for per-user LLMs`.

---

### Task 5: Separate Tools UI

**Files:**
- Create: `frontend/src/lib/tools.ts`, `frontend/src/hooks/useTools.ts`, `frontend/src/components/ToolsPanel.tsx`
- Modify: `frontend/src/app/page.tsx` (open/close state, render), the sidebar component that hosts the Settings button (find with `grep -n "onOpenSettings\|Settings" frontend/src/components/Sidebar.tsx`), `frontend/src/components/SettingsPanel.tsx` (remove Tavily block at the bottom and the `saveTavilyKey` import), `frontend/src/lib/credentials.ts` (delete `saveTavilyKey`, the `tavily` payload entry), `frontend/src/lib/types.ts` (remove `tavilyApiKey`), `frontend/src/lib/llm.ts`, `frontend/src/components/ConnectionsDiagram.tsx`

**Interfaces:**
- Consumes: Task 3 API shapes.
- Produces: `fetchTools(): Promise<Tool[]>`, `setToolEnabled(id, enabled)`, `saveToolCredential(provider, fields: Record<string,string>)`, `useTools(version): {tools: Tool[]; loading; error; reload()}`; `Tool = {id,label,description,requires_key,credential_provider:string|null,credential_fields:string[],configured,enabled,active}`.

- [ ] **Step 1: `lib/tools.ts`**

```ts
import { API_URL } from "./config";

export interface Tool {
  id: string; label: string; description: string; requires_key: boolean;
  credential_provider: string | null; credential_fields: string[];
  configured: boolean; enabled: boolean; active: boolean;
}

async function detail(res: Response, fallback: string): Promise<string> {
  try { return (await res.json()).detail || fallback; } catch { return fallback; }
}

export async function fetchTools(): Promise<Tool[]> {
  const res = await fetch(`${API_URL}/tools`, { credentials: "include" });
  if (!res.ok) throw new Error(await detail(res, `Couldn't load tools (HTTP ${res.status}).`));
  return (await res.json()).tools;
}

export async function setToolEnabled(id: string, enabled: boolean): Promise<void> {
  const res = await fetch(`${API_URL}/tools/${id}`, {
    method: "PUT", credentials: "include",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify({ enabled }),
  });
  if (!res.ok) throw new Error(await detail(res, "Couldn't update the tool."));
}

export async function saveToolCredential(provider: string, fields: Record<string, string>): Promise<void> {
  const cleaned = Object.fromEntries(Object.entries(fields).map(([k, v]) => [k, v.trim()]).filter(([, v]) => v));
  if (Object.keys(cleaned).length === 0) throw new Error("Enter the key before saving.");   // no 422 round trip
  const res = await fetch(`${API_URL}/credentials/${provider}`, {
    method: "PUT", credentials: "include",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(cleaned),
  });
  if (!res.ok) throw new Error(await detail(res, "Couldn't save. Try again."));
}
```

- [ ] **Step 2: `hooks/useTools.ts`** — same derive-don't-set pattern as `useProviderModels`:

```ts
import { useEffect, useState } from "react";
import { fetchTools, type Tool } from "@/lib/tools";

export function useTools(version: number) {
  const [result, setResult] = useState<{ version: number; tools: Tool[]; error: string | null } | null>(null);
  useEffect(() => {
    let cancelled = false;
    fetchTools()
      .then((tools) => !cancelled && setResult({ version, tools, error: null }))
      .catch((e: Error) => !cancelled && setResult({ version, tools: [], error: e.message }));
    return () => { cancelled = true; };
  }, [version]);
  const current = result && result.version === version ? result : null;
  return { tools: current?.tools ?? [], loading: current === null, error: current?.error ?? null };
}
```

- [ ] **Step 3: `ToolsPanel.tsx`** — one card per tool: label, description, status chip ("Needs API key" / "On" / "Off"), a toggle (disabled with a hint while `requires_key && !configured`), and for `requires_key` tools one `type="password"` input per `credential_fields` entry (label from the field name with `_` → space, title-cased) with Save/Remove buttons. Save calls `saveToolCredential(tool.credential_provider!, values)` then `onChanged()`; Remove calls the existing `removeCredential(provider)` then `onChanged()`; errors render in a `role="alert"` `<p>`. `onChanged` bumps a version in `page.tsx`, which re-runs `useTools` and `refreshConfigured`. Component props: `{ tools: Tool[]; loading: boolean; error: string|null; onChanged: () => void; onBack: () => void }`. Reuse `selectClass`/`Field`-style Tailwind classes from `SettingsPanel.tsx` (copy the two class strings; do not import from it).

- [ ] **Step 4: Wire into `page.tsx` and the sidebar** — `const [toolsOpen, setToolsOpen] = useState(false)`; a "Tools" button beside Settings in the sidebar that calls `onOpenTools`; render `<ToolsPanel …/>` when `toolsOpen` (mutually exclusive with `settingsOpen`). `const [toolsVersion, setToolsVersion] = useState(0)`; `const toolsState = useTools(toolsVersion)`; `onChanged = () => { setToolsVersion(v => v + 1); refreshConfigured(); }`.

- [ ] **Step 5: Remove Tavily from Settings and fix `hasLlm`.** Delete the Tavily `<Field>`+`SaveRemoveRow` block, `saveTavilyKey`, `tavilyApiKey` from `ApiKeyFields`/`EMPTY_KEY_FIELDS`, and the `tavily` entry in `PROVIDER_PAYLOAD`. In `lib/llm.ts` replace the `p !== "tavily"` test with an allow-list: `const LLM_PROVIDERS = ["local","openai","azure","anthropic","gemini","bedrock","openrouter"]; … configured.some((p) => LLM_PROVIDERS.includes(p))`. `ConnectionsDiagram` takes `tools: Tool[]` (active ones drawn enabled, others `(off)`), fed from `toolsState.tools` through `SettingsPanel`.

- [ ] **Step 6: Verify** — `cd frontend && npx tsc --noEmit && npx eslint src` clean. **Step 7: Commit** `feat(ui): separate Tools panel; Tavily key moves there`.

---

### Task 6: Fix list-content in the shared LLM generator

**Files:**
- Create: `backend/src/pos/llm/content.py`
- Modify: `backend/src/pos/llm/langchain_llm.py:101,126-127`
- Test: `backend/tests/test_llm_content.py`, `backend/tests/test_langchain_llm.py` (append)

**Interfaces:**
- Produces: `content_text(content: str | list | None) -> str`.

- [ ] **Step 1: Failing tests**

```python
# tests/test_llm_content.py
from pos.llm.content import content_text


def test_string_and_none():
    assert content_text("hi") == "hi"
    assert content_text(None) == ""


def test_text_blocks_are_joined_and_non_text_blocks_ignored():
    blocks = [{"type": "text", "text": "Hel"}, {"type": "thinking", "thinking": "x"},
              {"type": "text", "text": "lo"}, "!"]
    assert content_text(blocks) == "Hello!"
```
and in `test_langchain_llm.py`:

```python
def test_stream_yields_text_from_block_list_content(monkeypatch):
    chunks = [AIMessageChunk(content=[{"type": "text", "text": "hel"}]),
              AIMessageChunk(content=[{"type": "text", "text": "lo"}])]
    llm = _make_llm(monkeypatch, _FakeRunnable([chunks]), tools=[])
    assert list(llm.stream([{"role": "user", "content": "hi"}], threading.Event())) == ["hel", "lo"]


def test_generate_title_handles_block_list_content(monkeypatch):
    llm = _make_llm(monkeypatch, _FakeRunnable([]), tools=[])
    llm._model.invoke.return_value = AIMessageChunk(content=[{"type": "text", "text": '"Trip planning"'}])
    assert llm.generate_title("a", "b") == "Trip planning"
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement**

```python
# llm/content.py
"""Provider replies carry `content` as a str or, for Gemini/Anthropic with tools or
thinking, a list of content blocks. Everything downstream wants plain text."""

from __future__ import annotations


def content_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text") or "")
    return "".join(parts)
```
In `langchain_llm.py`: `text = content_text(chunk.content); if text: yield text` (replacing lines 126-127) and `title = content_text(response.content).strip().strip('"').strip("'")` (line 101); import `content_text`.

- [ ] **Step 4: Run** suite → PASS. **Step 5: Commit** `fix: read text out of block-list message content`.

---

### Task 7: Shared one-turn helper used by voice and text

**Files:**
- Create: `backend/src/pos/turn.py`
- Modify: `backend/src/pos/agent.py` (`respond`, lines ~403-457)
- Test: `backend/tests/test_turn.py`; existing `tests/test_agent.py` must stay green

**Interfaces:**
- Produces: `TurnStats(text: str, usage: dict, ttft: float | None, total: float | None, stopped: bool)`; `run_turn(llm, messages, cancel, on_piece) -> TurnStats` where `on_piece(piece: str) -> bool | None` returning `False` stops the turn (`stopped=True`).

- [ ] **Step 1: Failing test**

```python
import threading

from pos.turn import run_turn


class _Llm:
    def __init__(self, pieces, usage=None):
        self.pieces, self.usage = pieces, usage or {}

    def stream(self, messages, cancel, usage=None):
        for p in self.pieces:
            yield p
        if usage is not None:
            usage.update(self.usage)


def test_collects_text_usage_and_timing():
    got = []
    stats = run_turn(_Llm(["a", "b"], {"total_tokens": 3}), [], threading.Event(), got.append)
    assert got == ["a", "b"] and stats.text == "ab"
    assert stats.usage == {"total_tokens": 3} and stats.stopped is False
    assert stats.ttft is not None and stats.total >= stats.ttft


def test_on_piece_false_stops_early():
    stats = run_turn(_Llm(["a", "b", "c"]), [], threading.Event(), lambda p: p != "b")
    assert stats.stopped is True and stats.text == "ab"


def test_no_pieces_means_no_ttft():
    stats = run_turn(_Llm([]), [], threading.Event(), lambda p: None)
    assert stats.ttft is None and stats.total is None and stats.text == ""
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement**

```python
"""One LLM turn: stream text pieces, time them, collect usage. Shared by voice
(Agent.respond, which sentence-chunks pieces into TTS) and text (POST /chat/stream,
which forwards them as SSE) so both run the identical LlmBase.stream() path."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from pos.interfaces import LlmBase


@dataclass
class TurnStats:
    text: str
    usage: dict
    ttft: float | None
    total: float | None
    stopped: bool


def run_turn(
    llm: LlmBase,
    messages: list[dict],
    cancel: threading.Event,
    on_piece: Callable[[str], bool | None],
) -> TurnStats:
    usage: dict = {}
    parts: list[str] = []
    ttft: float | None = None
    stopped = False
    t0 = time.perf_counter()
    for piece in llm.stream(messages, cancel, usage):
        if ttft is None:
            ttft = time.perf_counter() - t0
        parts.append(piece)
        if on_piece(piece) is False:
            stopped = True
            break
    total = time.perf_counter() - t0 if ttft is not None else None
    return TurnStats("".join(parts), usage, ttft, total, stopped)
```
Refactor `Agent.respond`: replace the `try: for piece in self.llm.stream(...)` block with

```python
        def on_piece(piece: str) -> bool:
            nonlocal buf
            if self.cancel.is_set() or turn != self.current_turn():
                return False
            full_response.append(piece)
            buf += piece
            drained = drain(buf)
            if drained is None:
                return False
            buf = drained
            return True

        try:
            stats = run_turn(self.llm, messages, self.cancel, on_piece)
        except Exception as e:
            print(f"   [llm failed: {e}]")
            return
        if stats.stopped:
            return
        ttft, usage = stats.ttft, stats.usage
```
and compute `latency` from `stats.total` (`{"ttft": round(ttft,3), "total": round(stats.total,3)}`); keep every later line (metrics, conversation append, `bot_text`) unchanged. Remove the now-unused `t_start`.

- [ ] **Step 4: Run** `tests/test_turn.py tests/test_agent.py` and the full suite → PASS. **Step 5: Commit** `refactor: one run_turn() shared by voice and text`.

---

### Task 8: `POST /chat/stream` (SSE)

**Files:**
- Modify: `backend/src/pos/cli/server.py` (new route inside `create_app`, plus module-level `_sse()` helper)
- Test: `backend/tests/test_chat_stream.py`

**Interfaces:**
- Consumes: `run_turn`, `_resolve_user_llm`, `_NoLlmConfigured` (Task 4), `SessionStore`.
- Produces: `POST /chat/stream` body `{"message": str, "session_id": str|None, "provider": str|None, "llm_model": str|None}`. Pre-stream errors are plain HTTP: 401 (auth), 404 unknown session, 409 no LLM configured, 422 blank message. Stream events (`text/event-stream`): `session {"id"}`, `token {"text"}`, `ping {}` every 15 s of silence, `done {"text","usage"?,"latency"?}`, `title {"title"}`, `error {"message"}` (mid-stream failure; no assistant turn saved).

- [ ] **Step 1: Failing tests** (`tests/test_chat_stream.py`; reuse helpers by importing from `tests/test_server.py`? No — copy `_fresh_stores`, `_sign_in` bodies into a small local module `tests/chat_helpers.py` first, then import from it in both files, or import from `test_server`; keep it simple: `from test_server import _fresh_stores, _sign_in`)

```python
import json

from fakes import FakeLlm, FakeStt, FakeTts
from starlette.testclient import TestClient
from test_server import _fresh_stores, _sign_in  # noqa: F401  (pytest puts tests/ on sys.path)

from pos.cli.server import create_app


def _events(resp):
    out, event = [], None
    for line in resp.iter_lines():
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: "):
            out.append((event, json.loads(line[6:])))
    return out


def _client(llm):
    s, u = _fresh_stores()
    app = create_app(stt=FakeStt("x"), tts_engines={"kokoro": FakeTts}, llm_factory=lambda m: llm,
                     default_tts_engine="kokoro", session_store=s, user_store=u)
    c = TestClient(app)
    _sign_in(c)
    return c, s


def test_streams_session_tokens_then_done_and_saves_both_turns():
    client, store = _client(FakeLlm(reply="hello there"))
    with client.stream("POST", "/chat/stream", json={"message": "hi"}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = _events(resp)

    names = [n for n, _ in events]
    assert names[0] == "session" and names[-2:] == ["done", "title"] or names[-1] == "done"
    assert "".join(d["text"] for n, d in events if n == "token") == "hello there"
    done = next(d for n, d in events if n == "done")
    assert done["text"] == "hello there"
    session_id = events[0][1]["id"]
    turns = client.get(f"/sessions/{session_id}").json()["turns"]
    assert [t["role"] for t in turns] == ["user", "assistant"]


def test_continues_an_existing_session_with_history():
    llm = FakeLlm(reply="ok")
    client, _ = _client(llm)
    with client.stream("POST", "/chat/stream", json={"message": "one"}) as r:
        sid = _events(r)[0][1]["id"]
    with client.stream("POST", "/chat/stream", json={"message": "two", "session_id": sid}) as r:
        list(r.iter_lines())
    assert llm.calls[-1][-1] == {"role": "user", "content": "two"}
    assert {"role": "user", "content": "one"} in llm.calls[-1]


def test_pre_stream_errors_use_http_statuses():
    client, _ = _client(FakeLlm())
    assert client.post("/chat/stream", json={"message": "  "}).status_code == 422
    assert client.post("/chat/stream", json={"message": "hi", "session_id": "nope"}).status_code == 404
    assert TestClient(client.app).post("/chat/stream", json={"message": "hi"}).status_code == 401


def test_no_llm_configured_is_409(monkeypatch):
    from test_server import _PROVIDER_ENV
    for n in _PROVIDER_ENV:
        monkeypatch.delenv(n, raising=False)
    s, u = _fresh_stores()
    app = create_app(stt=FakeStt("x"), tts_engines={"kokoro": FakeTts}, default_tts_engine="kokoro",
                     session_store=s, user_store=u)          # real LLM path, nothing configured
    c = TestClient(app)
    _sign_in(c)
    assert c.post("/chat/stream", json={"message": "hi"}).status_code == 409


def test_a_failing_llm_emits_an_error_event_and_saves_no_assistant_turn():
    class Boom(FakeLlm):
        def stream(self, messages, cancel, usage=None):
            raise RuntimeError("provider exploded")
            yield  # pragma: no cover

    client, _ = _client(Boom())
    with client.stream("POST", "/chat/stream", json={"message": "hi"}) as resp:
        events = _events(resp)
    assert events[-1][0] == "error" and "provider exploded" in events[-1][1]["message"]
    sid = events[0][1]["id"]
    assert [t["role"] for t in client.get(f"/sessions/{sid}").json()["turns"]] == ["user"]
```
(`FakeLlm.calls` records message lists — confirm in `tests/fakes.py` and adapt the assertion to its actual shape.)

- [ ] **Step 2: Run** → FAIL (404). **Step 3: Implement** in `server.py`:

```python
def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
```
```python
    class ChatBody(BaseModel):
        message: str
        session_id: str | None = None
        provider: str | None = None
        llm_model: str | None = None

    @app.post("/chat/stream")
    async def chat_stream(body: ChatBody, user_id: str = Depends(require_user_id)):
        text = body.message.strip()
        if not text:
            raise HTTPException(status_code=422, detail="message is empty")
        history: list[dict] = []
        if body.session_id:
            existing = store.get_session(body.session_id, user_id)
            if existing is None:
                raise HTTPException(status_code=404, detail="session not found")
            history = [{"role": t["role"], "content": t["text"]} for t in existing["turns"]]
        try:
            llm = await _resolve_user_llm(user_id, body.provider or "local", body.llm_model or default_llm_model)
        except _NoLlmConfigured:
            raise HTTPException(status_code=409, detail="no LLM configured -- add a provider in Settings") from None

        loop = asyncio.get_running_loop()
        used_model = getattr(getattr(llm, "provider", None), "model", None) or body.llm_model or ""
        session_id = body.session_id or uuid.uuid4().hex
        new_session = body.session_id is None
        if new_session:
            store.create_session(session_id, user_id, mode="text", tts_engine=None, llm_model=used_model)
        store.add_turn(session_id, "user", text)
        messages = history[-config.HISTORY_TURNS * 2:] + [{"role": "user", "content": text}]

        async def events():
            cancel = threading.Event()
            queue: asyncio.Queue = asyncio.Queue()

            def push(item) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, item)

            def work() -> None:
                try:
                    stats = run_turn(llm, messages, cancel, lambda p: (push(("token", {"text": p})), not cancel.is_set())[1])
                    push(("stats", stats))
                except Exception as e:  # noqa: BLE001 -- surfaced to the client as an error event
                    push(("error", {"message": str(e)[:300] or type(e).__name__}))
                finally:
                    push(None)

            worker = loop.run_in_executor(None, work)
            try:
                yield _sse("session", {"id": session_id})
                while True:
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=15)
                    except TimeoutError:
                        yield _sse("ping", {})
                        continue
                    if item is None:
                        break
                    kind, payload = item
                    if kind == "stats":
                        if payload.text:
                            done = {"text": payload.text}
                            if payload.usage:
                                done["usage"] = payload.usage
                            if payload.ttft is not None:
                                done["latency"] = {"ttft": round(payload.ttft, 3), "total": round(payload.total, 3)}
                            store.add_turn(session_id, "assistant", payload.text, payload.usage or None)
                            yield _sse("done", done)
                            if new_session and hasattr(llm, "generate_title"):
                                title = await loop.run_in_executor(None, llm.generate_title, text, payload.text)
                                if title:
                                    store.set_title(session_id, title)
                                    yield _sse("title", {"title": title})
                    else:
                        yield _sse(kind, payload)
            finally:
                cancel.set()        # client disconnected (or finished): stop paying for tokens
                await worker

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
```
Imports: `threading`, `pydantic.BaseModel`, `fastapi.responses.StreamingResponse`, `pos.turn.run_turn`. Adapt the first test's ordering assertion to whatever `title` produces with `FakeLlm` (it has no `generate_title` unless `tests/fakes.py` defines one — if not, assert `names[-1] == "done"`).

- [ ] **Step 4: Run** `tests/test_chat_stream.py` then the suite → PASS. **Step 5: Commit** `feat: POST /chat/stream (SSE) for text chat`.

---

### Task 9: `/ws` becomes voice-only; drop text-mode plumbing

**Files:**
- Modify: `backend/src/pos/cli/server.py` (`ws_endpoint`, module docstring), `backend/src/pos/agent.py` (remove `text_only`, `on_text_message`, the `if not self.text_only` guards, and the unconditional startup prints only when they reference text mode — keep voice prints), `backend/src/pos/null_engines.py` (keep only what voice still uses; delete the rest and its tests), `backend/src/pos/audio/null_sink.py` (delete if unused)
- Test: `backend/tests/test_server.py`, `tests/test_agent.py`, `tests/test_null_engines.py`

**Interfaces:** `/ws?mode=text` → `{"event":"error","message":"text chat moved to POST /chat/stream"}` then close 1008. `mode` defaults to `voice`.

- [ ] **Step 1: Failing tests** — add

```python
def test_ws_rejects_text_mode_and_points_at_the_http_endpoint(monkeypatch):
    client, _ = _make_client(monkeypatch)
    _sign_in(client)
    with client.websocket_connect("/ws?mode=text") as ws:
        msg = ws.receive_json()
    assert msg["event"] == "error" and "/chat/stream" in msg["message"]
```
and rewrite every existing test that connects with `mode=text` / sends `{"text": …}` (grep `mode=text` and `"text":` in `tests/test_server.py`): tests about titles, sessions, resume and stored turns move to `tests/test_chat_stream.py` semantics (`POST /chat/stream`); tests that only needed a fast handshake switch to `mode=voice` with `FakeVad`/`FakeStt`. Do not delete coverage — port it.

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** In `ws_endpoint`: `mode = params.get("mode", "voice")`; `if mode != "voice": send error (text → the message above; anything else → "mode must be 'voice'") and close 1008`; delete the `if mode == "text": …receive_json… on_text_message` branch, the `NullTts/NullVad/NullAudioSink` conditionals (`tts = await get_tts(...)`, `vad = NullVad() if voice_input_mode == "push_to_talk" else vad_factory(...)` — push-to-talk still needs `NullVad`), `stored_tts_engine = tts_engine`, and update the docstring. `Agent.__init__`: drop `text_only`; `respond`: `self.tts_q.put(...)` unconditionally; delete `on_text_message`. This removes the misleading "barge-in / Loading models" lines from text chat because text chat no longer builds an `Agent` at all.

- [ ] **Step 4: Run** full suite → PASS. **Step 5: Commit** `refactor: /ws is voice-only; text chat lives on /chat/stream`.

---

### Task 10: Frontend text mode streams over SSE

**Files:**
- Create: `frontend/src/lib/chatStream.ts`
- Modify: `frontend/src/hooks/useVoiceSession.ts` (text branch of `connect`/send), `frontend/src/lib/types.ts` if a new state is needed
- Test: manual against Task 17's stack; `tsc`/`eslint`

**Interfaces:**
- Produces: `streamChat(body, handlers, signal): Promise<void>` with `handlers = { onSession(id), onToken(text), onDone(payload), onTitle(title), onError(message) }`.

- [ ] **Step 1: `lib/chatStream.ts`** — `EventSource` cannot POST, so read the response stream:

```ts
import { API_URL } from "./config";

export interface ChatHandlers {
  onSession: (id: string) => void;
  onToken: (text: string) => void;
  onDone: (d: { text: string; usage?: unknown; latency?: unknown }) => void;
  onTitle: (title: string) => void;
  onError: (message: string) => void;
}

export async function streamChat(
  body: { message: string; session_id?: string | null; provider?: string; llm_model?: string },
  h: ChatHandlers,
  signal: AbortSignal
): Promise<void> {
  const res = await fetch(`${API_URL}/chat/stream`, {
    method: "POST", credentials: "include", signal,
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
  });
  if (!res.ok || !res.body) {
    let detail = `Request failed (HTTP ${res.status}).`;
    try { detail = (await res.json()).detail || detail; } catch { /* keep generic */ }
    h.onError(detail);
    return;
  }
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) >= 0) {
      const raw = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const event = /^event: (.+)$/m.exec(raw)?.[1];
      const data = /^data: (.+)$/m.exec(raw)?.[1];
      if (!event || data === undefined) continue;
      const d = JSON.parse(data);
      if (event === "session") h.onSession(d.id);
      else if (event === "token") h.onToken(d.text);
      else if (event === "done") h.onDone(d);
      else if (event === "title") h.onTitle(d.title);
      else if (event === "error") h.onError(d.message);
      // "ping" keepalives carry nothing
    }
  }
}
```

- [ ] **Step 2: Hook integration** — in `useVoiceSession`: for `config.mode === "text"`, `connect` no longer opens a WebSocket: it sets `connected` state, stores `sessionIdRef` (resuming keeps `config.resumeSessionId`), and `sendText(text)` adds the user line, creates an empty bot line, calls `streamChat` with an `AbortController` kept in a ref, appends each `onToken` to that bot line, attaches `usage/latency` on `onDone`, calls `onTitle` → refresh history, and maps `onError` to the existing `reportError`/"system" line. `disconnect()` aborts the controller. Voice keeps the WebSocket path untouched. Reuse the hook's existing line-append helpers (`addLine`); add one `appendToLastBotLine(text)` helper next to them.

- [ ] **Step 3: Verify** `npx tsc --noEmit && npx eslint src` clean. **Step 4: Commit** `feat(ui): text chat streams tokens over SSE`.

---

### Task 11: Surface magic-link mail failures

**Files:** Modify `backend/src/pos/auth/routes.py` (`_send_magic_link`, `signup`, `request_magic_link`); Test `tests/test_auth_routes.py`.

- [ ] **Step 1: Failing test**

```python
def test_signup_without_password_reports_when_the_email_could_not_be_sent(monkeypatch):
    async def boom(email, link_url):
        raise RuntimeError("smtp down")

    monkeypatch.setattr("pos.auth.routes.send_magic_link_email", boom)
    client = _client()
    body = _signup(client, "a@test.com", password=None).json()
    assert body["magic_link_sent"] is False and "email" in body["message"].lower()
```
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** — make `_send_magic_link` return `True` when sent/throttled-away and `False` on a send exception (still log it); `signup` returns `{"magic_link_sent": sent, **({} if sent else {"message": "Your account was created but the sign-in email could not be sent. Try 'email me a link' again in a minute."})}`. `POST /auth/magic-link/request` keeps returning 200 `{"ok": True}` regardless (documented anti-enumeration behaviour), so its test stays.
- [ ] **Step 4: Run** suite. **Step 5: Commit** `fix: tell the user when the sign-in email failed to send`. Also have the signup page show `message` when present (`frontend/src/lib/auth.ts` `SignupResult` + the signup form component).

---

### Task 12: Investigate `GET /sessions → 401`

**Files:** Read `backend/src/pos/auth/tokens.py` (`create_access_token`, TTL), `backend/src/pos/auth/deps.py`, `frontend/src/lib/auth.ts` / `api.ts` (any 401 → `/auth/refresh` retry).

- [ ] **Step 1: Determine** the access-token TTL and whether the frontend retries on 401 by calling `POST /auth/refresh` once and repeating the request. Write the finding in the commit message.
- [ ] **Step 2: If there is no refresh-on-401**, add a single wrapper in `frontend/src/lib/api.ts`: `apiFetch(url, init)` → on 401 call `/auth/refresh` once, retry once, and if still 401 redirect to the login page; route `fetchOptions`, `fetchSessions`, `fetchSession`, `deleteSession`, `fetchConfiguredProviders`, `fetchTools`, `fetchProviderModels` through it. If it already exists, change nothing.
- [ ] **Step 3: Verify** `tsc`/`eslint`. **Step 4: Commit** (or record "no change needed").

---

### Task 13: Neutral URL placeholder

**Files:** Modify `frontend/src/components/SettingsPanel.tsx:~315`.

- [ ] Change `placeholder="http://localhost:1234/v1"` to `placeholder="http://your-server:port/v1"`; `tsc`; commit `chore(ui): neutral local-server placeholder`.

---

### Task 14: Bedrock model listing — verify or mark untested

**Files:** Modify `backend/src/pos/llm/model_listing.py` (`_list_bedrock`); Test `tests/test_model_listing.py`.

- [ ] **Step 1:** Check the installed boto3 exposes `bedrock` client `list_inference_profiles`: `uv run python -c "import boto3;print(hasattr(boto3.client('bedrock',region_name='us-east-1',aws_access_key_id='a',aws_secret_access_key='b'),'list_inference_profiles'))"`.
- [ ] **Step 2:** Add a unit test with a stub boto3 client (monkeypatch `boto3.client`) returning `{"inferenceProfileSummaries":[{"inferenceProfileId":"us.x.y","inferenceProfileName":"Y"}]}` and `{"modelSummaries":[{"modelId":"z","modelName":"Z","inferenceTypesSupported":["ON_DEMAND"]}]}`, asserting both are returned, profiles first, no duplicates; and a `ClientError` maps to `ModelListError("AWS rejected the request (AccessDenied)")`.
- [ ] **Step 3:** State in the module docstring that Bedrock is covered by stubs, not a live AWS account. **Step 4: Commit** `test: cover the Bedrock model lister with stubbed boto3`.

---

### Task 15: List models for server-env provider keys

**Files:** Modify `backend/src/pos/auth/routes.py` (`list_provider_models`); Test `tests/test_auth_routes.py`.

- [ ] **Step 1: Failing test** — with `monkeypatch.setenv("OPENAI_API_KEY", "sk-server")` and no saved credential, `GET /credentials/openai/models` calls `list_models("openai", …)` with the env key and returns 200 (monkeypatch `pos.auth.routes.list_models`), while an unset provider still returns 404.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** — `stored = users.get_credential(...)`; `env = {**os.environ, **(stored or {})}`; if `stored is None` and none of the provider's env vars (`PROVIDER_FIELDS[provider].values()`) are in `os.environ` → 404 as before; otherwise call `list_models(provider, env)`. Never return the key.
- [ ] **Step 4: Run** suite. **Step 5: Commit** `feat: list models for provider keys set on the server`.

---

### Task 16: Backend runs as a non-root user

**Files:** Modify `backend/Dockerfile`, `docker-compose.yml` (volume ownership note); Test: manual.

- [ ] **Step 1:** In `backend/Dockerfile` after `uv sync`, add `RUN useradd --create-home --uid 1000 app && chown -R app:app /app` and `USER app`, and set `ENV HOME=/home/app`. Change the compose mounts to `./models/huggingface:/home/app/.cache/huggingface` and `./models/nltk_data:/home/app/nltk_data`; set `NLTK_DATA=/home/app/nltk_data` and `HF_HOME=/home/app/.cache/huggingface` in the backend environment.
- [ ] **Step 2:** `sudo`-free ownership of the existing root-owned `./models`: run once `docker compose run --rm --user root backend chown -R 1000:1000 /home/app/.cache/huggingface /home/app/nltk_data`.
- [ ] **Step 3: Verify** after rebuild: `docker compose exec backend id -u` prints `1000`, warm-up still `tts warm-up: …s`, and files created in `./models` are owned by the host user. **Step 4: Commit** `chore(docker): run the backend as a non-root user`.

---

### Task 17: Rebuild and live verification

- [ ] **Step 1:** `docker compose build backend frontend && docker compose up -d`; wait for `/docs` = 200 and `tts warm-up` in the log.
- [ ] **Step 2: Gemini 400 fix live:** `docker compose exec -T backend python -c "from pos.llm.model_listing import list_models; …"` with a fake Gemini key → `rejected this key`.
- [ ] **Step 3: SSE with a throwaway user** (`tmp-check@example.com`, deleted by email at the end): sign up with `curl -c`, save a dummy OpenRouter key and a local URL pointing at a stub, then `curl -N -b jar -X POST localhost:8000/chat/stream -H 'content-type: application/json' -d '{"message":"hi"}'` and confirm the event order (`session`, `token…`, `done`) arrives incrementally (timestamps via `ts` or `while read`), that a 401 without cookie, a 422 blank message, and a 409 with no LLM behave, and that killing `curl` mid-stream cancels the model call (backend log shows the turn stop).
- [ ] **Step 4: Tools:** `GET /tools`, save/remove a Tavily key, toggle `get_current_time` off and confirm `resolve_enabled` reflects it on the next connect.
- [ ] **Step 5:** `DELETE FROM users WHERE email='tmp-check@example.com'`. **Step 6:** Record results in the task notes.

---

### Task 18: Browser check

- [ ] Use the `claude-in-chrome` skill if the extension is available: log in at `http://localhost:3000`, open Settings → pick a provider with a saved key → confirm the model dropdown fills; open Tools → toggle, enter a key, save/remove; text chat streams token by token; a bad key shows a readable error. If the extension is unavailable, say so plainly and list exactly what was not visually confirmed.

---

### Task 19: Docs

**Files:** Modify `README.md`.

- [ ] Document: the tools registry and "add a tool = one file + one import"; `/tools` and `/chat/stream` (event list and status codes) in the endpoint section; `/ws` voice-only; non-root container and `./models` ownership; the `pos_test` database and the guard. Remove any remaining text-mode WebSocket wording.

---

### Task 20: Final gate

- [ ] `cd backend && uv run pytest -q -p no:cacheprovider` → all pass; `uv run ruff check src tests` → no *new* findings compared with `git stash`-free baseline (record the count before Task 1 with `git stash`-free `git diff` check: run ruff at `HEAD~N` via `git worktree` if needed).
- [ ] `cd frontend && npx tsc --noEmit && npx eslint src` → clean.
- [ ] `git status --short` clean; `git log --oneline` shows one commit per task.
- [ ] Report: what changed, what was verified live vs. only by tests, and anything not visually confirmed.

---

## Self-Review

- **Spec coverage:** tool registry with one module per tool (T1), user selection persisted (T2) and API (T3), per-user binding (T4), separate UI incl. key forms (T5); SSE streaming that shares `run_turn` with voice (T7, T8, T10) with WS text mode removed (T9); list-content bug (T6); every agenda leftover has a task (T11–T16, T17–T20). The user's "text mode log noise" item is resolved by T9 (text chat no longer constructs an `Agent`).
- **Placeholder scan:** code is given for T1–T8 and T10; T5's component body, T9's test ports, T10's hook edits and T12's conditional edit are described precisely but not spelled out line-by-line because they must adapt to existing code the executor reads first (the plan names the files, helpers and the exact behaviour required).
- **Type consistency:** `resolve_enabled(settings, env)`, `build_tools(enabled, env)`, `llm_env_factory(model, env, enabled_tools)`, `run_turn(llm, messages, cancel, on_piece)`, `TurnStats`, and the `/tools` payload keys (`id,label,description,requires_key,credential_provider,credential_fields,configured,enabled,active`) are used identically in later tasks.

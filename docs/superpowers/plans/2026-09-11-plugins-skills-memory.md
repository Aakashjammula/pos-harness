# Plugins, Skills, and Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three extension mechanisms to the voice/text agent — installable tool Plugins, contextually-loaded Skills, and always-on Memory — plus a tabbed Settings redesign to hold them alongside the existing Model/Input/Provider/Connections sections.

**Architecture:** Plugins and Skills are both discovered from gitignored on-disk directories (`plugins/`, `skills/`) and both ultimately surface as entries in the same tools list `default_tools()` already builds (Skills via one new `load_skill` tool, reusing the existing hand-rolled tool-calling loop in `LangChainLlm.stream()` — no new execution mechanism). Memory is a single gitignored `memory/MEMORY.md` file injected into every turn via one new `extra_system_context` parameter threaded through `LlmBase.stream()` → `Agent.respond()`. The Settings page (`static/index.html`) gets a small tab bar so these three new sections don't pile onto an already-long single scroll.

**Tech Stack:** Python 3.14, FastAPI, `langchain-core` (`BaseTool`), PyYAML (new explicit dependency), vanilla JS/HTML (no framework — matches the rest of `static/index.html`).

**Spec:** `docs/superpowers/specs/2026-09-11-plugins-skills-memory-design.md`

## Global Constraints

- Every new directory (`plugins/`, `skills/`, `memory/`) is gitignored, matching the existing `/data/` pattern for local, generated, non-committed state.
- `discover_plugins()`/equivalent read-only scans (used by `GET /options`) must never execute plugin/skill code — only loading code (`load_enabled_plugin_tools()`/`make_load_skill_tool()`) does, and only when actually building the tools list for a connection.
- No `create_agent`/`deepagents` migration, no graph-based memory, no remote/git plugin installation, no plugin sandboxing — all explicitly out of scope per the spec.
- The full test suite (`uv run pytest tests/ -q`, 240 passing as of this plan) and `uv run ruff check .` must stay green after every task.
- Frontend changes follow this session's established verification: `node --check` on the extracted `<script>` block, a JS-id/HTML-id cross-check script, and (where a server is available) live verification.

---

### Task 1: Settings page tab bar

**Files:**
- Modify: `src/pos/static/index.html`

**Interfaces:**
- Consumes: nothing new.
- Produces: `showSettingsTab(name)` JS function and five named container elements (`settingsPanel-general`, `settingsPanel-providers`, `settingsPanel-plugins`, `settingsPanel-skills`, `settingsPanel-memory`) that later tasks add content into. `settingsPanel-plugins`/`settingsPanel-skills`/`settingsPanel-memory` start as empty `<div>`s (this task only builds the tab shell); Tasks 4, 7, and 11 fill them in.

This task is pure restructuring — every existing field, id, and behavior
(Model/Input/Voice detection fields under `configPanel`, the Provider
list, the seven provider key field-groups, the Tavily field, the
Connections diagram) moves into one of two new tab panels
(`settingsPanel-general`, `settingsPanel-providers`) with **no id
changes and no behavior changes** to anything that already exists.

- [ ] **Step 1: Add the tab bar CSS**

Find this existing rule (added when Settings became a full-screen view):
```css
  .settings-scroll {
    flex: 1;
    overflow-y: auto;
    padding: 8px 24px 40px;
  }
```

Add immediately before it:
```css
  .settings-tabs {
    display: flex;
    gap: 4px;
    padding: 0 24px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  .settings-tab {
    padding: 10px 14px;
    border: none;
    background: transparent;
    font-family: var(--font-ui);
    font-size: 13px;
    font-weight: 500;
    color: var(--text-muted);
    cursor: pointer;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
  }
  .settings-tab:hover { color: var(--text); }
  .settings-tab.active { color: var(--text); border-bottom-color: var(--accent); }
```

- [ ] **Step 2: Insert the tab bar and split the settings-scroll into panels**

Find:
```html
      <div class="settings-view" id="settingsView" hidden>
        <div class="settings-topbar">
          <button type="button" class="settings-back" id="settingsBackBtn">&larr; Back</button>
          <div class="settings-heading">Settings</div>
        </div>
        <div class="settings-scroll">
          <div class="settings-body" id="configPanel">
```

Replace with:
```html
      <div class="settings-view" id="settingsView" hidden>
        <div class="settings-topbar">
          <button type="button" class="settings-back" id="settingsBackBtn">&larr; Back</button>
          <div class="settings-heading">Settings</div>
        </div>
        <div class="settings-tabs" role="tablist">
          <button type="button" class="settings-tab active" data-tab="general" role="tab">General</button>
          <button type="button" class="settings-tab" data-tab="providers" role="tab">Providers</button>
          <button type="button" class="settings-tab" data-tab="plugins" role="tab">Plugins</button>
          <button type="button" class="settings-tab" data-tab="skills" role="tab">Skills</button>
          <button type="button" class="settings-tab" data-tab="memory" role="tab">Memory</button>
        </div>
        <div class="settings-scroll">
          <div id="settingsPanel-general">
          <div class="settings-body" id="configPanel">
```

Find the boundary between the Model/Input/Voice-detection fields and the
"API keys" section title:
```html
            </div>
          </div>

          <div class="settings-section-title">API keys</div>
```

Replace with (closes the `general` panel, opens `providers`):
```html
            </div>
          </div>
          </div>

          <div id="settingsPanel-providers" hidden>
          <div class="settings-section-title">API keys</div>
```

Find the end of the Connections section and the closing tags of
`settings-scroll`/`settings-view`:
```html
          <div class="settings-section-title">Connections</div>
          <div class="field-group">
            <div id="connectionsStatus" class="connections-status"></div>
            <div id="connectionsGraph" class="connections-graph"></div>
          </div>
        </div>
      </div>
    </div>
  </div>
```

Replace with (closes `providers`, adds the three new empty tab panels,
then closes `settings-scroll`/`settings-view` exactly as before):
```html
          <div class="settings-section-title">Connections</div>
          <div class="field-group">
            <div id="connectionsStatus" class="connections-status"></div>
            <div id="connectionsGraph" class="connections-graph"></div>
          </div>
          </div>

          <div id="settingsPanel-plugins" hidden></div>
          <div id="settingsPanel-skills" hidden></div>
          <div id="settingsPanel-memory" hidden></div>
        </div>
      </div>
    </div>
  </div>
```

- [ ] **Step 3: Add the tab-switching JS**

Find (near the other top-level element captures, e.g. right after the
`connectionsStatus`/`connectionsGraph` captures):
```js
  const connectionsStatus = el("connectionsStatus"), connectionsGraph = el("connectionsGraph");
```

Add immediately after it:
```js
  const settingsTabs = Array.from(document.querySelectorAll(".settings-tab"));
  const settingsPanels = {
    general: el("settingsPanel-general"),
    providers: el("settingsPanel-providers"),
    plugins: el("settingsPanel-plugins"),
    skills: el("settingsPanel-skills"),
    memory: el("settingsPanel-memory"),
  };

  function showSettingsTab(name) {
    for (const tab of settingsTabs) tab.classList.toggle("active", tab.dataset.tab === name);
    for (const [key, panel] of Object.entries(settingsPanels)) panel.hidden = key !== name;
  }

  settingsTabs.forEach((tab) => tab.addEventListener("click", () => showSettingsTab(tab.dataset.tab)));
  showSettingsTab("general");
```

- [ ] **Step 4: Verify JS syntax and id cross-check**

Run:
```bash
python3 -c "
import re
content = open('src/pos/static/index.html', encoding='utf-8').read()
m = re.search(r'<script>(.*)</script>', content, re.S)
open('scratch_extracted.js', 'w', encoding='utf-8').write(m.group(1))
"
node --check scratch_extracted.js && echo "SYNTAX OK"
rm scratch_extracted.js
python3 -c "
import re
content = open('src/pos/static/index.html', encoding='utf-8').read()
js_ids = set(re.findall(r'el\(\"([a-zA-Z0-9_]+)\"\)', content))
html_ids = set(re.findall(r'id=\"([a-zA-Z0-9_]+)\"', content))
print('JS refs missing from HTML:', js_ids - html_ids or 'none')
print('HTML ids not referenced in JS:', html_ids - js_ids or 'none')
"
```
Expected: `SYNTAX OK`, both diffs `none`.

- [ ] **Step 5: Run the full backend suite (sanity check, no backend touched)**

Run: `uv run pytest tests/ -q`
Expected: PASS (240 passed) — this task is frontend-only.

- [ ] **Step 6: Commit**

```bash
git add src/pos/static/index.html
git commit -m "feat: tabbed Settings page (General/Providers/Plugins/Skills/Memory)"
```

---

### Task 2: `llm/plugins.py` — discovery and enable/disable

**Files:**
- Create: `src/pos/llm/plugins.py`
- Test: `tests/test_plugins.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: nothing new.
- Produces: `discover_plugins(plugins_dir=PLUGINS_DIR) -> list[dict]` (each dict is the plugin's `plugin.json` content plus a `"_dir"` key), `set_enabled(name, enabled, plugins_dir=PLUGINS_DIR) -> None`, `is_enabled(name, plugins_dir=PLUGINS_DIR) -> bool`, `load_enabled_plugin_tools(plugins_dir=PLUGINS_DIR) -> list[BaseTool]`. Task 3 imports `discover_plugins`, `is_enabled`, `set_enabled`, `load_enabled_plugin_tools`. `plugins_dir` is a keyword parameter (defaulting to the real `plugins/` directory) purely so tests can point every function at a `tmp_path` instead — production call sites never pass it.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_plugins.py`:

```python
import json

from pos.llm import plugins


def _write_plugin(plugins_dir, name, description="A test plugin.", entry_point="tool:get_tools", body=None):
    plugin_dir = plugins_dir / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.json").write_text(json.dumps({
        "name": name, "version": "0.1.0", "description": description, "entry_point": entry_point,
    }))
    if body is not None:
        (plugin_dir / "tool.py").write_text(body)
    return plugin_dir


def test_discover_plugins_returns_empty_list_when_directory_missing(tmp_path):
    assert plugins.discover_plugins(plugins_dir=tmp_path / "nonexistent") == []


def test_discover_plugins_finds_a_plugin_manifest(tmp_path):
    _write_plugin(tmp_path, "weather", description="Look up weather.")

    found = plugins.discover_plugins(plugins_dir=tmp_path)

    assert len(found) == 1
    assert found[0]["name"] == "weather"
    assert found[0]["description"] == "Look up weather."
    assert found[0]["_dir"] == str(tmp_path / "weather")


def test_discover_plugins_does_not_execute_plugin_code(tmp_path):
    # tool.py deliberately raises on import -- discover_plugins() must
    # never import it, only read plugin.json.
    _write_plugin(tmp_path, "broken", body="raise RuntimeError('should never run')")

    found = plugins.discover_plugins(plugins_dir=tmp_path)

    assert found[0]["name"] == "broken"  # no exception raised


def test_is_enabled_false_by_default(tmp_path):
    _write_plugin(tmp_path, "weather")

    assert plugins.is_enabled("weather", plugins_dir=tmp_path) is False


def test_set_enabled_true_then_is_enabled_true(tmp_path):
    _write_plugin(tmp_path, "weather")

    plugins.set_enabled("weather", True, plugins_dir=tmp_path)

    assert plugins.is_enabled("weather", plugins_dir=tmp_path) is True


def test_set_enabled_false_removes_it(tmp_path):
    _write_plugin(tmp_path, "weather")
    plugins.set_enabled("weather", True, plugins_dir=tmp_path)

    plugins.set_enabled("weather", False, plugins_dir=tmp_path)

    assert plugins.is_enabled("weather", plugins_dir=tmp_path) is False


def test_set_enabled_persists_across_calls(tmp_path):
    _write_plugin(tmp_path, "weather")
    plugins.set_enabled("weather", True, plugins_dir=tmp_path)

    # A fresh call re-reads enabled.json from disk rather than relying
    # on any in-process cache.
    assert plugins.is_enabled("weather", plugins_dir=tmp_path) is True


def test_load_enabled_plugin_tools_returns_empty_when_none_enabled(tmp_path):
    _write_plugin(tmp_path, "weather", body="""
from langchain_core.tools import tool

@tool
def get_weather(location: str) -> str:
    '''Return current weather for a location.'''
    return f"sunny in {location}"

def get_tools():
    return [get_weather]
""")

    assert plugins.load_enabled_plugin_tools(plugins_dir=tmp_path) == []


def test_load_enabled_plugin_tools_imports_and_calls_entry_point(tmp_path):
    _write_plugin(tmp_path, "weather", body="""
from langchain_core.tools import tool

@tool
def get_weather(location: str) -> str:
    '''Return current weather for a location.'''
    return f"sunny in {location}"

def get_tools():
    return [get_weather]
""")
    plugins.set_enabled("weather", True, plugins_dir=tmp_path)

    tools = plugins.load_enabled_plugin_tools(plugins_dir=tmp_path)

    assert len(tools) == 1
    assert tools[0].name == "get_weather"
    assert tools[0].invoke({"location": "Paris"}) == "sunny in Paris"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_plugins.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pos.llm.plugins'`

- [ ] **Step 3: Write the implementation**

Create `src/pos/llm/plugins.py`:

```python
"""Installable tool bundles -- see
docs/superpowers/specs/2026-09-11-plugins-skills-memory-design.md.
A plugin is a directory with a plugin.json manifest and a Python module
implementing an entry-point callable that returns a list of tools.
discover_plugins() only reads plugin.json (never imports plugin code --
safe to call on every GET /options); load_enabled_plugin_tools() is the
only function that actually executes plugin code, and only for
plugins explicitly enabled via set_enabled()."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from langchain_core.tools import BaseTool

PLUGINS_DIR = Path("plugins")


def discover_plugins(plugins_dir: Path = PLUGINS_DIR) -> list[dict]:
    if not plugins_dir.is_dir():
        return []
    found = []
    for manifest_path in sorted(plugins_dir.glob("*/plugin.json")):
        manifest = json.loads(manifest_path.read_text())
        found.append({**manifest, "_dir": str(manifest_path.parent)})
    return found


def _enabled_names(plugins_dir: Path) -> set[str]:
    enabled_path = plugins_dir / "enabled.json"
    if not enabled_path.exists():
        return set()
    return set(json.loads(enabled_path.read_text()))


def is_enabled(name: str, plugins_dir: Path = PLUGINS_DIR) -> bool:
    return name in _enabled_names(plugins_dir)


def set_enabled(name: str, enabled: bool, plugins_dir: Path = PLUGINS_DIR) -> None:
    names = _enabled_names(plugins_dir)
    if enabled:
        names.add(name)
    else:
        names.discard(name)
    plugins_dir.mkdir(parents=True, exist_ok=True)
    (plugins_dir / "enabled.json").write_text(json.dumps(sorted(names)))


def load_enabled_plugin_tools(plugins_dir: Path = PLUGINS_DIR) -> list[BaseTool]:
    tools: list[BaseTool] = []
    enabled = _enabled_names(plugins_dir)
    for manifest in discover_plugins(plugins_dir):
        if manifest["name"] not in enabled:
            continue
        module_stem, func_name = manifest["entry_point"].split(":")
        module_path = Path(manifest["_dir"]) / f"{module_stem}.py"
        spec = importlib.util.spec_from_file_location(
            f"pos_plugin_{manifest['name']}", module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        tools.extend(getattr(module, func_name)())
    return tools
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_plugins.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Add `plugins/` to `.gitignore`**

Find:
```
# Local session/turn history (SQLite, see src/pos/storage.py) — personal
# conversation data, generated at runtime, one folder so it's easy to ignore/back up
/data/
```

Add immediately after it:
```

# Installable tool plugins a user drops in locally (code + manifests +
# enabled.json) -- see src/pos/llm/plugins.py. Not shipped with the
# repo, not something to commit on someone else's behalf.
/plugins/
```

- [ ] **Step 6: Run the full suite + ruff**

Run: `uv run pytest tests/ -q && uv run ruff check .`
Expected: PASS, all green (249 tests: 240 + 9 new)

- [ ] **Step 7: Commit**

```bash
git add src/pos/llm/plugins.py tests/test_plugins.py .gitignore
git commit -m "feat: plugin discovery/enable/load (src/pos/llm/plugins.py)"
```

---

### Task 3: Wire plugins into `default_tools()`, `GET /options`, and a toggle endpoint

**Files:**
- Modify: `src/pos/llm/tools.py`
- Modify: `src/pos/cli/server.py`
- Test: `tests/test_tools.py`, `tests/test_server.py`

**Interfaces:**
- Consumes: `plugins.load_enabled_plugin_tools()`, `plugins.discover_plugins()`, `plugins.is_enabled()`, `plugins.set_enabled()` from Task 2.
- Produces: `GET /options`'s `"plugins"` field (`[{"name", "description", "enabled"}, ...]`); `POST /plugins/{name}/toggle` (body `{"enabled": bool}`) calling `plugins.set_enabled()`. Task 4 (frontend) consumes both.

- [ ] **Step 1: Write the failing test for `default_tools()`**

Add to `tests/test_tools.py` (after the existing tests):

```python
def test_default_tools_includes_enabled_plugin_tools(tmp_path, monkeypatch):
    plugin_dir = tmp_path / "weather"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.json").write_text(json.dumps({
        "name": "weather", "version": "0.1.0", "description": "d", "entry_point": "tool:get_tools",
    }))
    (plugin_dir / "tool.py").write_text("""
from langchain_core.tools import tool

@tool
def get_weather(location: str) -> str:
    '''Return weather.'''
    return "sunny"

def get_tools():
    return [get_weather]
""")
    from pos.llm import plugins
    plugins.set_enabled("weather", True, plugins_dir=tmp_path)
    monkeypatch.setattr("pos.llm.tools.plugins.PLUGINS_DIR", tmp_path)

    tools = default_tools()

    assert "get_weather" in [t.name for t in tools]
```

Add `import json` to the top of `tests/test_tools.py` if not already present (check first — it currently only has `import os`).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tools.py::test_default_tools_includes_enabled_plugin_tools -v`
Expected: FAIL — `AttributeError: module 'pos.llm.tools' has no attribute 'plugins'`

- [ ] **Step 3: Wire `default_tools()` to include plugin tools**

In `src/pos/llm/tools.py`, add the import:
```python
from . import plugins
```
(place alongside the existing `from langchain_core.tools import BaseTool, tool` import)

Change:
```python
def default_tools(env: Mapping[str, str] | None = None) -> list[BaseTool]:
    """get_current_time always; web search only if TAVILY_API_KEY is set
    — langchain_tavily raises at construction time otherwise, and a
    voice agent shouldn't fail to start just because search isn't
    configured. env defaults to the real process environment; pass an
    overlay (e.g. a per-connection key typed into the browser's
    Settings page) to resolve against that instead -- passed explicitly
    to TavilySearch(tavily_api_key=...) rather than mutating os.environ,
    which would leak across every other concurrent session."""
    if env is None:
        env = os.environ
    tools: list[BaseTool] = [get_current_time]
    tavily_key = env.get("TAVILY_API_KEY")
    if tavily_key:
        from langchain_tavily import TavilySearch

        tools.append(TavilySearch(max_results=3, tavily_api_key=tavily_key))
    else:
        print("  TAVILY_API_KEY not set — web search tool disabled")
    return tools
```
to:
```python
def default_tools(env: Mapping[str, str] | None = None) -> list[BaseTool]:
    """get_current_time always; web search only if TAVILY_API_KEY is set
    — langchain_tavily raises at construction time otherwise, and a
    voice agent shouldn't fail to start just because search isn't
    configured. env defaults to the real process environment; pass an
    overlay (e.g. a per-connection key typed into the browser's
    Settings page) to resolve against that instead -- passed explicitly
    to TavilySearch(tavily_api_key=...) rather than mutating os.environ,
    which would leak across every other concurrent session. Also
    appends any enabled plugins' tools -- see llm/plugins.py."""
    if env is None:
        env = os.environ
    tools: list[BaseTool] = [get_current_time]
    tavily_key = env.get("TAVILY_API_KEY")
    if tavily_key:
        from langchain_tavily import TavilySearch

        tools.append(TavilySearch(max_results=3, tavily_api_key=tavily_key))
    else:
        print("  TAVILY_API_KEY not set — web search tool disabled")
    tools.extend(plugins.load_enabled_plugin_tools())
    return tools
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tools.py -v`
Expected: PASS (all tests in the file, including the new one)

- [ ] **Step 5: Write the failing tests for `GET /options` and the toggle endpoint**

Add to `tests/test_server.py` (near the other `/options` tests):

```python
def test_options_endpoint_includes_plugins_field(monkeypatch, tmp_path):
    # json is already imported at module level in this file
    plugin_dir = tmp_path / "weather"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.json").write_text(json.dumps({
        "name": "weather", "version": "0.1.0", "description": "Look up weather.", "entry_point": "tool:get_tools",
    }))
    monkeypatch.setattr("pos.cli.server.plugins.PLUGINS_DIR", tmp_path)
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm(),
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=SessionStore(":memory:"),
    )
    client = TestClient(app)

    resp = client.get("/options")

    assert resp.status_code == 200
    plugin_entries = resp.json()["plugins"]
    assert plugin_entries == [{"name": "weather", "description": "Look up weather.", "enabled": False}]


def test_toggle_plugin_endpoint_enables_it(monkeypatch, tmp_path):
    plugin_dir = tmp_path / "weather"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.json").write_text(json.dumps({
        "name": "weather", "version": "0.1.0", "description": "d", "entry_point": "tool:get_tools",
    }))
    monkeypatch.setattr("pos.cli.server.plugins.PLUGINS_DIR", tmp_path)
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm(),
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=SessionStore(":memory:"),
    )
    client = TestClient(app)

    resp = client.post("/plugins/weather/toggle", json={"enabled": True})

    assert resp.status_code == 200
    from pos.llm import plugins
    assert plugins.is_enabled("weather", plugins_dir=tmp_path) is True
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `uv run pytest tests/test_server.py -k "plugins_field or toggle_plugin" -v`
Expected: FAIL — `KeyError: 'plugins'` and `404 Not Found` respectively

- [ ] **Step 7: Implement in `server.py`**

Add the import (alongside the existing `from pos.llm.tools import tool_status`):
```python
from pos.llm import plugins
```

Add a module-level request model (same reasoning as `SessionKeysRequest` —
`from __future__ import annotations` stringifies annotations, so a
FastAPI body model must live at module level, not nested in `create_app`):
```python
class TogglePluginRequest(BaseModel):
    enabled: bool
```

In `GET /options`'s handler, change:
```python
        return {
            "tts": {name: cls.list_voices() for name, cls in tts_engines.items()},
            "llm_models": llm_models,
            "defaults": {"tts_engine": default_tts_engine, "llm_model": default_llm_model},
            "provider": {"name": provider.name, "model": provider.model},
            "tools": tool_status(),
        }
```
to:
```python
        return {
            "tts": {name: cls.list_voices() for name, cls in tts_engines.items()},
            "llm_models": llm_models,
            "defaults": {"tts_engine": default_tts_engine, "llm_model": default_llm_model},
            "provider": {"name": provider.name, "model": provider.model},
            "tools": tool_status(),
            "plugins": [
                {"name": p["name"], "description": p["description"], "enabled": plugins.is_enabled(p["name"])}
                for p in plugins.discover_plugins()
            ],
        }
```

Add a new route (near the `/session-keys` route):
```python
    @app.post("/plugins/{name}/toggle")
    async def toggle_plugin(name: str, body: TogglePluginRequest):
        plugins.set_enabled(name, body.enabled)
        return {"name": name, "enabled": body.enabled}
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/test_server.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 9: Run the full suite + ruff**

Run: `uv run pytest tests/ -q && uv run ruff check .`
Expected: PASS, all green

- [ ] **Step 10: Commit**

```bash
git add src/pos/llm/tools.py src/pos/cli/server.py tests/test_tools.py tests/test_server.py
git commit -m "feat: wire plugins into default_tools(), GET /options, POST /plugins/{name}/toggle"
```

---

### Task 4: Settings UI — Plugins tab

**Files:**
- Modify: `src/pos/static/index.html`

**Interfaces:**
- Consumes: `GET /options`'s `"plugins"` field, `POST /plugins/{name}/toggle` from Task 3; `settingsPanels.plugins` (the `#settingsPanel-plugins` div) from Task 1.
- Produces: nothing further downstream.

- [ ] **Step 1: Add the Plugins tab content**

Find:
```html
          <div id="settingsPanel-plugins" hidden></div>
```

Replace with:
```html
          <div id="settingsPanel-plugins" hidden>
            <div class="settings-section-title">Plugins</div>
            <p class="settings-hint">
              Installable tools the assistant can call. Drop a plugin into the
              server's <code>plugins/</code> folder, then enable it here.
            </p>
            <div class="provider-list" id="pluginsList"></div>
            <p class="settings-hint" id="pluginsEmptyHint" hidden>
              No plugins found in the server's <code>plugins/</code> folder.
            </p>
          </div>
```

- [ ] **Step 2: Render the plugin list from `/options` and wire the toggle**

Find `loadOptions()`'s end (right before its closing brace, after the
existing model-chip/provider-default logic that was added in earlier
work):
```js
    llmModelSel.value = options.defaults.llm_model;
    updateModelChip();
    // The Provider picker deliberately does NOT auto-select from
    // options.provider (the server's current default) -- Connect stays
    // blocked (see keyProviderValidationError()) until a provider is
    // explicitly chosen here, even if the server already has a working
    // one configured.
  }
```

Replace with (adds a call to a new `renderPluginsList()` function):
```js
    llmModelSel.value = options.defaults.llm_model;
    updateModelChip();
    // The Provider picker deliberately does NOT auto-select from
    // options.provider (the server's current default) -- Connect stays
    // blocked (see keyProviderValidationError()) until a provider is
    // explicitly chosen here, even if the server already has a working
    // one configured.
    renderPluginsList();
  }

  const pluginsList = el("pluginsList"), pluginsEmptyHint = el("pluginsEmptyHint");

  function renderPluginsList() {
    pluginsList.innerHTML = "";
    pluginsEmptyHint.hidden = options.plugins.length > 0;
    for (const plugin of options.plugins) {
      const row = document.createElement("label");
      row.className = "provider-row";

      const toggle = document.createElement("input");
      toggle.type = "checkbox";
      toggle.checked = plugin.enabled;
      toggle.addEventListener("change", async () => {
        await fetch(`/plugins/${plugin.name}/toggle`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ enabled: toggle.checked }),
        });
      });

      const name = document.createElement("span");
      name.className = "provider-name";
      name.textContent = plugin.name;

      const description = document.createElement("span");
      description.className = "provider-status";
      description.style.cssText = "width:auto;border-radius:0;background:none;color:var(--text-faint);font-size:11.5px;";
      description.textContent = plugin.description;

      row.append(toggle, name, description);
      pluginsList.appendChild(row);
    }
  }
```

- [ ] **Step 3: Verify JS syntax and id cross-check**

Run the same verification script as Task 1, Step 4.
Expected: `SYNTAX OK`, both diffs `none`.

- [ ] **Step 4: Manual verification**

Start the server (`uv run pos-server`), open `http://localhost:8000/`,
click the model chip to open Settings, click the "Plugins" tab. With no
`plugins/` directory present, confirm the empty-state hint shows and no
console errors appear. Create a throwaway `plugins/test-plugin/plugin.json`
(`{"name": "test-plugin", "version": "0.1.0", "description": "test", "entry_point": "tool:get_tools"}`)
and a `tool.py` with a trivial `@tool`-decorated function, reload the
page, confirm the row appears with a working toggle. Delete the
throwaway plugin directory afterward.

- [ ] **Step 5: Run the full backend suite (sanity check)**

Run: `uv run pytest tests/ -q`
Expected: PASS — this task is frontend-only.

- [ ] **Step 6: Commit**

```bash
git add src/pos/static/index.html
git commit -m "feat: Plugins tab in Settings (list + enable/disable toggle)"
```

---

### Task 5: `llm/skills.py` — parsing and the `load_skill` tool

**Files:**
- Create: `src/pos/llm/skills.py`
- Test: `tests/test_skills.py`
- Modify: `pyproject.toml`, `uv.lock` (via `uv add`)
- Modify: `.gitignore`

**Interfaces:**
- Consumes: nothing new.
- Produces: `discover_skills(skills_dir=SKILLS_DIR) -> list[dict]` (each dict: `{"name", "description", "_dir"}, parsed from frontmatter only, does not read the body), `is_skill_enabled(name, skills_dir=SKILLS_DIR) -> bool`, `set_skill_enabled(name, enabled, skills_dir=SKILLS_DIR) -> None`, `make_load_skill_tool(skills_dir=SKILLS_DIR) -> BaseTool | None` (returns `None` when no skill is enabled — Task 6 checks for this). Task 6 imports all four.

- [ ] **Step 1: Add PyYAML as an explicit dependency**

Run: `uv add pyyaml`

(Currently only pulled in transitively by another dependency — this
makes it an explicit, pinned direct dependency instead of relying on
that continuing to be true.)

- [ ] **Step 2: Write the failing tests**

Create `tests/test_skills.py`:

```python
from pos.llm import skills


def _write_skill(skills_dir, name, description="A test skill.", body="Do the thing."):
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n"
    )
    return skill_dir


def test_discover_skills_returns_empty_list_when_directory_missing(tmp_path):
    assert skills.discover_skills(skills_dir=tmp_path / "nonexistent") == []


def test_discover_skills_parses_frontmatter(tmp_path):
    _write_skill(tmp_path, "cooking-helper", description="Helps with recipes.")

    found = skills.discover_skills(skills_dir=tmp_path)

    assert len(found) == 1
    assert found[0]["name"] == "cooking-helper"
    assert found[0]["description"] == "Helps with recipes."


def test_is_skill_enabled_false_by_default(tmp_path):
    _write_skill(tmp_path, "cooking-helper")

    assert skills.is_skill_enabled("cooking-helper", skills_dir=tmp_path) is False


def test_set_skill_enabled_true_then_is_enabled_true(tmp_path):
    _write_skill(tmp_path, "cooking-helper")

    skills.set_skill_enabled("cooking-helper", True, skills_dir=tmp_path)

    assert skills.is_skill_enabled("cooking-helper", skills_dir=tmp_path) is True


def test_set_skill_enabled_false_removes_it(tmp_path):
    _write_skill(tmp_path, "cooking-helper")
    skills.set_skill_enabled("cooking-helper", True, skills_dir=tmp_path)

    skills.set_skill_enabled("cooking-helper", False, skills_dir=tmp_path)

    assert skills.is_skill_enabled("cooking-helper", skills_dir=tmp_path) is False


def test_make_load_skill_tool_returns_none_when_no_skill_enabled(tmp_path):
    _write_skill(tmp_path, "cooking-helper")  # discovered but not enabled

    assert skills.make_load_skill_tool(skills_dir=tmp_path) is None


def test_make_load_skill_tool_description_lists_enabled_skills(tmp_path):
    _write_skill(tmp_path, "cooking-helper", description="Helps with recipes.")
    _write_skill(tmp_path, "trip-planner", description="Plans trips.")
    skills.set_skill_enabled("cooking-helper", True, skills_dir=tmp_path)

    tool = skills.make_load_skill_tool(skills_dir=tmp_path)

    assert "cooking-helper" in tool.description
    assert "Helps with recipes." in tool.description
    assert "trip-planner" not in tool.description  # not enabled


def test_load_skill_tool_returns_the_bodys_content(tmp_path):
    _write_skill(tmp_path, "cooking-helper", body="You are a cooking assistant.")
    skills.set_skill_enabled("cooking-helper", True, skills_dir=tmp_path)

    tool = skills.make_load_skill_tool(skills_dir=tmp_path)

    assert tool.invoke({"name": "cooking-helper"}) == "You are a cooking assistant."


def test_load_skill_tool_returns_a_message_for_unknown_skill(tmp_path):
    _write_skill(tmp_path, "cooking-helper")
    skills.set_skill_enabled("cooking-helper", True, skills_dir=tmp_path)

    tool = skills.make_load_skill_tool(skills_dir=tmp_path)

    assert tool.invoke({"name": "nonexistent"}) == "unknown skill: nonexistent"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_skills.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pos.llm.skills'`

- [ ] **Step 4: Write the implementation**

Create `src/pos/llm/skills.py`:

```python
"""Contextually-loaded instruction sets -- see
docs/superpowers/specs/2026-09-11-plugins-skills-memory-design.md.
A skill is skills/<name>/SKILL.md -- YAML frontmatter (name, description)
plus a Markdown body. Unlike plugins, nothing here executes code: this
is all just reading files. make_load_skill_tool() returns one new tool
whose description lists every *enabled* skill so the model can decide
when a skill is relevant; calling it returns that skill's full body as
the tool result, folded into the conversation the same way any other
tool result is."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from langchain_core.tools import BaseTool, tool

SKILLS_DIR = Path("skills")


def _parse_skill_md(md_path: Path) -> dict:
    raw = md_path.read_text()
    _, frontmatter, body = raw.split("---", 2)
    meta = yaml.safe_load(frontmatter)
    return {"name": meta["name"], "description": meta["description"], "body": body.strip()}


def discover_skills(skills_dir: Path = SKILLS_DIR) -> list[dict]:
    if not skills_dir.is_dir():
        return []
    found = []
    for skill_dir in sorted(skills_dir.glob("*/")):
        md_path = skill_dir / "SKILL.md"
        if not md_path.exists():
            continue
        parsed = _parse_skill_md(md_path)
        found.append({"name": parsed["name"], "description": parsed["description"], "_dir": str(skill_dir)})
    return found


def _enabled_names(skills_dir: Path) -> set[str]:
    enabled_path = skills_dir / "enabled.json"
    if not enabled_path.exists():
        return set()
    return set(json.loads(enabled_path.read_text()))


def is_skill_enabled(name: str, skills_dir: Path = SKILLS_DIR) -> bool:
    return name in _enabled_names(skills_dir)


def set_skill_enabled(name: str, enabled: bool, skills_dir: Path = SKILLS_DIR) -> None:
    names = _enabled_names(skills_dir)
    if enabled:
        names.add(name)
    else:
        names.discard(name)
    skills_dir.mkdir(parents=True, exist_ok=True)
    (skills_dir / "enabled.json").write_text(json.dumps(sorted(names)))


def make_load_skill_tool(skills_dir: Path = SKILLS_DIR) -> BaseTool | None:
    enabled = _enabled_names(skills_dir)
    enabled_skills = [s for s in discover_skills(skills_dir) if s["name"] in enabled]
    if not enabled_skills:
        return None

    bodies = {s["name"]: _parse_skill_md(Path(s["_dir"]) / "SKILL.md")["body"] for s in enabled_skills}
    catalog = "\n".join(f"- {s['name']}: {s['description']}" for s in enabled_skills)

    @tool
    def load_skill(name: str) -> str:
        f"""Load a skill's full instructions by name when its topic
        becomes relevant to the conversation. Available skills:
        {catalog}"""
        return bodies.get(name, f"unknown skill: {name}")

    return load_skill
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_skills.py -v`
Expected: PASS (9 tests)

- [ ] **Step 6: Add `skills/` to `.gitignore`**

Add immediately after the `/plugins/` entry from Task 2:
```

# Contextually-loaded skill instructions (Markdown, see
# src/pos/llm/skills.py) -- same local/personal reasoning as plugins/.
/skills/
```

- [ ] **Step 7: Run the full suite + ruff**

Run: `uv run pytest tests/ -q && uv run ruff check .`
Expected: PASS, all green

- [ ] **Step 8: Commit**

```bash
git add src/pos/llm/skills.py tests/test_skills.py pyproject.toml uv.lock .gitignore
git commit -m "feat: skill discovery/enable + load_skill tool (src/pos/llm/skills.py)"
```

---

### Task 6: Wire skills into `default_tools()`, `GET /options`, and a toggle endpoint

**Files:**
- Modify: `src/pos/llm/tools.py`
- Modify: `src/pos/cli/server.py`
- Test: `tests/test_tools.py`, `tests/test_server.py`

**Interfaces:**
- Consumes: `skills.make_load_skill_tool()`, `skills.discover_skills()`, `skills.is_skill_enabled()`, `skills.set_skill_enabled()` from Task 5.
- Produces: `GET /options`'s `"skills"` field (`[{"name", "description", "enabled"}, ...]`); `POST /skills/{name}/toggle`. Task 7 (frontend) consumes both.

- [ ] **Step 1: Write the failing test for `default_tools()`**

Add to `tests/test_tools.py`:

```python
def test_default_tools_includes_load_skill_when_a_skill_is_enabled(tmp_path, monkeypatch):
    skill_dir = tmp_path / "cooking-helper"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: cooking-helper\ndescription: Helps with recipes.\n---\n\nBe a cooking assistant.\n"
    )
    from pos.llm import skills
    skills.set_skill_enabled("cooking-helper", True, skills_dir=tmp_path)
    monkeypatch.setattr("pos.llm.tools.skills.SKILLS_DIR", tmp_path)

    tools = default_tools()

    assert "load_skill" in [t.name for t in tools]


def test_default_tools_omits_load_skill_when_none_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr("pos.llm.tools.skills.SKILLS_DIR", tmp_path)

    tools = default_tools()

    assert "load_skill" not in [t.name for t in tools]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tools.py -k load_skill -v`
Expected: FAIL — `AttributeError: module 'pos.llm.tools' has no attribute 'skills'`

- [ ] **Step 3: Wire `default_tools()` to include `load_skill`**

In `src/pos/llm/tools.py`, add the import alongside the `plugins`
import from Task 3:
```python
from . import plugins, skills
```

Change the end of `default_tools()`:
```python
    tools.extend(plugins.load_enabled_plugin_tools())
    return tools
```
to:
```python
    tools.extend(plugins.load_enabled_plugin_tools())
    load_skill_tool = skills.make_load_skill_tool()
    if load_skill_tool is not None:
        tools.append(load_skill_tool)
    return tools
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tools.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Write the failing tests for `GET /options` and the toggle endpoint**

Add to `tests/test_server.py`:

```python
def test_options_endpoint_includes_skills_field(monkeypatch, tmp_path):
    skill_dir = tmp_path / "cooking-helper"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: cooking-helper\ndescription: Helps with recipes.\n---\n\nBe a cooking assistant.\n"
    )
    monkeypatch.setattr("pos.cli.server.skills.SKILLS_DIR", tmp_path)
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm(),
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=SessionStore(":memory:"),
    )
    client = TestClient(app)

    resp = client.get("/options")

    assert resp.status_code == 200
    assert resp.json()["skills"] == [{"name": "cooking-helper", "description": "Helps with recipes.", "enabled": False}]


def test_toggle_skill_endpoint_enables_it(monkeypatch, tmp_path):
    skill_dir = tmp_path / "cooking-helper"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: cooking-helper\ndescription: d\n---\n\nBody.\n"
    )
    monkeypatch.setattr("pos.cli.server.skills.SKILLS_DIR", tmp_path)
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm(),
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=SessionStore(":memory:"),
    )
    client = TestClient(app)

    resp = client.post("/skills/cooking-helper/toggle", json={"enabled": True})

    assert resp.status_code == 200
    from pos.llm import skills
    assert skills.is_skill_enabled("cooking-helper", skills_dir=tmp_path) is True
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `uv run pytest tests/test_server.py -k "skills_field or toggle_skill" -v`
Expected: FAIL — `KeyError: 'skills'` and `404 Not Found` respectively

- [ ] **Step 7: Implement in `server.py`**

Add the import alongside the `plugins` import from Task 3:
```python
from pos.llm import plugins, skills
```

Add a second module-level request model, next to `TogglePluginRequest`:
```python
class ToggleSkillRequest(BaseModel):
    enabled: bool
```

In `GET /options`'s handler, add one more field to the returned dict:
```python
            "skills": [
                {"name": s["name"], "description": s["description"], "enabled": skills.is_skill_enabled(s["name"])}
                for s in skills.discover_skills()
            ],
```

Add a new route next to `toggle_plugin`:
```python
    @app.post("/skills/{name}/toggle")
    async def toggle_skill(name: str, body: ToggleSkillRequest):
        skills.set_skill_enabled(name, body.enabled)
        return {"name": name, "enabled": body.enabled}
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/test_server.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 9: Run the full suite + ruff**

Run: `uv run pytest tests/ -q && uv run ruff check .`
Expected: PASS, all green

- [ ] **Step 10: Commit**

```bash
git add src/pos/llm/tools.py src/pos/cli/server.py tests/test_tools.py tests/test_server.py
git commit -m "feat: wire skills into default_tools(), GET /options, POST /skills/{name}/toggle"
```

---

### Task 7: Settings UI — Skills tab

**Files:**
- Modify: `src/pos/static/index.html`

**Interfaces:**
- Consumes: `GET /options`'s `"skills"` field, `POST /skills/{name}/toggle` from Task 6; `settingsPanels.skills` (the `#settingsPanel-skills` div) from Task 1.
- Produces: nothing further downstream.

- [ ] **Step 1: Add the Skills tab content**

Find:
```html
          <div id="settingsPanel-skills" hidden></div>
```

Replace with:
```html
          <div id="settingsPanel-skills" hidden>
            <div class="settings-section-title">Skills</div>
            <p class="settings-hint">
              Contextual instructions the assistant can load mid-conversation
              when relevant. Drop a <code>SKILL.md</code> into the server's
              <code>skills/</code> folder, then enable it here.
            </p>
            <div class="provider-list" id="skillsList"></div>
            <p class="settings-hint" id="skillsEmptyHint" hidden>
              No skills found in the server's <code>skills/</code> folder.
            </p>
          </div>
```

- [ ] **Step 2: Render the skills list from `/options` and wire the toggle**

Find the `renderPluginsList()` function added in Task 4 and add a
sibling function right after it, plus a call to it from `loadOptions()`:

Find:
```js
    renderPluginsList();
  }

  const pluginsList = el("pluginsList"), pluginsEmptyHint = el("pluginsEmptyHint");
```

Replace with:
```js
    renderPluginsList();
    renderSkillsList();
  }

  const pluginsList = el("pluginsList"), pluginsEmptyHint = el("pluginsEmptyHint");
```

Find the end of `renderPluginsList()`'s closing brace:
```js
      row.append(toggle, name, description);
      pluginsList.appendChild(row);
    }
  }
```

Add immediately after it:
```js

  const skillsList = el("skillsList"), skillsEmptyHint = el("skillsEmptyHint");

  function renderSkillsList() {
    skillsList.innerHTML = "";
    skillsEmptyHint.hidden = options.skills.length > 0;
    for (const skill of options.skills) {
      const row = document.createElement("label");
      row.className = "provider-row";

      const toggle = document.createElement("input");
      toggle.type = "checkbox";
      toggle.checked = skill.enabled;
      toggle.addEventListener("change", async () => {
        await fetch(`/skills/${skill.name}/toggle`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ enabled: toggle.checked }),
        });
      });

      const name = document.createElement("span");
      name.className = "provider-name";
      name.textContent = skill.name;

      const description = document.createElement("span");
      description.className = "provider-status";
      description.style.cssText = "width:auto;border-radius:0;background:none;color:var(--text-faint);font-size:11.5px;";
      description.textContent = skill.description;

      row.append(toggle, name, description);
      skillsList.appendChild(row);
    }
  }
```

- [ ] **Step 3: Verify JS syntax and id cross-check**

Run the same verification script as Task 1, Step 4.
Expected: `SYNTAX OK`, both diffs `none`.

- [ ] **Step 4: Manual verification**

Same style as Task 4, Step 4: start the server, open Settings, click
"Skills", confirm the empty state, then drop a throwaway
`skills/test-skill/SKILL.md` (frontmatter + a one-line body), reload,
confirm the row and toggle work, then delete the throwaway skill.

- [ ] **Step 5: Run the full backend suite (sanity check)**

Run: `uv run pytest tests/ -q`
Expected: PASS — this task is frontend-only.

- [ ] **Step 6: Commit**

```bash
git add src/pos/static/index.html
git commit -m "feat: Skills tab in Settings (list + enable/disable toggle)"
```

---

### Task 8: `extra_system_context` on `LlmBase.stream()` / `LangChainLlm.stream()` / `OpenAiCompatibleLlm.stream()`

**Files:**
- Modify: `src/pos/interfaces/llm.py`
- Modify: `src/pos/llm/langchain_llm.py`
- Modify: `src/pos/llm/openai_compatible.py`
- Test: `tests/test_langchain_llm.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `LlmBase.stream(messages, cancel, usage=None, extra_system_context=None)` — the interface every `LlmBase` implementation now satisfies. Task 9 (`Agent.respond()`) calls `self.llm.stream(messages, self.cancel, usage, extra_context)`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_langchain_llm.py` (near the other `stream()`-focused
tests, using the file's existing `_make_llm`/`_FakeRunnable` helpers):

```python
def test_stream_appends_extra_system_context_as_a_second_system_message(monkeypatch):
    runnable = _FakeRunnable([[_text_chunk("hi")]])
    llm = _make_llm(monkeypatch, runnable, tools=[])

    list(llm.stream(
        [{"role": "user", "content": "hi"}], threading.Event(), None,
        extra_system_context="Known facts about the user:\n- prefers short answers",
    ))

    sent_messages = runnable.stream_calls[0]
    system_messages = [m for m in sent_messages if m.__class__.__name__ == "SystemMessage"]
    assert len(system_messages) == 2
    assert "prefers short answers" in system_messages[1].content


def test_stream_without_extra_system_context_sends_only_the_persona_system_message(monkeypatch):
    runnable = _FakeRunnable([[_text_chunk("hi")]])
    llm = _make_llm(monkeypatch, runnable, tools=[])

    list(llm.stream([{"role": "user", "content": "hi"}], threading.Event()))

    sent_messages = runnable.stream_calls[0]
    system_messages = [m for m in sent_messages if m.__class__.__name__ == "SystemMessage"]
    assert len(system_messages) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_langchain_llm.py -k extra_system_context -v`
Expected: FAIL — `TypeError: LangChainLlm.stream() takes from 3 to 4 positional arguments but 5 were given`

- [ ] **Step 3: Update `LlmBase`**

In `src/pos/interfaces/llm.py`, change:
```python
    @abstractmethod
    def stream(
        self, messages: list[dict], cancel: threading.Event, usage: dict | None = None
    ) -> Iterator[str]: ...
```
to:
```python
    @abstractmethod
    def stream(
        self, messages: list[dict], cancel: threading.Event, usage: dict | None = None,
        extra_system_context: str | None = None,
    ) -> Iterator[str]:
        """extra_system_context, if given, is appended as a second
        system message after the engine's own persona system_prompt --
        e.g. always-on user memory (see Agent's memory_facts
        parameter). Unlike `usage`, this is plain input, not an
        output-by-mutation parameter."""
```

- [ ] **Step 4: Update `LangChainLlm.stream()`**

In `src/pos/llm/langchain_llm.py`, change:
```python
    def stream(
        self, messages: list[dict], cancel: threading.Event, usage: dict | None = None
    ) -> Iterator[str]:
        full: list[BaseMessage] = [SystemMessage(self.system_prompt)]
        for m in messages:
```
to:
```python
    def stream(
        self, messages: list[dict], cancel: threading.Event, usage: dict | None = None,
        extra_system_context: str | None = None,
    ) -> Iterator[str]:
        full: list[BaseMessage] = [SystemMessage(self.system_prompt)]
        if extra_system_context:
            full.append(SystemMessage(extra_system_context))
        for m in messages:
```

- [ ] **Step 5: Update `OpenAiCompatibleLlm.stream()` for interface consistency**

In `src/pos/llm/openai_compatible.py`, change:
```python
    def stream(
        self, messages: list[dict], cancel: threading.Event, usage: dict | None = None
    ) -> Iterator[str]:
        full_messages = [{"role": "system", "content": self.system_prompt}] + messages
```
to:
```python
    def stream(
        self, messages: list[dict], cancel: threading.Event, usage: dict | None = None,
        extra_system_context: str | None = None,
    ) -> Iterator[str]:
        full_messages = [{"role": "system", "content": self.system_prompt}]
        if extra_system_context:
            full_messages.append({"role": "system", "content": extra_system_context})
        full_messages += messages
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_langchain_llm.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 7: Run the full suite + ruff**

Run: `uv run pytest tests/ -q && uv run ruff check .`
Expected: PASS, all green

- [ ] **Step 8: Commit**

```bash
git add src/pos/interfaces/llm.py src/pos/llm/langchain_llm.py src/pos/llm/openai_compatible.py tests/test_langchain_llm.py
git commit -m "feat: extra_system_context parameter on LlmBase.stream()"
```

---

### Task 9: `Agent` — thread `memory_facts` into `respond()`

**Files:**
- Modify: `src/pos/agent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `LlmBase.stream(..., extra_system_context=...)` from Task 8.
- Produces: `Agent(..., memory_facts: Callable[[], str | None] | None = None)`. Task 10 (`server.py`) passes a closure reading `memory/MEMORY.md`.

- [ ] **Step 1: Update `FakeLlm.stream()` to accept `extra_system_context`**

This must happen before Step 4 (`respond()` calling `self.llm.stream()`
with 4 positional arguments) or every existing `test_agent.py` test that
uses the default `FakeLlm` (via `_build_agent()`) would break with a
`TypeError` — not just the new tests this task adds.

In `tests/fakes.py`, change:
```python
    def stream(
        self, messages: list[dict], cancel: threading.Event, usage: dict | None = None
    ) -> Iterator[str]:
        self.calls.append(messages)
        if usage is not None and self.fake_usage is not None:
            usage.update(self.fake_usage)
        for word in self.reply.split():
            yield word + " "
```
to:
```python
    def stream(
        self, messages: list[dict], cancel: threading.Event, usage: dict | None = None,
        extra_system_context: str | None = None,
    ) -> Iterator[str]:
        self.calls.append(messages)
        if usage is not None and self.fake_usage is not None:
            usage.update(self.fake_usage)
        for word in self.reply.split():
            yield word + " "
```

Run: `uv run pytest tests/ -q`
Expected: PASS (240) — this change alone is backward-compatible (existing
calls only ever passed 3 positional args), confirming nothing broke
before continuing.

- [ ] **Step 2: Write the failing tests**

Add to `tests/test_agent.py` (using the file's existing `_build_agent`
helper and `FakeLlm`, now updated in Step 1):

```python
def test_respond_passes_memory_facts_to_llm_stream():
    calls = []

    class _SpyLlm(FakeLlm):
        def stream(self, messages, cancel, usage=None, extra_system_context=None):
            calls.append(extra_system_context)
            return super().stream(messages, cancel, usage, extra_system_context)

    agent = _build_agent(llm=_SpyLlm("hi there"), memory_facts=lambda: "prefers short answers")

    agent.respond("hello", agent.new_turn(), stt_t=0.0)

    assert calls == ["prefers short answers"]


def test_respond_passes_none_when_memory_facts_not_given():
    calls = []

    class _SpyLlm(FakeLlm):
        def stream(self, messages, cancel, usage=None, extra_system_context=None):
            calls.append(extra_system_context)
            return super().stream(messages, cancel, usage, extra_system_context)

    agent = _build_agent(llm=_SpyLlm("hi there"))  # no memory_facts

    agent.respond("hello", agent.new_turn(), stt_t=0.0)

    assert calls == [None]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_agent.py -k memory_facts -v`
Expected: FAIL — `TypeError: Agent.__init__() got an unexpected keyword argument 'memory_facts'`

- [ ] **Step 4: Implement in `agent.py`**

In `src/pos/agent.py`, change the constructor signature:
```python
    def __init__(
        self,
        vad: VadBase | None = None,
        stt: SttBase | None = None,
        tts: TtsBase | None = None,
        llm: LlmBase | None = None,
        trigger_word: str | None = None,
        audio_sink: AudioSinkBase | None = None,
        on_event: Callable[[str, dict], None] | None = None,
        text_only: bool = False,
        conversation: list[dict] | None = None,
    ):
```
to:
```python
    def __init__(
        self,
        vad: VadBase | None = None,
        stt: SttBase | None = None,
        tts: TtsBase | None = None,
        llm: LlmBase | None = None,
        trigger_word: str | None = None,
        audio_sink: AudioSinkBase | None = None,
        on_event: Callable[[str, dict], None] | None = None,
        text_only: bool = False,
        conversation: list[dict] | None = None,
        memory_facts: Callable[[], str | None] | None = None,
    ):
```

Add the assignment near the other simple attribute assignments (right
after `self.text_only = text_only` at the end of `__init__`):
```python
        # Always-on user memory (see docs/superpowers/specs/
        # 2026-09-11-plugins-skills-memory-design.md) -- a callback
        # (same pattern as on_event) rather than a raw string, so
        # respond() reads memory/MEMORY.md fresh every turn instead of
        # a value frozen at Agent-construction time. None when not
        # given -- text-mode's construction and every existing caller
        # that doesn't pass it keep working unchanged.
        self.memory_facts = memory_facts
```

In `respond()`, change:
```python
        t_start = time.perf_counter()
        usage: dict = {}
        try:
            for piece in self.llm.stream(messages, self.cancel, usage):
```
to:
```python
        t_start = time.perf_counter()
        usage: dict = {}
        extra_context = self.memory_facts() if self.memory_facts else None
        try:
            for piece in self.llm.stream(messages, self.cancel, usage, extra_context):
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_agent.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 6: Run the full suite + ruff**

Run: `uv run pytest tests/ -q && uv run ruff check .`
Expected: PASS, all green

- [ ] **Step 7: Commit**

```bash
git add src/pos/agent.py tests/test_agent.py tests/fakes.py
git commit -m "feat: Agent.memory_facts callback threaded into respond()'s stream() call"
```

---

### Task 10: `server.py` — `GET/POST /memory` + wiring `memory_facts` into `Agent`

**Files:**
- Modify: `src/pos/cli/server.py`
- Test: `tests/test_server.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `Agent(..., memory_facts=...)` from Task 9.
- Produces: `GET /memory` (`{"content": str}`), `POST /memory` (body `{"content": str}`, writes `memory/MEMORY.md`). Task 11 (frontend) consumes both.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_server.py`:

```python
def test_get_memory_returns_empty_string_when_file_does_not_exist(monkeypatch, tmp_path):
    monkeypatch.setattr("pos.cli.server.MEMORY_PATH", tmp_path / "MEMORY.md")
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm(),
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=SessionStore(":memory:"),
    )
    client = TestClient(app)

    resp = client.get("/memory")

    assert resp.status_code == 200
    assert resp.json() == {"content": ""}


def test_post_memory_writes_the_file_and_get_reflects_it(monkeypatch, tmp_path):
    monkeypatch.setattr("pos.cli.server.MEMORY_PATH", tmp_path / "MEMORY.md")
    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: FakeLlm(),
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=SessionStore(":memory:"),
    )
    client = TestClient(app)

    post_resp = client.post("/memory", json={"content": "- prefers short answers"})
    get_resp = client.get("/memory")

    assert post_resp.status_code == 200
    assert get_resp.json() == {"content": "- prefers short answers"}


def test_ws_passes_memory_content_to_agent(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MIN_SPEECH_SEC", 0.01)
    memory_path = tmp_path / "MEMORY.md"
    memory_path.write_text("- prefers short answers")
    monkeypatch.setattr("pos.cli.server.MEMORY_PATH", memory_path)
    captured = {}

    class _SpyLlm(FakeLlm):
        def stream(self, messages, cancel, usage=None, extra_system_context=None):
            captured["extra_system_context"] = extra_system_context
            return super().stream(messages, cancel, usage, extra_system_context)

    app = create_app(
        stt=FakeStt("hello"),
        tts_engines={"kokoro": FakeTts},
        llm_factory=lambda model: _SpyLlm("hi there"),
        vad_factory=lambda **kw: FakeVad(start_at=1, end_at=3),
        default_tts_engine="kokoro",
        session_store=SessionStore(":memory:"),
    )
    client = TestClient(app)

    with client.websocket_connect("/ws?mode=text") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "hello"})
        ws.receive_json()  # user_text
        ws.receive_json()  # bot_text

    assert captured["extra_system_context"] == "- prefers short answers"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_server.py -k memory -v`
Expected: FAIL — `AttributeError: module 'pos.cli.server' has no attribute 'MEMORY_PATH'` (first two), and the third fails because `_SpyLlm` never receives real memory content (still `None`)

- [ ] **Step 3: Implement in `server.py`**

Add a module-level constant near the top of the file (alongside other
module-level definitions like `SessionKeysRequest`):
```python
MEMORY_PATH = Path("memory/MEMORY.md")


class MemoryRequest(BaseModel):
    content: str
```

Add two new routes (near `/session-keys`):
```python
    @app.get("/memory")
    async def get_memory():
        try:
            content = MEMORY_PATH.read_text()
        except FileNotFoundError:
            content = ""
        return {"content": content}

    @app.post("/memory")
    async def post_memory(body: MemoryRequest):
        MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        MEMORY_PATH.write_text(body.content)
        return {"content": body.content}
```

In `ws_endpoint`, add a closure and pass it into `Agent(...)`. Find:
```python
        agent = Agent(
            vad=vad, stt=stt, tts=tts, llm=llm,
            trigger_word=trigger_word, audio_sink=sink, on_event=emit,
            text_only=(mode == "text"), conversation=resumed_conversation,
        )
```
Replace with:
```python
        def read_memory() -> str | None:
            try:
                return MEMORY_PATH.read_text() or None
            except FileNotFoundError:
                return None

        agent = Agent(
            vad=vad, stt=stt, tts=tts, llm=llm,
            trigger_word=trigger_word, audio_sink=sink, on_event=emit,
            text_only=(mode == "text"), conversation=resumed_conversation,
            memory_facts=read_memory,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_server.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Add `memory/` to `.gitignore`**

Add immediately after the `/skills/` entry from Task 5:
```

# Always-on user memory, a single Markdown file (see
# src/pos/cli/server.py's MEMORY_PATH) -- same local/personal
# reasoning as plugins/ and skills/.
/memory/
```

- [ ] **Step 6: Run the full suite + ruff**

Run: `uv run pytest tests/ -q && uv run ruff check .`
Expected: PASS, all green

- [ ] **Step 7: Commit**

```bash
git add src/pos/cli/server.py tests/test_server.py .gitignore
git commit -m "feat: GET/POST /memory, wire memory/MEMORY.md into Agent's memory_facts"
```

---

### Task 11: Settings UI — Memory tab

**Files:**
- Modify: `src/pos/static/index.html`

**Interfaces:**
- Consumes: `GET /memory`, `POST /memory` from Task 10; `settingsPanels.memory` (the `#settingsPanel-memory` div) from Task 1.
- Produces: nothing further downstream (final task).

- [ ] **Step 1: Add the Memory tab content**

Find:
```html
          <div id="settingsPanel-memory" hidden></div>
```

Replace with:
```html
          <div id="settingsPanel-memory" hidden>
            <div class="settings-section-title">Memory</div>
            <p class="settings-hint">
              Always loaded into every conversation, unlike Skills (which load
              only when relevant). Plain Markdown -- edit here or directly on
              disk at <code>memory/MEMORY.md</code> on the server.
            </p>
            <textarea id="memoryTextarea" class="memory-textarea" rows="14"
              placeholder="- prefers short answers&#10;- timezone is IST"></textarea>
            <div class="memory-actions">
              <button type="button" class="secondary" id="memorySaveBtn">Save</button>
              <span class="settings-hint" id="memorySaveStatus"></span>
            </div>
          </div>
```

- [ ] **Step 2: Add the textarea CSS**

Find the `.settings-tab.active` rule added in Task 1 and add immediately
after it:
```css
  .memory-textarea {
    width: 100%;
    max-width: 640px;
    background: var(--surface-sunken);
    border: 1px solid transparent;
    border-radius: 8px;
    color: var(--text);
    font-family: var(--font-mono);
    font-size: 13px;
    padding: 10px 12px;
    outline: none;
    resize: vertical;
  }
  .memory-textarea:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-tint); }
  .memory-textarea::placeholder { color: var(--text-faint); }
  .memory-actions { display: flex; align-items: center; gap: 10px; margin-top: 10px; }
```

- [ ] **Step 3: Load and save the memory content**

Find the tab-switching JS added in Task 1:
```js
  settingsTabs.forEach((tab) => tab.addEventListener("click", () => showSettingsTab(tab.dataset.tab)));
  showSettingsTab("general");
```

Replace with (loads memory content the first time the Memory tab is
opened, not eagerly on page load, matching the Connections diagram's own
lazy-load pattern):
```js
  let memoryLoaded = false;
  const memoryTextarea = el("memoryTextarea"), memorySaveBtn = el("memorySaveBtn"), memorySaveStatus = el("memorySaveStatus");

  async function loadMemoryIfNeeded() {
    if (memoryLoaded) return;
    memoryLoaded = true;
    try {
      const data = await (await fetch("/memory")).json();
      memoryTextarea.value = data.content;
    } catch (e) {
      memorySaveStatus.textContent = "Couldn't load memory from the server.";
    }
  }

  memorySaveBtn.onclick = async () => {
    memorySaveBtn.disabled = true;
    memorySaveStatus.textContent = "Saving…";
    try {
      await fetch("/memory", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: memoryTextarea.value }),
      });
      memorySaveStatus.textContent = "Saved.";
    } catch (e) {
      memorySaveStatus.textContent = "Couldn't save -- check the server.";
    }
    memorySaveBtn.disabled = false;
  };

  settingsTabs.forEach((tab) => tab.addEventListener("click", () => {
    showSettingsTab(tab.dataset.tab);
    if (tab.dataset.tab === "memory") loadMemoryIfNeeded();
  }));
  showSettingsTab("general");
```

- [ ] **Step 4: Verify JS syntax and id cross-check**

Run the same verification script as Task 1, Step 4.
Expected: `SYNTAX OK`, both diffs `none`.

- [ ] **Step 5: Manual verification**

Start the server, open Settings, click "Memory", confirm the textarea
loads (empty, since no `memory/MEMORY.md` exists yet). Type
`- prefers short answers`, click Save, confirm the status text says
"Saved." and that `memory/MEMORY.md` now exists on disk with that
content. Reload the page, reopen the Memory tab, confirm the saved
content reloads.

- [ ] **Step 6: Run the full backend suite (sanity check)**

Run: `uv run pytest tests/ -q`
Expected: PASS — this task is frontend-only.

- [ ] **Step 7: Commit**

```bash
git add src/pos/static/index.html
git commit -m "feat: Memory tab in Settings (textarea + Save, backed by memory/MEMORY.md)"
```

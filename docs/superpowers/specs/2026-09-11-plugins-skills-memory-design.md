# Plugins, Skills, and Memory Design

## Motivation

The user wants three capabilities modeled after Claude Code's own extension
system (plugins, skills, memory) and DeepSeek Harness's "everything is a
plugin" philosophy, adapted to this app's actual shape: a local FastAPI
voice/text agent with an LLM provider registry (see
`docs/superpowers/specs/2026-09-10-llm-provider-registry-and-src-layout-design.md`)
and a hand-rolled tool-calling loop in `llm/langchain_llm.py`.

Research findings that shaped this design (see conversation history for
full detail):

- **Claude Code**: a plugin is a directory with a manifest
  (`.claude-plugin/plugin.json`) pointing at conventional subfolders
  (`skills/`, `commands/`, `agents/`, `hooks/`). A **skill** is just
  `skills/<name>/SKILL.md` — YAML frontmatter (`name`, `description`) plus
  an instruction body; the model sees every skill's one-line description
  and decides contextually whether to load the full body. **Memory**
  (`CLAUDE.md`) is different in kind from skills: always loaded into every
  session, not contextually decided. Anthropic's own guidance: keep it
  small (under ~300 lines) and push detail into separate files loaded on
  demand (progressive disclosure) — the same shape as the memory system
  this assistant already uses across sessions.
- **DeepSeek Harness** (`deepseek-ai/deepseek-harness`): "everything is a
  plugin," including provider selection — validates that this app's
  existing provider registry (`llm/providers/`) is already the right
  shape for that piece; no change needed there.
- **LangChain's `create_agent` + `deepagents` package**: `deepagents`
  ships `SkillsMiddleware`/`MemoryMiddleware` that do almost exactly this
  — but as a spike confirmed, adopting them means adopting the separate
  `deepagents` framework (`FilesystemMiddleware`, `SubAgentMiddleware`,
  `StateBackend`, etc.) for two small pieces of behavior this app can
  implement directly in far less code, with no new dependency. Spike also
  confirmed `create_agent`'s `stream_mode="messages"` preserves the
  token-by-token streaming this app's TTS pipeline depends on — so a
  future engine migration is *possible* if ever wanted, but is
  **explicitly out of scope for this spec** (see Non-goals).
- **Graph-based memory** (Zep/Graphiti, Mem0): built for multi-entity,
  temporally-evolving relationship tracking at enterprise scale. This
  app's memory need (a handful of personal facts for one local user) has
  no relational complexity to justify a graph store — confirmed via
  research, not assumed. A flat Markdown file is right-sized here.

## Architecture overview

Three distinct concepts, two of which converge on machinery this app
already has:

- **Plugins** are *code* — installable bundles that add new tools the LLM
  can call. They plug into the exact spot `get_current_time`/`web_search`
  already occupy: `default_tools()` in `src/pos/llm/tools.py`.
- **Skills** are *data* — Markdown files, not code. Implemented as one new
  built-in tool, `load_skill(name)`, added to the *same* tools list
  plugins populate — reusing the existing hand-rolled tool-execution loop
  in `LangChainLlm.stream()` (see `src/pos/agent.py`'s `worker_thread`
  → `respond()` → `self.llm.stream(...)`). No new execution mechanism.
- **Memory** is *always-on* — concatenated into the system prompt every
  turn, unconditionally, unlike skills which the model decides whether to
  load. Needs one new parameter threaded through `LlmBase.stream()` and
  `Agent.respond()`.

All three get a Settings UI section using the same row-list pattern
already built for Providers (`src/pos/static/index.html`'s
`.provider-list`/`.provider-row` — see the 2026-09-11 API keys UI work):
one row per item, a status dot, an enable/disable toggle where relevant.

## Plugins

### Storage layout

```
plugins/                       <- gitignored, like data/ (see .gitignore)
  weather/
    plugin.json
    tool.py
  enabled.json                 <- ["weather"] -- which installed plugins are active
```

`plugin.json`:
```json
{
  "name": "weather",
  "version": "0.1.0",
  "description": "Look up current weather for a location.",
  "entry_point": "tool:get_tools"
}
```

`entry_point` is `"<module_file_stem>:<callable_name>"`, resolved relative
to the plugin's own directory. The callable takes no arguments and
returns `list[BaseTool]` (plain LangChain `@tool`-decorated functions,
same style as `get_current_time` in `llm/tools.py` today):

```python
# plugins/weather/tool.py
from langchain_core.tools import tool

@tool
def get_weather(location: str) -> str:
    """Return current weather conditions for a location."""
    ...

def get_tools():
    return [get_weather]
```

### Discovery and loading

New module `src/pos/llm/plugins.py`:

```python
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from langchain_core.tools import BaseTool

PLUGINS_DIR = Path("plugins")


def discover_plugins() -> list[dict]:
    """One dict per plugins/*/plugin.json found on disk -- does not
    import any plugin code, safe to call on every /options request."""
    if not PLUGINS_DIR.is_dir():
        return []
    found = []
    for manifest_path in sorted(PLUGINS_DIR.glob("*/plugin.json")):
        manifest = json.loads(manifest_path.read_text())
        found.append({**manifest, "_dir": str(manifest_path.parent)})
    return found


def _enabled_names() -> set[str]:
    enabled_path = PLUGINS_DIR / "enabled.json"
    if not enabled_path.exists():
        return set()
    return set(json.loads(enabled_path.read_text()))


def set_enabled(name: str, enabled: bool) -> None:
    names = _enabled_names()
    names.add(name) if enabled else names.discard(name)
    PLUGINS_DIR.mkdir(exist_ok=True)
    (PLUGINS_DIR / "enabled.json").write_text(json.dumps(sorted(names)))


def load_enabled_plugin_tools() -> list[BaseTool]:
    """Imports and calls each enabled plugin's entry_point -- unlike
    discover_plugins(), this DOES execute plugin code, so it's only
    called from default_tools(), not from /options."""
    tools: list[BaseTool] = []
    enabled = _enabled_names()
    for manifest in discover_plugins():
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

`default_tools(env=None)` in `llm/tools.py` gains one line appending
`plugins.load_enabled_plugin_tools()`'s result (importing the new
`pos.llm.plugins` module), after the existing
`get_current_time`/`web_search` logic.

### Settings UI

New "Plugins" section: `GET /options` gains a `"plugins"` field (from
`discover_plugins()` merged with enabled-state — cheap, no code
execution, same reasoning as today's `tool_status()`). A new
`POST /plugins/{name}/toggle` endpoint calls `set_enabled()`. Frontend:
one row per discovered plugin, toggle switch instead of Provider's radio
(multiple plugins can be enabled at once, unlike providers).

**Explicitly deferred to a follow-up** (not this spec): a "install from a
git URL" flow. v1 only discovers plugins already dropped into `plugins/`
by hand — the enable/disable/store *UI* is the real ask; remote install
is a separate, smaller follow-up once the core mechanism exists.

## Skills

### Storage layout

```
skills/
  cooking-helper/
    SKILL.md
  trip-planner/
    SKILL.md
  enabled.json                 <- ["cooking-helper"] -- which skills the model can load
```

`SKILL.md`:
```markdown
---
name: cooking-helper
description: Helps with recipes, substitutions, and cooking techniques.
---

You are now acting as a cooking assistant. When the user asks about a
recipe, substitution, or technique, give concrete, concise guidance...
```

### The `load_skill` tool

`PyYAML` is currently only a *transitive* dependency (pulled in by
something else in the tree, confirmed importable today) — relying on
that implicitly is fragile, so the implementation plan adds it as an
explicit direct dependency (`uv add pyyaml`) rather than assuming it
stays available.

New module `src/pos/llm/skills.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import yaml
from langchain_core.tools import tool

SKILLS_DIR = Path("skills")


def _enabled_skills() -> list[dict]:
    """[{"name", "description", "body"}] for every enabled skill --
    frontmatter parsed eagerly (cheap), body read lazily only when
    load_skill() actually picks one."""
    enabled_path = SKILLS_DIR / "enabled.json"
    if not SKILLS_DIR.is_dir() or not enabled_path.exists():
        return []
    enabled = set(json.loads(enabled_path.read_text()))
    skills = []
    for skill_dir in sorted(SKILLS_DIR.glob("*/")):
        if skill_dir.name not in enabled:
            continue
        md_path = skill_dir / "SKILL.md"
        if not md_path.exists():
            continue
        raw = md_path.read_text()
        _, frontmatter, body = raw.split("---", 2)
        meta = yaml.safe_load(frontmatter)
        skills.append({"name": meta["name"], "description": meta["description"], "body": body.strip()})
    return skills


def make_load_skill_tool():
    """A fresh tool instance per LangChainLlm construction (not a module
    singleton) so its description reflects whichever skills are enabled
    *right now* -- descriptions are baked into the tool schema at
    bind_tools() time, not re-read per call."""
    skills_by_name = {s["name"]: s["body"] for s in _enabled_skills()}
    catalog = "\n".join(f"- {s['name']}: {s['description']}" for s in _enabled_skills())

    @tool
    def load_skill(name: str) -> str:
        f"""Load a skill's full instructions by name when its topic
        becomes relevant to the conversation. Available skills:
        {catalog}"""
        return skills_by_name.get(name, f"unknown skill: {name}")

    return load_skill
```

`default_tools(env=None)` appends `make_load_skill_tool()`'s result only
when at least one skill is enabled (an empty catalog would be a
useless, confusing tool).

### Settings UI

Same row-list pattern as Plugins: one row per `skills/*/SKILL.md` found,
toggle to enable/disable, reusing `enabled.json`'s shape.

## Memory

### Storage

One file, `memory/MEMORY.md` — gitignored, plain Markdown, human-editable
directly on disk or through Settings. No database.

### Injection

`LlmBase.stream()` (`src/pos/interfaces/llm.py`) gains one new
optional parameter:

```python
@abstractmethod
def stream(
    self, messages: list[dict], cancel: threading.Event,
    usage: dict | None = None, extra_system_context: str | None = None,
) -> Iterator[str]: ...
```

`LangChainLlm.stream()` uses it right where it already builds the
message list:

```python
full: list[BaseMessage] = [SystemMessage(self.system_prompt)]
if extra_system_context:
    full.append(SystemMessage(extra_system_context))
```

`Agent.__init__` (`src/pos/agent.py`) gains one new constructor
parameter, `memory_facts: Callable[[], str | None] | None = None` — same
callback pattern as the existing `on_event` parameter. `Agent.respond()`
calls it right before `self.llm.stream(...)`:

```python
extra_context = self.memory_facts() if self.memory_facts else None
for piece in self.llm.stream(messages, self.cancel, usage, extra_context):
```

`server.py`'s `ws_endpoint` passes a closure reading `memory/MEMORY.md`
(try/except for "file doesn't exist yet", same defensive pattern already
used around `store.add_turn`/`store.set_title`):

```python
def read_memory() -> str | None:
    try:
        return Path("memory/MEMORY.md").read_text() or None
    except FileNotFoundError:
        return None

agent = Agent(..., memory_facts=read_memory)
```

`OpenAiCompatibleLlm` (the other `LlmBase` implementation, "kept for
reference/tests" per its own module docstring) gets the same parameter
added to its `stream()` signature for interface consistency, ignored if
unused.

### Settings UI

New "Memory" section: a plain `<textarea>` showing `memory/MEMORY.md`'s
current content (via a new `GET /memory` endpoint) and a "Save" button
(`POST /memory` writing the full replacement content). No list/delete-row
UI — it's one file, edited as a whole, the same way a person would edit
`CLAUDE.md` directly.

## Non-goals (explicit, for this spec)

- **No migration of `LangChainLlm`'s hand-rolled tool loop to
  `create_agent`.** The spike confirmed streaming compatibility, but
  migrating the engine is a separate, larger decision not required to
  ship plugins/skills/memory — deferred.
- **No `deepagents` dependency.** Its `SkillsMiddleware`/`MemoryMiddleware`
  do similar things but require adopting its whole framework
  (`FilesystemMiddleware`, `SubAgentMiddleware`, `StateBackend`) for
  functionality this spec implements directly in ~150 lines total.
- **No graph-based memory** (Zep/Graphiti/Mem0) — this app's memory need
  has no multi-entity relational complexity to justify it (confirmed via
  research, not assumed).
- **No remote/git-based plugin installation** in v1 — plugins are
  discovered from `plugins/` on disk; a "paste a git URL to install" flow
  is a follow-up once the enable/disable/discovery mechanism is proven.
- **No plugin sandboxing.** A plugin's Python code runs with the same
  trust level as the rest of this local, single-user server process —
  consistent with this app having no auth/multi-tenancy model at all
  today. Explicitly not addressed here; installing a plugin is an
  intentional, trusted action, same as installing any local Python
  package.
- **No multi-user memory.** `memory/MEMORY.md` is process-wide, matching
  every other piece of local state in this app (SQLite `data/sessions.db`,
  the provider env-var defaults) — this is a single-user local tool, not
  a hosted multi-tenant product.

## Testing strategy

- `llm/plugins.py`: `discover_plugins()`/`set_enabled()`/
  `load_enabled_plugin_tools()` unit-tested with temp directories (`tmp_path`
  fixture), mirroring `test_providers_*.py`'s style — no real plugin
  package installation needed for tests.
- `llm/skills.py`: `_enabled_skills()`/`make_load_skill_tool()` tested
  the same way, with temp `SKILL.md` files.
- `LangChainLlm.stream()`: new test asserting `extra_system_context`
  becomes a second `SystemMessage`, following the existing
  `test_langchain_llm.py` fake-runnable pattern.
- `Agent`: new test asserting `respond()` calls `memory_facts()` and
  threads its result into `self.llm.stream(...)`'s new parameter,
  following `test_agent.py`'s existing `FakeLlm`-based style.
- `server.py`: new tests for `GET/POST /memory`, `GET /options`'s new
  `"plugins"` field, and `POST /plugins/{name}/toggle`, following
  `test_server.py`'s existing `TestClient` patterns.

## Follow-ups (not this spec)

- Remote plugin installation (git URL → clone into `plugins/`).
- Revisiting the `create_agent` engine migration once plugins/skills/
  memory are shipped and stable, if there's a concrete reason to want
  `deepagents`' broader middleware ecosystem (subagents, retries, PII
  filtering, human-in-the-loop).
- Splitting `memory/MEMORY.md` into an index + per-topic detail files
  (progressive disclosure) if a single file's content starts growing
  past the ~300-line guidance this design is based on.

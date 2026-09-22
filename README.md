# pos-harness

A local coding and research agent with a browser UI. You open a folder, it
reads and writes files in it, runs commands on your machine, searches the
web, and shows you exactly what every turn cost.

Single user, runs on your own computer. There is no login, no server, and
no account — the model's access to your files is the point, not a
side-effect.

Built on [LangChain](https://docs.langchain.com/)'s `create_agent` with
[deepagents](https://github.com/langchain-ai/deepagents) for the filesystem,
shell and skills. Azure OpenAI only.

## Install and run

```bash
uvx pos-backend          # run it without installing
uv tool install pos-backend && pos   # or install it, then `pos`
```

[uv](https://docs.astral.sh/uv/) downloads Python if you don't have it, so
that is the whole dependency list. The app opens in your browser at
<http://127.0.0.1:8000>.

Before the first run, put your Azure credentials in a `.env` file beside
wherever you run it — see [`.env.example`](.env.example). Nothing is ever
entered through the UI.

## What it does

- **Open a folder** through the real OS dialog. The agent is sandboxed to
  it: it cannot read above that folder, and your `.env` stays out of reach.
- **Chat**, with replies streamed token by token. Markdown, tables and code
  blocks render properly.
- **Tools**: read, write, edit, search and list files; run shell commands;
  search the web with [Tavily](https://tavily.com/) if you set a key.
- **Skills** — folders of reference material (pdf, docx, xlsx, pptx and
  more) that the agent opens only when a task matches. It sees each one's
  name and description up front and reads the rest on demand.
- **AGENTS.md** — standing instructions per project, following the
  [agents.md](https://agents.md/) spec. A global one applies everywhere; a
  folder's own layers on top.
- **Trace** — every turn as a graph: each model round, each tool call, what
  each one added to the context, cache hits, and cost.
- **Usage** — tokens, cost and activity across all your chats.

## Development

Two servers, hot reload on both:

```bash
# terminal 1
cd backend && uv run uvicorn pos.app:app --reload --port 8000

# terminal 2
cd frontend && npm install && npm run dev
```

The UI is on <http://localhost:3000> and talks to the API on `:8000`. The
backend enables CORS for that origin only while running this way.

### How it is packaged

The frontend is Next.js, but nothing Node-related is needed to *run* the
app — only to build it. `next build` with `output: 'export'` produces plain
files, a [hatchling](https://hatch.pypa.io/) build hook copies them into the
Python package, and FastAPI serves them. So the packaged app is one process
on one port, and a user installs Python and nothing else.

```bash
cd backend && uv build --wheel     # builds the UI and bundles it
```

The hook removes its output from your checkout afterwards, so development
keeps using the Next dev server.

### Where things live

| | |
|---|---|
| `backend/src/pos/app.py` | the endpoints |
| `backend/src/pos/agent.py` | how the agent is assembled |
| `backend/src/pos/prompt.py` | the system prompt |
| `backend/src/pos/trace.py` | per-round and per-tool token accounting |
| `backend/skills/` | the global skills |
| `backend/memory/AGENTS.md` | global standing instructions |
| `frontend/src/components/` | the UI |

Your chats live in SQLite — `data/checkpoint.db` in a checkout, or your OS
user-data folder when installed. `POS_DATA_DIR` overrides both.

## Configuration

Everything is read from `.env`; see [`.env.example`](.env.example) for the
full list. The ones you are most likely to change:

| | |
|---|---|
| `AZURE_OPENAI_DEPLOYMENT` | which deployment to call |
| `AZURE_OPENAI_CONTEXT_WINDOW` | used by the context meter |
| `AZURE_OPENAI_PRICE_*` | per-1M-token rates, for the cost figures |
| `TAVILY_API_KEY` | enables web search |
| `POS_ROOT_DIR` | the folder used when none has been picked |

## Docker

```bash
WORKSPACE=C:/Users/me/projects docker compose up --build
```

One service, not two: the image builds the UI and the backend serves it, so
there is nothing else to run.

This is for deploying the app somewhere, **not** for daily use on your own
machine. A container has no display, so the folder picker cannot open --
`POST /fs/pick` answers 501 and you set the folder with `POS_ROOT_DIR`
instead. `execute` runs inside the container rather than on your computer,
and the agent sees only what `WORKSPACE` mounts.

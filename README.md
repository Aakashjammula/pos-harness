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

One server, always:

```bash
uv run uvicorn pos.app:app --reload --port 8000
```

That serves the API *and* the UI. There is no second process.

### Changing the UI

The frontend is Next.js, but nothing Node-related runs when the app runs —
Node is only needed to build. `next build` with `output: 'export'` produces
plain files, and those files are committed at `backend/src/pos/static/`,
which is what the server hands out.

So after editing anything under `frontend/src/`:

```bash
cd frontend && npm run build:app
```

That rebuilds and copies the result into `backend/src/pos/static/`. Commit
that folder along with your source change — it is the part that actually
ships.

Changes do not appear until you rebuild. That is the trade for having one
command instead of two.

### Where things live

| | |
|---|---|
| `src/pos/app.py` | the endpoints |
| `src/pos/agent.py` | how the agent is assembled |
| `src/pos/prompt.py` | the system prompt |
| `src/pos/trace.py` | per-round and per-tool token accounting |
| `src/pos/static/` | the built UI, committed |
| `skills/` | the global skills |
| `memory/AGENTS.md` | global standing instructions |
| `frontend/src/components/` | the UI |

Your chats live in SQLite — `data/checkpoint.db` in a checkout, or your OS
user-data folder when installed. `POS_DATA_DIR` overrides both.

## Configuration

Everything is read from `.env`; see [`.env.example`](.env.example) for the
full list. The ones you are most likely to change:

| | |
|---|---|
| `AZURE_OPENAI_*` / `OPENAI_API_KEY` | whichever you set decides the provider |
| `POS_MODEL` | the model, or on Azure the deployment name |
| `POS_MODELS` | Azure deployments to offer. OpenAI models list themselves from the key |
| `TAVILY_API_KEY` | enables web search |
| `POS_ROOT_DIR` | the folder used when none has been picked |

Prices and context windows are **not** configured — they are looked up from
[models.dev](https://models.dev) for whichever model you name, cached to
disk for a day. Its figures for `gpt-5.6-luna` were checked against
Microsoft's own [Retail Prices API](https://prices.azure.com/api/retail/prices)
and matched exactly. The `AZURE_OPENAI_PRICE_*` variables still exist as
overrides for a deployment it doesn't know.

## Docker

```bash
WORKSPACE=C:/Users/me/projects docker compose up --build
```

One service, and no Node stage in the image: the UI is already built and
committed, so the image only installs Python dependencies.

This is for deploying the app somewhere, **not** for daily use on your own
machine. A container has no display, so the folder picker cannot open --
`POST /fs/pick` answers 501 and you set the folder with `POS_ROOT_DIR`
instead. `execute` runs inside the container rather than on your computer,
and the agent sees only what `WORKSPACE` mounts.

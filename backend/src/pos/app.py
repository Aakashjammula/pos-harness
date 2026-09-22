"""The FastAPI app: the endpoints the frontend needs.

  POST /chat/stream    Send a message, stream the reply back as
                       server-sent events: `session {id}`, `token {text}`,
                       `title {title}` (once, after the first reply),
                       `done {text, model, finish_reason,
                       reasoning_effort, usage, tool_calls}`,
                       `error {message}`.
  GET  /sessions       List past sessions, newest first.
  GET  /sessions/{id}  One session's stored turns.
  DEL  /sessions/{id}  Delete a session, its turns, and its memory.
  POST /fs/pick        Opens the native OS folder dialog, blocks until
                       closed, returns the chosen path (or none).

Single local user: no auth, no per-user isolation.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel

from pos import agent as agent_mod
from pos import config
from pos import db
from pos import fs
from pos import prompt
from pos import titles
from pos import trace as trace_mod
from pos import usage


class ChatBody(BaseModel):
    """Request body for POST /chat/stream."""

    message: str
    thread_id: str | None = None
    folder: str | None = None
    reasoning_effort: str = "medium"


def _sse(event: str, data: dict) -> str:
    """Formats one server-sent event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _resolve_root_dir(folder: str | None) -> str | None:
    """The agent's filesystem root for one request.

    `folder` is whatever the client sent -- normally a real absolute path
    from POST /fs/pick, but a client could still send nothing or a stale
    path, so it's validated rather than trusted outright. None here means
    "use agent.build_agent's own default".
    """
    if folder and Path(folder).is_dir():
        return folder
    return None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Opens the checkpointer once, for the server's lifetime.

    The agent itself is built fresh per request (see chat_stream) so each
    request's filesystem tool can point at that request's own folder --
    building it is cheap (no I/O), unlike the checkpointer's connection.
    """
    db.connect().close()  # ensure the app's own tables exist before serving
    with SqliteSaver.from_conn_string(str(db.DB_PATH)) as checkpointer:
        app.state.checkpointer = checkpointer
        yield


app = FastAPI(lifespan=lifespan)

# Single local user, frontend and backend on different ports (:3000 / :8000)
# during development -- without this the browser blocks every request.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat/stream")
async def chat_stream(body: ChatBody):
    """Streams one reply as server-sent events."""
    text = body.message.strip()
    if not text:
        raise HTTPException(status_code=422, detail="message is empty")

    new_session = body.thread_id is None
    thread_id = body.thread_id or str(uuid.uuid4())
    root_dir = _resolve_root_dir(body.folder)

    async def events() -> AsyncIterator[str]:
        yield _sse("session", {"id": thread_id})

        conn = db.connect()
        try:
            if new_session:
                # A provisional title (old-style truncation) until the real
                # one lands after the first reply -- see the `title` event
                # below. Better than a blank sidebar entry in the meantime.
                provisional = text if len(text) <= 60 else text[:57] + "..."
                conn.execute(
                    "INSERT INTO sessions (id, folder, title) VALUES (?, ?, ?)",
                    (thread_id, body.folder, provisional),
                )
                conn.commit()
            conn.execute(
                "INSERT INTO turns (thread_id, role, text) VALUES (?, 'user', ?)",
                (thread_id, text),
            )
            conn.commit()

            agent = agent_mod.build_agent(app.state.checkpointer, root_dir=root_dir)
            cfg = {"configurable": {"thread_id": thread_id}}
            # Messages the thread already had, so we can tell which ones
            # THIS call adds (see the backend design discussion: a fresh
            # thread's state is `{}`, not a "messages": [] key).
            prior_count = len(agent.get_state(cfg).values.get("messages", []))

            try:
                stream = agent.stream_events(
                    {"messages": [{"role": "user", "content": text}]},
                    version="v3",
                    config=cfg,
                    reasoning_effort=body.reasoning_effort.lower(),
                )
            except Exception as e:  # noqa: BLE001 -- reported as an error event
                yield _sse("error", {"message": str(e)[:300] or type(e).__name__})
                return

            full_text = ""
            seen_calls: set[str] = set()
            try:
                for message in stream.messages:
                    # Tool calls repeat on every chunk as the message accumulates,
                    # so announce each one only the first time its id appears --
                    # this is what turns the silent "thinking" dots into
                    # "Reading /skills/pdf/SKILL.md...".
                    for call in getattr(message, "tool_calls", None) or []:
                        call_id = call.get("id")
                        if call_id and call_id not in seen_calls:
                            seen_calls.add(call_id)
                            yield _sse("activity", {"tool": call.get("name"), "args": call.get("args") or {}})
                    for delta in message.text:
                        full_text += delta
                        yield _sse("token", {"text": delta})
            except Exception as e:  # noqa: BLE001 -- mid-stream failure
                yield _sse("error", {"message": str(e)[:300] or type(e).__name__})
                return

            final_state = stream.output
            new_messages = final_state["messages"][prior_count:]
            payload = trace_mod.build_trace(new_messages, reasoning_effort=body.reasoning_effort.lower())
            payload["text"] = full_text

            conn.execute(
                """INSERT INTO turns
                   (thread_id, role, text, model, finish_reason, reasoning_effort, usage_json, tool_calls_json)
                   VALUES (?, 'assistant', ?, ?, ?, ?, ?, ?)""",
                (
                    thread_id,
                    full_text,
                    payload["model"],
                    payload["finish_reason"],
                    payload["reasoning_effort"],
                    json.dumps(payload["usage"]),
                    json.dumps(payload["tool_calls"]),
                ),
            )
            conn.commit()

            yield _sse("done", payload)

            if new_session:
                title = titles.generate(text, full_text)
                if title:
                    conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, thread_id))
                    conn.commit()
                    yield _sse("title", {"title": title})
        finally:
            conn.close()

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/sessions")
def list_sessions(folder: str | None = None):
    """Lists past sessions, newest first."""
    conn = db.connect()
    try:
        if folder is not None:
            rows = conn.execute(
                "SELECT id, folder, title, created_at FROM sessions WHERE folder = ? ORDER BY created_at DESC",
                (folder,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, folder, title, created_at FROM sessions ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    """One session's stored turns, oldest first."""
    conn = db.connect()
    try:
        session = conn.execute("SELECT id FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        rows = conn.execute(
            """SELECT role, text, model, finish_reason, reasoning_effort, usage_json, tool_calls_json, created_at
               FROM turns WHERE thread_id = ? ORDER BY id""",
            (session_id,),
        ).fetchall()
        turns = []
        for r in rows:
            turn = dict(r)
            # .pop() always runs here (unlike inside a ternary, where the
            # untaken branch -- and its pop -- never executes, leaking the
            # raw *_json fields into turns with no usage/tool_calls).
            usage_json = turn.pop("usage_json")
            tool_calls_json = turn.pop("tool_calls_json")
            turn["usage"] = json.loads(usage_json) if usage_json else None
            turn["tool_calls"] = json.loads(tool_calls_json) if tool_calls_json else None
            turns.append(turn)
        return {"turns": turns}
    finally:
        conn.close()


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Deletes a session: its metadata, its turns, and its memory.

    All three have to go together. Leaving the checkpoint behind would mean
    a new chat that reused the id would silently inherit the old
    conversation, and leaving the turns behind would keep it in the usage
    totals. Permanent -- there's no undo and no backup.
    """
    if not db.delete_session(session_id):
        raise HTTPException(status_code=404, detail="session not found")
    app.state.checkpointer.delete_thread(session_id)
    return {"deleted": session_id}


@app.post("/fs/pick")
def pick_folder():
    """Opens the native OS folder dialog; blocks until it's closed."""
    return {"path": fs.pick_folder()}


@app.get("/usage")
def get_usage(range: str = "all"):  # noqa: A002 -- matches the query param's name
    """Aggregated usage stats for the settings dashboard."""
    return usage.summary(range)


@app.get("/tools")
def list_tools():
    """What the agent can currently do, for the settings UI."""
    return {
        "tools": agent_mod.active_tools(),
        "skills": agent_mod.available_skills(),
        "model": config.MODEL_NAME,
    }


@app.get("/system-prompt")
def get_system_prompt():
    """The agent's system prompt, read-only, for the settings UI.

    Only the part we write: the file and skills middleware append their
    own instructions at request time, which is why this is shorter than
    what the model actually receives.
    """
    return {"prompt": prompt.SYSTEM_PROMPT}

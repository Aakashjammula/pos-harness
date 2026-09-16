# Transport / Provider / LLM-harness architecture

Date: 2026-09-09
Status: draft, awaiting review

## Context

`pos` is currently one local process: `main.py` builds an `Agent`
that owns a `sounddevice` mic `InputStream`, runs VAD/STT/LLM/TTS on three
background threads connected by queues, and writes synthesized audio to a
`sounddevice` `OutputStream`. VAD/STT/LLM/TTS already sit behind small
ABCs (`src/pos/interfaces/`) so engines are swappable, but only one
concrete implementation per stage is wired up today (STT: Parakeet-TDT
via onnx-asr; LLM: any OpenAI-compatible server, currently LM Studio;
TTS: Kokoro or Supertonic — both local/CPU).

Three initiatives are wanted, all touching this same pipeline:

1. Replace direct `sounddevice` ownership with a websocket-based
   transport, supporting three run modes: local (no network, today's
   behavior), a FastAPI server + browser/HTML client, and a FastAPI
   server + a Python CLI client.
2. Harden the provider abstraction so cloud engines can be added
   alongside local ones for STT/TTS/LLM without redesigning interfaces.
3. Add an LLM harness (LangChain or equivalent) for tool/function
   calling.

These are independent enough to spec and implement in phases, but they
share one foundation (the provider interfaces) and this document lays
out all three so later phases don't conflict with decisions made now.

## Decisions already made (from brainstorming)

- All three transport modes are wanted: local, FastAPI+HTML, FastAPI+CLI.
- Audio over the wire: raw PCM16 mono 16kHz frames (no codec) — matches
  the pipeline's existing internal format, zero encode/decode complexity.
- The FastAPI server supports **multiple concurrent sessions**, each
  with isolated conversation/turn state.
- Provider abstraction: keep today's ABC-per-stage pattern; prove out
  cloud support with one example per stage rather than committing to
  specific vendors up front. LLM already has an OpenAI-compatible
  client (works against both LM Studio and real OpenAI with just
  different constructor args) — STT/TTS need one new cloud
  implementation each to prove the interface holds.
- LLM harness priority: tool/function calling (not just provider-agnostic
  model switching, though that falls out of the same work).

## Phase 1 — transport-agnostic pipeline + FastAPI server + 3 clients

### Core refactor: split transport from pipeline

Two changes to `Agent` make it transport-agnostic:

```python
# agent.py
class Agent:
    def __init__(self, ..., audio_sink: AudioSinkBase | None = None):
        ...
        self.audio_out = audio_sink or LocalAudioSink(self.tts.sample_rate, ...)

    def feed_audio(self, frame: np.ndarray) -> None:
        """Replaces mic_callback's sounddevice-specific signature —
        any transport calls this with a mono float32 frame."""
        self.mic_q.put(frame)

    def start(self) -> list[threading.Thread]:
        """Split out of run(): spawn vad/worker/tts threads and return
        them, but don't touch any input device. run() (local mode) wraps
        this and additionally owns an sd.InputStream; the FastAPI
        endpoint (server mode) calls this directly per-connection."""
        threads = [
            threading.Thread(target=self.vad_thread, daemon=True),
            threading.Thread(target=self.worker_thread, daemon=True),
            threading.Thread(target=self.tts_thread, daemon=True),
        ]
        for t in threads:
            t.start()
        return threads

    def shutdown(self, threads: list[threading.Thread]) -> None:
        self.stop.set()
        self.cancel.set()
        for t in threads:
            t.join(timeout=1.0)
        self.audio_out.close()
```

`run()` (local mode) becomes a thin wrapper: call `start()`, open the
`sd.InputStream` with `callback=lambda i, f, t, s: self.feed_audio(i.flatten().astype(np.float32))`,
sleep-loop, then `shutdown()` + `report()` on Ctrl+C — this is exactly
what it does today, just factored so the FastAPI path can reuse
`start()`/`shutdown()` without an input device.

### `AudioSinkBase` — new interface, same shape `AudioOutput` already has

```python
# interfaces/audio_sink.py
class AudioSinkBase(ABC):
    """Receives synthesized audio for playback/delivery. Implementations
    own their own real-time pacing — push() queues audio, the
    implementation is responsible for draining it at the correct rate
    (a real output device does this via its callback; a websocket sink
    needs an explicit timer thread to do the same job)."""

    underruns: int

    @abstractmethod
    def push(self, audio: np.ndarray) -> None: ...
    @abstractmethod
    def flush(self) -> None: ...
    @abstractmethod
    def close(self) -> None: ...
    @property
    @abstractmethod
    def playing(self) -> bool: ...
    @property
    @abstractmethod
    def elapsed_ms(self) -> float: ...
```

- `LocalAudioSink` = today's `AudioOutput`, renamed, implementing this
  ABC explicitly. No behavior change.
- `WebSocketAudioSink(websocket)`: same `_buf`/`_lock`/fade/underrun
  logic as `AudioOutput._callback`, but instead of a PortAudio callback
  pulling frames on demand, a background thread wakes up every
  `blocksize / rate` seconds, pulls one block the same way, and does
  `asyncio.run_coroutine_threadsafe(websocket.send_bytes(...), loop)`.
  This reproduces `playing`/`elapsed_ms`/underrun semantics exactly, so
  `agent.py`'s barge-in logic (`self.audio_out.playing`,
  `self.audio_out.elapsed_ms`) needs **zero changes**.

Known limitation to document, not solve now: `EchoControl`'s `"aec"`
mode assumes mic and speaker are the same local device — it doesn't
make sense over a websocket where they're on a remote client.
Server-mode sessions should default `ECHO_MODE` to `"duck"` (mute VAD
while the bot is talking) rather than `"headphones"`/`"aec"`, since the
server has no way to verify the client has real headphone isolation.

### WebSocket protocol

One endpoint, `/ws`, binary frames for audio + a JSON control frame for
everything else (FastAPI's `WebSocket` supports both `receive_bytes`/
`send_bytes` and `receive_json`/`send_json` on the same connection):

| Direction | Type | Payload |
|---|---|---|
| client → server | binary | raw PCM16 mono 16kHz frame (matches `config.FRAME` size) |
| server → client | binary | raw PCM16 mono audio to play, chunked at the sink's blocksize |
| server → client | JSON | `{"event": "turn_start"}` / `{"event": "bot_text", "text": "..."}` / `{"event": "interrupted"}` — for UI feedback (captions, "thinking" indicator); optional for the CLI client, used by the HTML client |
| client → server | JSON | `{"event": "config", "trigger_word": "...", "tts": "kokoro", ...}` sent once right after connecting, before any audio — lets one server binary serve differently-configured sessions |

PCM16 chosen over float32 for the wire format specifically (half the
bytes vs. float32, no precision loss that matters for speech) — convert
to/from the pipeline's internal float32 at the transport edge only
(`frame.astype(np.float32) / 32768.0` in, `(audio * 32767).astype(np.int16)`
out), one line in each direction.

### Server: shared vs. per-session state (the resource-safety issue)

Naively building a fresh `Agent(...)` per connection would call
`OnnxAsrEngine()`/`KokoroTts()` per connection — reloading ~950MB +
~1.5GB of model weights **per concurrent user**, which doesn't scale
past a couple of sessions. `SttBase`/`TtsBase` implementations are
already fully stateless per call (no instance-held conversation state),
so ONNX Runtime sessions are safe to share and call concurrently from
multiple threads. Only `VadBase` (per-utterance start/end iterator
state) and the LLM's conversation history are inherently per-session.

This forces one more interface change, worth doing now rather than
retrofitting later: **`LlmBase.stream()` becomes stateless**, taking the
full message list instead of owning `self.history` internally —

```python
# interfaces/llm.py
class LlmBase(ABC):
    @abstractmethod
    def stream(self, messages: list[dict], cancel: threading.Event) -> Iterator[str]: ...
```

`OpenAiCompatibleLlm` drops `self.history`/warmup-per-instance; `Agent`
(or a small `Conversation` helper it owns) tracks `messages` per
session and passes the full list in. This also happens to be exactly
the shape LangChain / most chat-model APIs expect, so Phase 3 slots in
without touching this again.

Net server startup:

```python
# server.py (new)
stt = OnnxAsrEngine()          # loaded once
tts = KokoroTts()              # loaded once
llm = OpenAiCompatibleLlm()    # loaded once, no warmup-per-session, no owned history

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    agent = Agent(vad=SileroVad(...), stt=stt, tts=tts, llm=llm,
                   audio_sink=WebSocketAudioSink(websocket))
    threads = agent.start()
    try:
        while True:
            frame_bytes = await websocket.receive_bytes()
            agent.feed_audio(pcm16_to_float32(frame_bytes))
    except WebSocketDisconnect:
        pass
    finally:
        agent.shutdown(threads)
```

Each session still gets its own `SileroVad` instance (cheap — the model
is tiny, ~247MB RSS in the README's number is mostly PyTorch's own
import cost, paid once per process either way, not per instance) and its
own conversation history, while the expensive model weights are loaded
exactly once regardless of concurrent session count.

### Clients

- **HTML client** (`static/index.html`, served by FastAPI): Web Audio
  API (`AudioWorklet` or `ScriptProcessorNode`) captures mic frames,
  sends as binary WS frames; receives binary frames back and plays via
  an `AudioContext` buffer queue; listens for JSON events to show
  captions/status. No pipeline logic — a relay plus minimal UI.
- **CLI client** (new top-level script, e.g. `ws_client.py`): the exact
  mirror in Python — `sounddevice.InputStream` callback sends frames
  over `websockets`/`httpx-ws`, a receive loop plays incoming frames via
  `sounddevice.OutputStream`. Reuses none of `Agent` — it's a client, not
  a server component.
- **Local mode**: `main.py`, barely changed (see refactor above).

### File layout (new/changed)

```
main.py                          local mode (unchanged behavior, thin wrapper over Agent.start/shutdown)
server.py                        NEW — FastAPI app, /ws endpoint, serves static/index.html
ws_client.py                     NEW — CLI relay client
static/index.html                NEW — browser relay client
src/pos/
  agent.py                       feed_audio()/start()/shutdown() split out of mic_callback()/run()
  interfaces/
    audio_sink.py                NEW — AudioSinkBase
    llm.py                       CHANGED — stream(messages, cancel), no owned history
  audio/
    output.py                    AudioOutput -> LocalAudioSink(AudioSinkBase)
    ws_sink.py                   NEW — WebSocketAudioSink(AudioSinkBase)
  llm/openai_compatible.py       CHANGED — drop self.history/per-instance warmup
```

### Testing plan (Phase 1)

- Local mode: re-run today's manual smoke test (`uv run main.py`) to
  confirm zero behavior change after the `feed_audio`/`start`/`shutdown`
  split.
- Server mode, single session: `uv run server.py`, open the HTML client
  in a browser, confirm a full turn (speak → hear response) and barge-in
  both work identically to local mode.
- Server mode, concurrency: two browser tabs (or one browser + the CLI
  client) connected simultaneously, confirm each has independent
  conversation history and turn-taking, and that RSS stays close to the
  single-session baseline (proving shared STT/TTS instances, not
  per-session).
- CLI client against the same server the HTML client uses, confirming
  protocol parity.

## Phase 2 — provider abstraction hardening (spec'd separately, summarized here)

Builds directly on Phase 1's stateless `LlmBase`:

- Add a small provider registry per stage (name -> constructor), driven
  by config/CLI/env rather than hardcoded classes, e.g.
  `STT_PROVIDERS = {"parakeet-local": OnnxAsrEngine, "openai-cloud": OpenAiSttEngine}`.
- LLM cloud support needs **no new class** — `OpenAiCompatibleLlm`
  already works against real OpenAI, just `base_url="https://api.openai.com/v1"`
  and a real API key. The actual work is making that configurable and
  making sure the key comes from an environment variable
  (`OPENAI_API_KEY`, the SDK's own default lookup) — never hardcoded or
  passed on the CLI, per this org's data-handling policy.
- One new cloud `SttBase` implementation (e.g. OpenAI's transcription
  API) and one new cloud `TtsBase` implementation (e.g. OpenAI's TTS
  API) to prove the interface needs no changes for a cloud engine whose
  call is a network round-trip instead of a local ONNX session.

## Phase 3 — LLM harness / tool calling via LangChain (spec'd separately, summarized here)

Researched against LangChain's Python OSS docs (`docs.langchain.com/oss/python/langchain/`:
overview, agents, models, tools, messages, short-term-memory, streaming,
middleware, quickstart). Key finding: LangChain offers two integration
levels, and only one of them fits cleanly on top of what Phases 1/2
already build.

**Not used: `create_agent`'s own state ownership.** `create_agent(model=,
tools=, ...)` is convenient but owns conversation memory itself via
`AgentState` + a checkpointer + `thread_id` — a second, parallel
state-tracking mechanism alongside `Agent`'s own turn/cancel/barge-in
state and (per Phase 1) its own `self.conversation` history. Adopting it
would mean two systems tracking the same conversation. Not worth it for
what we need (a few tool calls mid-conversation, not a multi-agent
graph).

**Used instead: the model-level API.** `init_chat_model(model=...,
model_provider="openai", base_url=..., api_key=...)` returns a plain
chat model — a direct drop-in for today's `OpenAiCompatibleLlm` (same
three kwargs already point it at LM Studio; pointing at real cloud
OpenAI/Anthropic/etc. later is the same three kwargs, subsuming Phase
2's LLM-side provider story). `model.bind_tools([...])` adds tool
calling; the caller hand-rolls the loop: invoke, check
`response.tool_calls`, execute, append `ToolMessage`s, invoke again —
exactly the shape shown in LangChain's own "Tool Execution Loop"
example. Crucially, this loop takes a **plain external list of
`{"role", "content"}` dicts** each call — stateless, no framework-owned
history — which is exactly Phase 1 Task 5's `LlmBase.stream(messages,
cancel)` contract. No interface change needed for Phase 3.

**Where the tool loop lives:** entirely inside a new `LangChainLlm
(LlmBase)`'s `stream()` implementation — invisible to `Agent`/`respond()`,
which keeps calling `stream(messages, cancel)` and getting back plain
text pieces to speak, unchanged from today:

```python
def stream(self, messages, cancel):
    full = [{"role": "system", "content": self.system_prompt}] + messages
    for _ in range(self.max_tool_rounds):          # bounded — avoid an infinite tool-call loop
        if cancel.is_set():
            return
        response = self.model_with_tools.invoke(full)   # not streamed: no user-facing text yet
        if not response.tool_calls:
            break
        full.append(response)
        for call in response.tool_calls:
            full.append(self._tools[call["name"]].invoke(call))   # -> ToolMessage
    for chunk in self.model_with_tools.stream(full):    # final round: stream the spoken answer
        if cancel.is_set():
            return
        if chunk.text:
            yield chunk.text
```

Only the final, no-more-tool-calls round is streamed token-by-token
(what's actually spoken); intermediate tool-decision rounds are quick
blocking calls since they produce no speakable text anyway — this keeps
`agent.py`'s sentence-chunking/TTS flow completely untouched.

Open items for Phase 3's own design pass: which tools to expose first
(needs a product decision, not a technical one), `max_tool_rounds`'
value, and whether `langchain` + `langchain-openai` are worth the
dependency weight vs. hand-rolling the same loop directly against the
`openai` SDK's own `tools=` param (LangChain's main value-add here is
provider-agnostic model swapping, not the tool loop itself, which is
~10 lines either way).

## Risks / open questions

- Concurrent `InferenceSession.run()` calls on one shared ONNX Runtime
  session are documented as thread-safe, but latency under concurrent
  load hasn't been measured on the dev machine — worth a quick
  benchmark with 2-3 simultaneous sessions before assuming it scales
  fine.
- `WebSocketAudioSink`'s timer-driven pacing is new code (no direct
  precedent in this repo, unlike `LocalAudioSink` which is today's
  proven `AudioOutput`) — budget real testing time for it, particularly
  underrun behavior under network jitter (not just the local-loopback
  case).
- `"aec"` echo mode's server-mode story (ducking-only) needs a decision
  on whether that's acceptable long-term or whether a client-reported
  playback-position signal is needed for real echo cancellation later.

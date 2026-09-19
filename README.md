# pos-harness

Full-duplex local speech-to-speech voice agent: VAD -> STT -> LLM -> TTS,
with barge-in (interrupt the bot by talking over it) and per-turn latency
instrumentation printed live. Everything runs on CPU except the LLM, which
talks to any OpenAI-compatible server (developed against
[LM Studio](https://lmstudio.ai/)).

The repo is split into `backend/` (this Python voice pipeline + FastAPI
server) and `frontend/` (the browser client). See `docker-compose.yml` to
run backend + Postgres + frontend together, or run each piece directly as
described below.

Every stage — VAD, STT, LLM, TTS — sits behind a small abstract interface
(`backend/src/pos/interfaces/`), so any of them can be swapped for a different
implementation without touching the pipeline code in `agent.py`.

## Requirements

- **Python 3.14+**, [uv](https://docs.astral.sh/uv/) for dependency management
- **Windows** with a working audio input + output device (built on
  `sounddevice`/PortAudio; not tested on macOS/Linux)
- **[LM Studio](https://lmstudio.ai/)** (or any OpenAI-compatible chat
  completions server) running locally with a model loaded — this project
  does not bundle or manage the LLM itself
- No GPU required — VAD, STT, and TTS all run on CPU via ONNX Runtime

### RAM

Measured directly on the dev machine (i5-1235U, 12 logical cores) by
loading each component and reading the process's actual working set —
not estimated from model file sizes:

| stack loaded | peak RSS |
|---|---|
| Python + numpy baseline | ~14 MB |
| + VAD (Silero) | ~247 MB |
| + STT (Parakeet-TDT 0.6B, int8) | ~950 MB |
| + TTS: Kokoro (fp32, default) | ~1.5 GB |
| + TTS: Supertonic (instead of Kokoro) | ~1.5 GB (~1.47 GB) |

So budget **~1.5-2 GB RAM** for this process with either TTS engine
(VAD + STT + one TTS engine loaded at a time — the app never loads both).
Silero VAD pulls in PyTorch as a dependency even in ONNX mode, which is
most of the "+VAD" jump above.

This does **not** include LM Studio + your chosen LLM model, which runs as
a separate process with its own RAM budget — check that model's card in
LM Studio (a small model like `lfm2.5-230m` needs comparatively little;
budget accordingly for whatever you actually load).

### Disk

| item | size |
|---|---|
| Python venv (`uv sync`) | ~820 MB (mostly PyTorch, pulled in by Silero VAD) |
| STT: Parakeet-TDT 0.6B (int8), auto-downloaded | ~640 MB |
| TTS: Kokoro fp32 (default), auto-downloaded | ~380 MB |
| TTS: Kokoro int8 (alternative, not recommended — see below) | ~140 MB |
| TTS: Supertonic (optional, manual setup — see below) | ~400 MB |

Total for the default stack (Kokoro): **~1.8 GB**. Add ~400 MB if you also
set up Supertonic.

## Install

```
cd backend
uv sync
```

This creates `backend/.venv/` and installs everything, including this project
itself (editable), so the `pos-agent`/`pos-server`/`pos-client` console
scripts and `src/pos/cli/local.py` can `from pos... import ...`. All `uv
run`/`uv sync` commands below are run from `backend/`.

You'll also need a Postgres database for session history — see
"Running with Docker Compose" below, or point `DATABASE_URL` at your own
instance (defaults to `postgresql://pos:pos@localhost:5432/pos`).

No LLM endpoint is assumed. To use a local model, start LM Studio (or any
OpenAI-compatible server), load a model, start its server, and give the
backend its URL — `LOCAL_BASE_URL=http://localhost:1234/v1`, or per user in
the web UI's Settings. Until then (and with no provider API key), sessions
are refused with "no LLM configured".

### Optional: Supertonic TTS

Kokoro downloads automatically on first run. Supertonic is an alternative
TTS engine that needs a one-time manual checkout (same layout as the
[Supertone/supertonic-3](https://huggingface.co/Supertone/supertonic-3)
Hub repo):

```
git clone https://huggingface.co/Supertone/supertonic-3 assets
```

`SupertonicTts` auto-detects `./assets` at startup and skips the network
entirely if it's present (falls back to downloading via `huggingface_hub`
otherwise). `assets/` is gitignored — it's ~400 MB with its own nested
`.git`/LFS history and should never be committed to this repo.

### Optional: trigger word

Off by default (respond to everything, as today). When set via
`--trigger-word`, the agent still runs VAD + STT on every utterance
exactly as it does today — nothing changes in the audio pipeline — but
after transcription it checks whether the trigger phrase (case-insensitive,
matched on a word boundary so `"computer"` doesn't match `"computers"`)
appears within the transcript's first few words (`config.TRIGGER_LOOKAHEAD_WORDS`,
default 2 words of tolerance beyond the trigger phrase itself) — not
strictly at position 0, since STT commonly transcribes a beat of
"uh"/"hey"/"okay" before the real trigger word, and requiring an exact
prefix drops genuine commands. If the trigger doesn't appear in that
leading window, the turn is silently dropped (logged, no LLM/TTS call)
exactly like today's empty-transcript drop path. If it does, the full,
unmodified STT transcript (trigger word included) is sent to the LLM —
the trigger phrase is only used to decide whether to respond at all, not
edited out of what the LLM sees:

```
uv run pos-agent --trigger-word "computer"
```

Say "computer, what time is it" and the LLM receives the full "computer,
what time is it"; say anything not leading with "computer" and the bot
stays silent. No separate audio model, no training, no threshold to
tune — it's a plain prefix check on the STT output you already have.

## Usage

```
uv run pos-agent                              # default: Kokoro, voice af_bella
uv run pos-agent --tts supertonic             # use Supertonic instead
uv run pos-agent --tts supertonic --voice M1  # + pick its voice
uv run pos-agent --tts kokoro --voice af_sky  # or a different Kokoro voice

# trigger word (see "Optional: trigger word" above for setup)
uv run pos-agent --trigger-word "computer"
```

Speak any time, including while the bot is talking (barge-in cancels its
current response and starts listening to you). Ctrl+C stops cleanly and
prints a session summary:

```
--- session summary ---
  stt            avg=0.32s  min=0.18s  max=0.49s  n=9
  llm_ttft       avg=2.14s  ...
  llm_total      avg=2.90s  ...
  tts_synth      avg=0.73s  ...
  tts_rtf        avg=0.99x  ...          # <1.0 = synthesis faster than realtime
  ttfa           avg=3.31s  ...          # end-of-speech to first audio out
  interruptions  0
  underruns      0                        # TTS fell behind playback mid-turn
```

## Running: local vs. server

Three ways to run this, all sharing the same VAD/STT/LLM/TTS pipeline:

1. **Local** (`uv run pos-agent`) — everything in one process, direct
   `sounddevice` mic/speaker access. No network involved. This is the
   original mode and still the simplest for single-user local use.
2. **FastAPI server + browser client** (`uv run pos-server`, then open
   `http://localhost:8000/`) — the pipeline runs server-side; the
   browser captures your mic and plays responses via a websocket at
   `/ws`. Supports multiple simultaneous browser tabs/users, each with
   independent conversation history.
3. **FastAPI server + CLI client** (`uv run pos-server`, then in another
   terminal `uv run pos-client`) — same server, a Python relay client
   instead of a browser. Useful for scripting/headless use, or testing
   the server without a browser.

Modes 2 and 3 share 100% of the server-side code — the only difference
is which relay client captures/plays your audio. Audio crosses the
websocket as raw PCM16 mono frames (16kHz client→server, matching the
TTS engine's own sample rate server→client) — no codec, matching what
the pipeline already uses internally.

Every mode assumes real headphones — no echo suppression, no mic
muting, full barge-in always on. If the mic can actually hear the
speakers (laptop speakers instead of a real headset, or a remote client
without headphones), the bot's own voice will be picked up as false
speech; there's no fallback mode for that anymore, so use real
headphones with any of the three run modes.

`fastapi`, `uvicorn[standard]`, and `websockets` are dependencies added
for server mode; `pytest` is a dev-only dependency (`uv run pytest` to
run the test suite) — neither is needed just to run `pos-agent`.

### Browser client: model/provider selection

The browser client (`http://localhost:8000/`) has a config panel — TTS
engine + voice, LLM model (populated live from LM Studio's own
`/v1/models`), trigger word, and input microphone — picked before
clicking Connect. Changing any of these requires disconnecting and
reconnecting; there's no live mid-session reconfiguration. A live
transcript pane shows what you said and what the bot replied as the
conversation happens.

The server (`GET /options`) lazily constructs and caches one TTS
instance per distinct `(engine, voice)` combination actually requested,
and one LLM client per distinct model name, sharing each across every
session that asks for the same combination — picking N different
voices across a server's lifetime costs roughly N × ~1.5GB RAM (each
`KokoroTts`/`SupertonicTts` instance loads its own model weights), so
that's a real resource tradeoff to be aware of on a long-running server
with many different voices requested, not something this bounds
automatically.

### Mic selection and mute

All three run modes support picking a specific input device:

```
uv run pos-agent --list-mics              # print available input devices
uv run pos-agent --mic "USB"               # by name substring, or an index
uv run pos-client --list-mics
uv run pos-client --mic 3
```

The browser client has an equivalent microphone dropdown (populated via
`navigator.mediaDevices.enumerateDevices()` — device labels only appear
after mic permission has been granted once).

**Mute**: in `pos-agent`/`pos-client`, press Enter in the terminal to
toggle muting (frames are dropped before they ever reach VAD, so the
agent stays idle) — press Enter again to unmute. In the browser, a Mute
button next to Connect does the same, and also disables the mic track
so the browser's own hardware-in-use indicator turns off.

### CLI client provider selection (`pos-client`)

```
uv run pos-client --tts supertonic --voice M1 --llm-model gemma-3-1b-it
uv run pos-client --trigger-word "computer"
```

These are forwarded to `pos-server` as the same `/ws` query params the
browser client uses — see `backend/src/pos/cli/server.py`'s own module
docstring for the full query-param reference.

### VAD tuning

All three run modes let you tune Silero VAD's sensitivity and
turn-taking without touching code:

```
uv run pos-agent --vad-threshold 0.35        # lower = more sensitive (default: 0.5)
uv run pos-agent --vad-min-silence-ms 800    # shorter pause before a turn ends (default: MIN_SILENCE_MS)
uv run pos-agent --vad-speech-pad-ms 200     # less padding kept around detected speech (default: SPEECH_PAD_MS)

uv run pos-client --vad-threshold 0.35 --vad-min-silence-ms 800
```

forwarded to `pos-server` as `vad_threshold`/`vad_min_silence_ms`/
`vad_speech_pad_ms` query params (same as above); the browser client has
matching "vad sensitivity"/"pause length"/"speech padding" fields in its
config panel. `vad_threshold` must be in `[0, 1]`; all three are
validated before the session starts — an invalid value gets an `error`
event and the socket is closed rather than silently falling back.

### LLM: LangChain + tool calling

The LLM stage (`backend/src/pos/llm/langchain_llm.py`, `LangChainLlm`) runs
on `langchain` + `langchain-openai`'s `ChatOpenAI` instead of talking to
the OpenAI SDK directly — same LM Studio (or any OpenAI-compatible)
backend, no new server required. `LlmBase.stream(messages, cancel)`'s
contract is unchanged, so `Agent`, `pos.cli.server`, and every
other caller needed no changes.

This unlocks tool calling: the model can call a bound tool
mid-conversation and get its result folded back in before finishing its
answer. The tool-execution loop (invoke, check `tool_calls`, run the
tool, append its result, invoke again — bounded by
`max_tool_rounds=3`) is hand-rolled per LangChain's own documented "Tool
Execution Loop" pattern, not `create_agent` — that owns its own
conversation memory (`AgentState` + a checkpointer), which would
duplicate the history `Agent` already tracks in `self.conversation`.

Tools bound by default (`backend/src/pos/llm/tools.py`):
- **`get_current_time`** — always on, no setup. A voice assistant with
  no sense of the current date/time is asked about it constantly.
- **web search** (via [Tavily](https://tavily.com/), free tier available)
  — only enabled if `TAVILY_API_KEY` is set in the environment; skipped
  (with a startup log line) otherwise, so search being unconfigured
  doesn't stop the agent from starting.

```
# PowerShell
$env:TAVILY_API_KEY = "tvly-..."
uv run pos-agent

# bash
export TAVILY_API_KEY=tvly-...
uv run pos-agent
```

Pass a custom `tools=[...]` list to `LangChainLlm(...)` to add more —
any LangChain `BaseTool` (including a plain `@tool`-decorated function)
works.

**Backend selection** (env vars, decided once per run — no in-session picker):

| Backend | Selected by | Other vars |
|---|---|---|
| Local | `LOCAL_BASE_URL` (no default) | `LOCAL_MODEL` (optional; otherwise the first model the server reports), `LOCAL_API_KEY` |
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

## Latency

Real numbers, not estimates — aggregated from actual `TTFA = stt + llm_ttft
+ tts + other` breakdowns printed across several live sessions on the dev
machine (i5-1235U), `--tts supertonic --voice F3`, LM Studio serving
`lfm2.5-230m` locally. 9 completed turns:

| component | avg | range | share of TTFA |
|---|---|---|---|
| stt (Parakeet-TDT 0.6B int8) | 0.32s | 0.18-0.49s | ~10% |
| llm_ttft (time to first token) | 2.14s | 0.64-2.41s | ~65% |
| tts (first chunk synth) | 0.73s | 0.62-1.09s | ~22% |
| other (scheduling/overhead) | 0.12s | 0.09-0.15s | ~4% |
| **TTFA (end-of-speech to first audio)** | **3.31s** | **1.71-4.13s** | |

**`llm_ttft` dominates** — it's ~65% of the wait before you hear anything,
and it's LM Studio-side latency, not something this project's code
controls. 2+ seconds is actually abnormal for a model this small
(`lfm2.5-230m`) — see "Reducing `llm_ttft`" below before assuming you need
a different model; the likely cause is GPU offload being off and/or
threads spanning E-cores, not the model itself.

`stt` and `llm_ttft` are the same regardless of which TTS engine you pick
— only the `tts` component changes. Kokoro's better RTF (see below) mostly
shows up on *later* chunks in a multi-sentence response, not the first
chunk shown here — the first chunk is kept short on purpose
(`FIRST_CHUNK_CHARS=25`) specifically to minimize this number, so the gap
between engines on it is smaller than their RTF difference would suggest.

`tts_rtf` (synthesis time / audio duration, all chunks, not just the
first) is the more honest per-engine comparison — see the TTS engine
choice table further down.

### Reducing `llm_ttft`

2+ seconds to first token is abnormal for a 230M-parameter model — that
class of model should hit well under 500ms on CPU with correct threading.
This isn't code this repo controls (LM Studio is an external app), but
it's worth fixing since it's ~65% of what you're waiting on. In priority
order:

1. **Enable Vulkan iGPU offload.** LM Studio has supported Vulkan offload
   for Intel integrated graphics since v0.4.17, but it ships **disabled by
   default**. Settings → Runtime → switch engine to "Vulkan llama.cpp" →
   load the model → gear icon next to it → GPU Offload slider to Max. A
   230M model is trivial for an Iris Xe iGPU and this sidesteps the
   CPU-scheduling problem below entirely — likely the single biggest win
   available.
2. **If staying CPU-only, pin threads to physical P-cores.** 12th-gen
   Intel hybrid CPUs (P-cores + E-cores) have a
   [documented llama.cpp issue](https://github.com/ggml-org/llama.cpp/discussions/572)
   where Windows' scheduler scatters inference threads onto E-cores,
   measured at ~2.4x slower than P-core-only (482ms -> 202ms/eval on a 7B
   model). Find LM Studio's thread-pool/CPU-threads setting (per-model
   inference config, usually defaults to "Auto") and set it explicitly to
   your P-core count instead of leaving it to span all logical cores. This
   is the *opposite* tuning direction from this repo's own TTS engines —
   see the "TTS engine choice" section further down for why more threads
   helped onnxruntime's int8 kernels but hurts llama.cpp here; they're
   different engines with different optimal configs on the same CPU,
   don't conflate them.
3. **Enable Flash Attention** if the toggle is available for the loaded
   model — reduces memory traffic during prefill, which is exactly the
   phase `llm_ttft` measures.
4. Confirm the model isn't being unloaded between idle periods (adds a
   reload spike on the next turn) — less likely to be the cause here since
   the delay was consistently ~2s across turns rather than a one-time
   spike, but cheap to rule out.

Change one thing at a time and re-run — `report()`'s `llm_ttft` line is
the feedback loop, no separate benchmarking needed. Observed variance
between two otherwise-identical live sessions was 2.39s vs 1.51s with no
settings changed between them, so don't over-read a single before/after
comparison; look for a consistent shift across several turns.

## Configuration

Shared, engine-agnostic tuning lives in `backend/src/pos/config.py`. Notable
knobs:

- **`MIN_SILENCE_MS`** (default `1200`) — how long a pause must last before
  VAD considers your turn finished. Lower = snappier turn-taking but risks
  cutting off mid-sentence pauses (fragmenting one utterance into several
  short segments, which are more prone to the STT engine returning an
  empty transcript — see `agent.py`'s comment on this constant for the
  reasoning). Higher = more tolerant of hesitation, but adds latency after
  you stop talking.
- **`BARGE_IN_FRAMES`** / **`BARGE_IN_GRACE_MS`** — how many consecutive
  VAD-active frames (and how long after playback starts) before an
  in-progress response is interrupted.
- **`TRIGGER_LOOKAHEAD_WORDS`** (default `2`) — only matters if
  `--trigger-word` is set. How many leading words of STT filler
  ("uh"/"hey"/"okay") to tolerate before the trigger phrase — see
  "Optional: trigger word" above for the false-trigger trade-off of
  raising it.
- **`HISTORY_TURNS`** (default `3`) — how many prior user/assistant turn
  pairs `Agent` includes as LLM context. Previously owned by the LLM
  engine itself; moved to `Agent` so the LLM engine can be a single
  shared, stateless instance across concurrent server sessions.
- **`FIRST_CHUNK_CHARS`** / **`MAX_CHUNK_CHARS`** — bound how much text is
  handed to the TTS engine per chunk. Kept small for the first chunk to
  minimize time-to-first-audio; capped thereafter so no single TTS call
  takes disproportionately long regardless of how the LLM streams (a
  single large delta from the LLM server is bounded the same as
  token-by-token streaming — see `respond()`'s `drain()`).

Engine-specific settings (model repo, voice, thread count, LLM base
URL/model name, etc.) are constructor defaults on each concrete
implementation instead, so swapping an engine doesn't require touching
`config.py`:

- `backend/src/pos/vad/silero.py` — `SileroVad`
- `backend/src/pos/stt/onnx_asr_engine.py` — `OnnxAsrEngine`
- `backend/src/pos/tts/kokoro.py` — `KokoroTts`
- `backend/src/pos/tts/supertonic.py` — `SupertonicTts`
- `backend/src/pos/llm/langchain_llm.py` — `LangChainLlm` (base URL, API
  key, model name, system prompt, tools, `max_tool_rounds` — this is
  where you'd point at a different LM Studio model or port;
  conversation history length is `config.HISTORY_TURNS`, owned by
  `Agent` instead — see Configuration above and "LLM: LangChain + tool
  calling" below)

### TTS engine choice

Benchmarked on the dev machine (i5-1235U — 12th-gen Intel mobile, no
AVX-512/VNNI, so results won't transfer directly to other CPUs; re-run
your own comparison before trusting this elsewhere):

| engine | RTF (lower is faster) |
|---|---|
| Kokoro, fp32, 8 threads (default) | ~0.5-0.8 |
| Kokoro, int8 | ~1.5 — **slower** than fp32 here; CPUs without VNNI get no hardware acceleration for quantized ops, so int8 is a pessimization, not an optimization |
| Supertonic, 8 threads | ~1.0-1.1 |

Kokoro fp32 is the faster, better-tuned default. Supertonic is available
as a drop-in alternative (different voices/prosody) via `--tts supertonic`.

## Architecture

```
backend/src/pos/
  cli/
    local.py                     local-mode CLI entrypoint (--tts, --voice, --trigger-word, --vad-*)
                                  — console script: `uv run pos-agent`
    server.py                    FastAPI multi-session websocket server (see "Running" above)
                                  — console script: `uv run pos-server`
    relay_client.py              Python CLI relay client for server.py
                                  — console script: `uv run pos-client`
  config.py                      shared, engine-agnostic settings
  utils.py                       resample_linear, pcm16_to_float32, float32_to_pcm16
  agent.py                       orchestrator: threads + queues wiring; feed_audio/on_text_message/
                                  start/shutdown are the transport-agnostic entry points
                                  cli/local.py and cli/server.py both drive
  storage.py                     SessionStore — Postgres session/turn history (create/add_turn/
                                  set_title/delete_session/list_sessions/get_session)
  null_engines.py                NullTts/NullVad — no-op stand-ins for text-mode sessions; kept
                                  outside tts/ and vad/ on purpose, see the file's own docstring
  interfaces/
    vad.py, stt.py, tts.py, llm.py, audio_sink.py   abstract base classes (the swap contracts)
  audio/
    output.py                    LocalAudioSink(AudioSinkBase) — playback ring buffer, click-free underrun handling
    ws_sink.py                   WebSocketAudioSink(AudioSinkBase) — same contract, paced by a timer thread instead of a device callback
    null_sink.py                 NullAudioSink(AudioSinkBase) — no-op, for text-mode sessions
  vad/silero.py                  SileroVad(VadBase)
  stt/onnx_asr_engine.py         OnnxAsrEngine(SttBase)
  tts/kokoro.py                  KokoroTts(TtsBase)
  tts/supertonic.py              SupertonicTts(TtsBase)
  llm/langchain_llm.py           LangChainLlm(LlmBase) — bound tools, hand-rolled tool loop,
                                  usage/cost/context-window reporting, generate_title()
  llm/providers/                 provider registry — base.py (LlmProviderBase ABC + ProviderConfig),
                                  registry.py (dispatch), local.py/openai.py/azure.py (one
                                  self-registering provider each; add a new backend by adding
                                  one file here, no other file needs to change)
  llm/tools.py                   get_current_time, web search (Tavily, needs TAVILY_API_KEY)
tests/                           pytest suite (fakes.py holds shared no-hardware/no-network test doubles); run with `uv run pytest`
```

### Code standards

`uv run ruff check .` is the lint gate — unused imports, unsorted imports,
deprecated syntax, and common bug patterns (e.g. `zip()` without
`strict=`). Config lives in `pyproject.toml`'s `[tool.ruff]` (line length
is a generous 120, matching this codebase's existing style of long
explanatory inline comments). `ruff format` is intentionally **not**
enforced — it would reformat nearly every file to its own opinionated
line-wrapping style with no correctness benefit, on top of formatting
Python code fences inside `docs/*.md`, which isn't wanted for planning
documents. Run it manually (`uv run ruff format --diff src tests *.py`,
excluding `docs/`) only if you want to see what it would change.

`Agent` is built by dependency injection —
`Agent(vad=..., stt=..., tts=..., llm=..., trigger_word=..., audio_sink=..., on_event=...)`
— defaulting to the concrete classes above (`trigger_word` defaults to
`None`, feature off; `audio_sink` defaults to `LocalAudioSink`;
`on_event` defaults to a no-op, called with `("user_text"|"bot_text"|
"interrupted", data)` for UI captions). `agent.muted` (a
`threading.Event`) gates `feed_audio()` — set it to drop incoming
frames before they reach VAD, for any transport. To add a new engine,
implement the matching interface (usually just one `__call__`/`stream`
method) and pass an instance in; no changes to `agent.py` needed.
`LlmBase.stream(messages, cancel)` is stateless — it
takes the full prior-turns message list each call rather than owning
conversation history itself, so one `LlmBase` instance can be shared
across every concurrent session in server mode (`Agent` owns the
history, in `self.conversation`).

Pipeline stages run as separate threads connected by queues (mirroring how
[huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech)
structures its own VAD/STT/LLM/TTS handlers), so a slow stage doesn't
stall the others — most importantly, TTS synthesis for one sentence
overlaps with LLM generation of the next one instead of blocking it.

## Known limitations

- Real headphones (mic genuinely cannot hear the speakers) are assumed
  unconditionally — there's no echo suppression or mic-muting fallback
  mode. Without real isolation, the bot's own voice gets picked up as
  false speech, degrading VAD/STT/turn-taking.
- The STT engine (Parakeet-TDT 0.6B int8) occasionally returns an empty
  transcript for short (~2-3s) segments even when they contain real
  speech — confirmed by feeding known-good audio through it directly, not
  just observed live. `MIN_SILENCE_MS` is tuned to reduce how often short
  fragments get created in the first place, but this isn't eliminated.
- Playback starts immediately with no pre-buffer, matching the reference
  implementation's own approach — an underrun is possible if TTS falls
  behind, though it's now handled as a clean fade rather than a click (see
  `underruns` in the session summary). A startup pre-buffer would reduce
  frequency further at the cost of higher time-to-first-audio; not added,
  since that's a latency/smoothness tradeoff rather than a bug.
- Thread counts baked into `KokoroTts`/`SupertonicTts`/`OnnxAsrEngine`
  defaults are tuned for the dev machine specifically — re-benchmark on
  different hardware rather than trusting them as universal.
- `--trigger-word` is a plain text check, not a fuzzy/phonetic match —
  it won't catch a transcript the STT engine got slightly wrong (e.g. a
  homophone, or "computers" instead of "computer" due to the
  word-boundary match). It also only looks within the transcript's
  first few words (`config.TRIGGER_LOOKAHEAD_WORDS`), so it won't fire on
  a sentence that merely *mentions* the trigger word later on (e.g. "tell
  me about the computer") — by design, but it also means more than ~2
  leading filler words before the trigger will cause a genuine command to
  be dropped; raise `TRIGGER_LOOKAHEAD_WORDS` if that happens often in
  your own speech.

## Running with Docker Compose

Copy `.env.example` to `.env` and fill in `JWT_SECRET` and `ENCRYPTION_KEY`
(both `openssl rand -base64 32`). The stack will not start without them.
Losing `ENCRYPTION_KEY` makes every stored API credential unrecoverable.

```
docker compose up --build
```

Starts three services:

- **postgres** — Postgres 16, database `pos`, user/password `pos`/`pos`
  (override via `.env`, see `docker-compose.yml`), data persisted in the
  `pgdata` named volume
- **backend** — the FastAPI server (`backend/Dockerfile`), on
  `http://localhost:8000`, with `DATABASE_URL` pointed at the `postgres`
  service
- **frontend** — the Next.js browser client (`frontend/Dockerfile`), on
  `http://localhost:3000`, with `NEXT_PUBLIC_API_URL`/`NEXT_PUBLIC_WS_URL`
  pointed at the backend service

The `backend` image does not bundle the STT/TTS model weights (~1GB+).
Compose mounts `./models` (Hugging Face and NLTK caches) and `./assets`
(Supertonic) into the container, so they download once on first run and
survive container recreation. Both are gitignored.

No LLM endpoint is assumed. LM Studio (or another OpenAI-compatible server)
runs on the host, outside Compose; give the backend its URL with
`LOCAL_BASE_URL` (from inside the container that is
`http://host.docker.internal:1234/v1`), or let each user enter theirs in
Settings.

The Postgres port is published on `127.0.0.1` only, and backend tests
refuse to run against any database not named `*_test` (default `pos_test`;
create it with `docker compose exec postgres psql -U pos -d postgres -c
'CREATE DATABASE pos_test'`).

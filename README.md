# asr-test

Full-duplex local speech-to-speech voice agent: VAD -> STT -> LLM -> TTS,
with barge-in (interrupt the bot by talking over it) and per-turn latency
instrumentation printed live. Everything runs on CPU except the LLM, which
talks to any OpenAI-compatible server (developed against
[LM Studio](https://lmstudio.ai/)).

Every stage — VAD, STT, LLM, TTS — sits behind a small abstract interface
(`src/asr_test/interfaces/`), so any of them can be swapped for a different
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
uv sync
```

This creates `.venv/` and installs everything, including this project
itself (editable), so `main.py` can `from asr_test... import ...`.

Then start LM Studio, load a model, and start its local server (default
`http://localhost:1234/v1` — matches `OpenAiCompatibleLlm`'s default; see
Configuration below if yours differs).

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
uv run main.py --trigger-word "computer"
```

Say "computer, what time is it" and the LLM receives the full "computer,
what time is it"; say anything not leading with "computer" and the bot
stays silent. No separate audio model, no training, no threshold to
tune — it's a plain prefix check on the STT output you already have.

## Usage

```
uv run main.py                              # default: Kokoro, voice af_bella
uv run main.py --tts supertonic             # use Supertonic instead
uv run main.py --tts supertonic --voice M1  # + pick its voice
uv run main.py --tts kokoro --voice af_sky  # or a different Kokoro voice

# trigger word (see "Optional: trigger word" above for setup)
uv run main.py --trigger-word "computer"
```

Speak any time, including while the bot is talking (barge-in cancels its
current response and starts listening to you). Ctrl+C stops cleanly and
prints a session summary:

```
--- session summary ---
  echo mode      headphones (barge-in on)
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

1. **Local** (`uv run main.py`) — everything in one process, direct
   `sounddevice` mic/speaker access. No network involved. This is the
   original mode and still the simplest for single-user local use.
2. **FastAPI server + browser client** (`uv run server.py`, then open
   `http://localhost:8000/`) — the pipeline runs server-side; the
   browser captures your mic and plays responses via a websocket at
   `/ws`. Supports multiple simultaneous browser tabs/users, each with
   independent conversation history.
3. **FastAPI server + CLI client** (`uv run server.py`, then in another
   terminal `uv run ws_client.py`) — same server, a Python relay client
   instead of a browser. Useful for scripting/headless use, or testing
   the server without a browser.

Modes 2 and 3 share 100% of the server-side code — the only difference
is which relay client captures/plays your audio. Audio crosses the
websocket as raw PCM16 mono frames (16kHz client→server, matching the
TTS engine's own sample rate server→client) — no codec, matching what
the pipeline already uses internally.

Server mode defaults each session's echo handling to `"duck"`
(`?echo_mode=duck`, or omit it), regardless of `config.ECHO_MODE`'s own
module-level default — the server has no way to verify a remote client
has real headphone isolation, and `"headphones"` mode assumes that (it
would otherwise mistake the bot's own voice for your speech). If you
genuinely are on headphones, pass `?echo_mode=headphones` (browser: the
"echo handling" dropdown; CLI: `--echo-mode headphones`) for full
barge-in instead of the mic being cut while the bot talks — an invalid
value gets an `"error"` event and the connection is closed rather than
silently falling back to something else.

**Why this matters for audio quality, not just barge-in**: `"duck"`
mode discards mic frames entirely (not just attenuates them) while the
bot's response is still draining, so starting to talk again *before*
the bot fully finishes can truncate the start of your next utterance —
often showing up as an empty or garbled STT transcript, not a genuine
STT failure. This is expected behavior for `"duck"`'s safety tradeoff,
not a bug; `"headphones"` mode doesn't have this limitation, but only
actually helps if you're wearing real headphones (it does zero echo
suppression, so on open speakers it just lets the bot's own voice back
into the mic as false speech instead).

**On loud speakers with no headset mic** (loud/noisy room, laptop
speakers, no headphones): `?echo_mode=aec` is the one built for exactly
this — active echo cancellation instead of muting, so the mic stays
live and isolates your voice from the bot's own output rather than
either dropping your speech (`"duck"`) or picking up the bot's voice as
false input (`"headphones"` without real isolation). Needs `voiceclean`
installed — see the `ECHO_MODE` entry under Configuration below,
including the caveat that this project's `voiceclean` integration is
fixed but not yet verified against the real package with live audio.

`fastapi`, `uvicorn[standard]`, and `websockets` are dependencies added
for server mode; `pytest` is a dev-only dependency (`uv run pytest` to
run the test suite) — neither is needed just to run `main.py`.

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
uv run main.py --list-mics              # print available input devices
uv run main.py --mic "USB"               # by name substring, or an index
uv run ws_client.py --list-mics
uv run ws_client.py --mic 3
```

The browser client has an equivalent microphone dropdown (populated via
`navigator.mediaDevices.enumerateDevices()` — device labels only appear
after mic permission has been granted once).

**Mute**: in `main.py`/`ws_client.py`, press Enter in the terminal to
toggle muting (frames are dropped before they ever reach VAD, so the
agent stays idle) — press Enter again to unmute. In the browser, a Mute
button next to Connect does the same, and also disables the mic track
so the browser's own hardware-in-use indicator turns off.

### CLI client provider selection (`ws_client.py`)

```
uv run ws_client.py --tts supertonic --voice M1 --llm-model gemma-3-1b-it
uv run ws_client.py --trigger-word "computer"
```

These are forwarded to `server.py` as the same `/ws` query params the
browser client uses — see `server.py`'s own module docstring for the
full query-param reference.

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

Shared, engine-agnostic tuning lives in `src/asr_test/config.py`. Notable
knobs:

- **`ECHO_MODE`** (`"headphones" | "duck" | "aec"`) — `"headphones"` does
  zero echo suppression and assumes the mic genuinely cannot hear the
  speakers. If you're on laptop speakers/mic instead of real headphones,
  `"aec"` (active echo cancellation) is the one that keeps full barge-in
  and doesn't cut off the start of what you say while the bot is still
  talking, unlike `"duck"` (mutes/drops mic input entirely while the bot
  talks — the safest default, but in a loud room this can truncate your
  next sentence if you start before the bot fully finishes, which shows
  up as an empty or garbled STT transcript, not an STT bug). `"aec"`
  needs the optional `voiceclean` package (`uv add voiceclean` — pulls in
  `soxr`; `numpy`/`onnxruntime` are already dependencies here) and falls
  back to `"duck"` if it's not installed. Using `"headphones"` without
  actual headphone isolation causes the bot's own voice to be picked up as
  false speech. `Agent(echo_mode=...)` overrides this per-instance (what
  `--echo-mode`/`?echo_mode=` actually set) without touching the module
  default other callers still see.
  **Verified with the real package installed** (not just against docs):
  `voiceclean`'s `process()` doesn't return exactly one frame_size chunk
  per call — it buffers internally on its own frame size and emits
  variable-length (sometimes empty) output, confirmed live feeding
  512-sample frames and getting back 0 or 640 samples, never 512.
  `EchoControl` buffers and re-chunks this to exactly `frame_size` per
  call (same pattern as `WebSocketAudioSink`'s own drain logic) — a
  second real bug beyond the original API-shape mismatch, caught by
  actually running it rather than trusting the docs alone.
  **Do not install `pyrnnoise`** (voiceclean's optional noise-suppression
  extra) — as of `voiceclean==0.3.6` + `pyrnnoise==0.4.3`, enabling it
  makes every `process()` call raise internally (`Graph.__init__() got
  an unexpected keyword argument 'rate'`, an upstream version mismatch
  between the two packages), which `EchoControl` catches safely but then
  silently no-ops AEC on every frame — worse than not installing it at
  all, since you get raw mic passthrough with no indication anything's
  wrong beyond a `[aec failed: ...]` log line per frame. Plain
  `voiceclean` (AEC only, no noise suppression) works correctly.
  **On speakers, VAD may still fire on long/garbled segments even with
  `"aec"` active** — this is residual echo the canceller didn't fully
  suppress, not a VAD bug (it doesn't happen on `"headphones"`, which has
  no echo to begin with). `voiceclean`'s AEC works by correlation-based
  suppression (attenuate a chunk once it correlates with the reference
  above a threshold), confirmed against the real installed
  `voiceclean.aec.AEC` class — not classic adaptive-filter cancellation.
  Its own docs recommend lowering the 0.15 default `correlation_threshold`
  toward 0.10 for exactly this kind of challenging echo (their own
  example: PSTN telephony), so this project ships `0.10` as its own
  default rather than requiring you to discover it (see
  `_AEC_CORRELATION_THRESHOLD` in `src/asr_test/audio/echo.py`). Still not
  independently verified with real speech — if residual echo keeps
  triggering VAD, try lowering it further (`0.08` per voiceclean's docs)
  before assuming `"aec"` doesn't work for your setup.
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

- `src/asr_test/vad/silero.py` — `SileroVad`
- `src/asr_test/stt/onnx_asr_engine.py` — `OnnxAsrEngine`
- `src/asr_test/tts/kokoro.py` — `KokoroTts`
- `src/asr_test/tts/supertonic.py` — `SupertonicTts`
- `src/asr_test/llm/openai_compatible.py` — `OpenAiCompatibleLlm` (base
  URL, API key, model name, system prompt — this is where you'd point at
  a different LM Studio model or port; conversation history length is
  `config.HISTORY_TURNS`, owned by `Agent` instead — see Configuration
  above)

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
main.py                          local-mode CLI entrypoint (--tts, --voice, --trigger-word)
server.py                        FastAPI multi-session websocket server (see "Running" above)
ws_client.py                     Python CLI relay client for server.py
static/index.html                browser relay client for server.py
src/asr_test/
  config.py                      shared, engine-agnostic settings
  utils.py                       resample_linear, pcm16_to_float32, float32_to_pcm16
  agent.py                       orchestrator: threads + queues wiring; feed_audio/start/shutdown
                                  are the transport-agnostic entry points main.py and server.py
                                  both drive
  interfaces/
    vad.py, stt.py, tts.py, llm.py, audio_sink.py   abstract base classes (the swap contracts)
  audio/
    output.py                    LocalAudioSink(AudioSinkBase) — playback ring buffer, click-free underrun handling
    ws_sink.py                   WebSocketAudioSink(AudioSinkBase) — same contract, paced by a timer thread instead of a device callback
    echo.py                      EchoControl — headphones/duck/aec modes, barge-in gating
  vad/silero.py                  SileroVad(VadBase)
  stt/onnx_asr_engine.py         OnnxAsrEngine(SttBase)
  tts/kokoro.py                  KokoroTts(TtsBase)
  tts/supertonic.py              SupertonicTts(TtsBase)
  llm/openai_compatible.py       OpenAiCompatibleLlm(LlmBase)
tests/                           pytest suite (fakes.py holds shared no-hardware/no-network test doubles); run with `uv run pytest`
```

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

- `"headphones"` echo mode assumes real mic/speaker isolation; using it
  without that causes false VAD triggers from the bot's own voice.
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

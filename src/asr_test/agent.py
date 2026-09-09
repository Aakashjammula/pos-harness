from __future__ import annotations

import queue
import re
import statistics
import threading
import time
from collections.abc import Callable

import numpy as np
import sounddevice as sd

from . import config
from .audio.echo import EchoControl
from .audio.output import LocalAudioSink
from .interfaces import AudioSinkBase, LlmBase, SttBase, TtsBase, VadBase
from .llm import OpenAiCompatibleLlm
from .stt import OnnxAsrEngine
from .tts import KokoroTts
from .vad import SileroVad


class Agent:
    def __init__(
        self,
        vad: VadBase | None = None,
        stt: SttBase | None = None,
        tts: TtsBase | None = None,
        llm: LlmBase | None = None,
        trigger_word: str | None = None,
        audio_sink: AudioSinkBase | None = None,
        on_event: Callable[[str, dict], None] | None = None,
    ):
        print("Loading models...")
        t0 = time.perf_counter()

        self.echo = EchoControl(config.ECHO_MODE, config.MIC_RATE, config.FRAME)

        self.vad = vad or SileroVad(
            sample_rate=config.MIC_RATE,
            min_silence_ms=config.MIN_SILENCE_MS,
            speech_pad_ms=config.SPEECH_PAD_MS,
        )
        self.stt = stt or OnnxAsrEngine()
        self.tts = tts or KokoroTts()
        self.llm = llm or OpenAiCompatibleLlm()

        # Matches the trigger phrase (case-insensitive, word boundary so
        # "computer" doesn't match "computers"), plus any trailing
        # comma/whitespace, so `.end()` lands right at the start of the
        # actual command. Searched (not anchored) within a leading window
        # of the transcript, not just position 0 — STT/VAD commonly pick up
        # a beat of "uh"/"hey"/"okay" before the real trigger word, so
        # requiring an exact prefix match drops genuine commands. Bounding
        # the search to a small leading window (rather than searching the
        # whole transcript) keeps "must be said near the start" intact —
        # e.g. "tell me about the computer" still won't trigger.
        self._trigger_re = (
            re.compile(rf"\b{re.escape(trigger_word)}\b[,]?\s*", re.IGNORECASE)
            if trigger_word
            else None
        )
        self._trigger_window_words = (
            len(trigger_word.split()) + config.TRIGGER_LOOKAHEAD_WORDS if trigger_word else 0
        )

        self.audio_out = audio_sink or LocalAudioSink(
            self.tts.sample_rate,
            blocksize=config.OUT_BLOCK,
            on_played=lambda a: self.echo.note_playback(a, self.tts.sample_rate),
        )

        print(f"Models loaded in {time.perf_counter() - t0:.2f}s")
        print(f"  barge-in: {'ON' if self.echo.barge_in else 'OFF'}")
        if trigger_word is not None:
            print(f"  trigger word: {trigger_word!r} — ignoring speech that doesn't lead with it")
        print()

        self.mic_q: queue.Queue[np.ndarray] = queue.Queue()
        self.seg_q: queue.Queue[tuple[int, np.ndarray]] = queue.Queue()
        # (turn, chunk_no, text, stt_t, turn_start) — sentence chunks awaiting synthesis.
        # Kept on its own queue/thread so TTS synthesis overlaps LLM generation instead
        # of blocking it (see respond() / tts_thread()).
        self.tts_q: queue.Queue[tuple[int, int, str, float, float]] = queue.Queue()

        self.turn_id = 0
        self.turn_lock = threading.Lock()
        self.cancel = threading.Event()
        self.stop = threading.Event()

        self.speech_buf: list[np.ndarray] = []
        self.in_speech = False
        self.speech_run = 0
        self._barge_in_speech = False

        self.m_asr: list[float] = []
        self.m_ttft: list[float] = []
        self.m_llm: list[float] = []
        self.m_tts: list[float] = []
        self.m_rtf: list[float] = []
        self.m_ttfa: list[float] = []
        self.interrupts = 0
        self.turn_start: float | None = None

        self.conversation: list[dict] = []
        # UI hook for browser/CLI clients — e.g. "user_text"/"bot_text" for
        # live captions, "interrupted" for barge-in feedback. No-op by
        # default so local mode (main.py) behaves exactly as before.
        self._on_event = on_event or (lambda name, data: None)

        # Set to mute: feed_audio() drops frames instead of queuing them,
        # so VAD never sees anything and the agent stays idle. One flag
        # works for every transport (local mic, websocket) since they all
        # funnel through feed_audio().
        self.muted = threading.Event()

    def new_turn(self) -> int:
        with self.turn_lock:
            self.turn_id += 1
            return self.turn_id

    def current_turn(self) -> int:
        with self.turn_lock:
            return self.turn_id

    def interrupt(self):
        self.interrupts += 1
        print("      interrupted")
        self._on_event("interrupted", {})
        self.cancel.set()
        self.audio_out.flush()
        self.new_turn()

    def feed_audio(self, frame: np.ndarray) -> None:
        """Transport-agnostic entry point for one mono audio frame.
        Any transport (local InputStream, websocket receive loop) calls
        this the same way. Dropped while muted."""
        if self.muted.is_set():
            return
        self.mic_q.put(frame)

    def start(self) -> list[threading.Thread]:
        """Spawn the vad/worker/tts threads without opening any input
        device. run() (local mode) wraps this; server mode calls it
        directly per connection."""
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

    def vad_thread(self):
        while not self.stop.is_set():
            try:
                raw = self.mic_q.get(timeout=0.2)
            except queue.Empty:
                continue

            playing = self.audio_out.playing
            frame = self.echo.process(raw.flatten().astype(np.float32), playing)
            if frame is None:
                continue

            event = self.vad(frame)

            if playing and self.echo.barge_in:
                if self.audio_out.elapsed_ms < config.BARGE_IN_GRACE_MS:
                    continue
                # VADIterator fires "start" once, on the onset frame only —
                # it does NOT repeat while speech continues. So count frames
                # inside an unclosed speech episode (start..end), not literal
                # repeats of the "start" event, or this can never reach
                # BARGE_IN_FRAMES (confirmed: with the old "count literal
                # start events" logic, speech_run never exceeded 1).
                if event and "start" in event:
                    self._barge_in_speech = True
                    self.speech_run = 1
                elif event and "end" in event:
                    self._barge_in_speech = False
                    self.speech_run = 0
                elif self._barge_in_speech:
                    self.speech_run += 1
                else:
                    self.speech_run = 0
                if self.speech_run >= config.BARGE_IN_FRAMES:
                    self.speech_run = 0
                    self._barge_in_speech = False
                    if config.REALTIME_LOG:
                        print("      [VAD] barge-in detected")
                    self.interrupt()
                    self.in_speech = True
                    self.speech_buf = [frame]
                continue

            self.speech_run = 0
            self._barge_in_speech = False

            if event:
                if "start" in event:
                    self.in_speech = True
                    self.speech_buf = [frame]
                    if config.REALTIME_LOG:
                        print("      [VAD] speech start")
                elif "end" in event:
                    self.in_speech = False
                    if self.speech_buf:
                        seg = np.concatenate(self.speech_buf)
                        if config.REALTIME_LOG:
                            print(f"      [VAD] speech end    dur={len(seg) / config.MIC_RATE:.2f}s")
                        self.seg_q.put((self.current_turn(), seg))
                        self.speech_buf = []
            elif self.in_speech:
                self.speech_buf.append(frame)

    def worker_thread(self):
        while not self.stop.is_set():
            try:
                turn, audio = self.seg_q.get(timeout=0.2)
            except queue.Empty:
                continue

            dur = len(audio) / config.MIC_RATE
            if dur < config.MIN_SPEECH_SEC:
                continue
            if dur > config.MAX_SEGMENT_SEC:
                audio = audio[: int(config.MAX_SEGMENT_SEC * config.MIC_RATE)]

            self.turn_start = time.perf_counter()
            self.cancel.clear()

            if config.REALTIME_LOG:
                print(f"      [STT] triggered      audio={dur:.2f}s")

            t0 = time.perf_counter()
            try:
                text = self.stt(audio)
            except Exception as e:
                print(f"   [stt failed: {e}]")
                continue
            stt_t = time.perf_counter() - t0
            self.m_asr.append(stt_t)

            if len(text) < 2 or turn != self.current_turn():
                if config.REALTIME_LOG:
                    reason = "stale turn" if turn != self.current_turn() else f"empty/short STT output {text!r}"
                    print(f"      [STT] dropped        {reason}  (audio={dur:.2f}s, stt={stt_t:.2f}s)")
                continue

            if self._trigger_re is not None:
                words = text.split()
                window = " ".join(words[: self._trigger_window_words])
                m = self._trigger_re.search(window)
                remainder = " ".join(words[len(window[: m.end()].split()):]) if m else ""
                if not m or len(remainder) < 2:
                    if config.REALTIME_LOG:
                        reason = "no trigger word" if not m else "nothing after trigger word"
                        print(f"      [STT] dropped        {reason}  (text={text!r})")
                    continue
                # Gate on the trigger, but send the LLM the full, unmodified
                # STT output (not the stripped remainder) — text is left as-is.

            print(f"USER: {text}")
            self._on_event("user_text", {"text": text})
            self.respond(text, turn, stt_t)

    def respond(self, text: str, turn: int, stt_t: float):
        """LLM producer stage: streams tokens and hands finished sentence
        chunks to tts_q as soon as each is ready. Never calls the TTS engine
        itself — that runs on tts_thread, concurrently with generation of
        the *next* sentence, instead of stalling the token stream while it
        synthesizes the current one.
        """
        buf = ""
        first = True
        spoken: list[str] = []
        full_response: list[str] = []
        chunk_no = 0
        turn_start = self.turn_start or time.perf_counter()

        def limit() -> int:
            return config.FIRST_CHUNK_CHARS if first else config.MAX_CHUNK_CHARS

        def enqueue(chunk: str) -> bool:
            nonlocal first, chunk_no
            chunk = chunk.strip()
            if not chunk:
                return True
            if self.cancel.is_set() or turn != self.current_turn():
                return False

            chunk_no += 1
            first = False
            self.tts_q.put((turn, chunk_no, chunk, stt_t, turn_start))
            spoken.append(chunk)
            return True

        def drain(buf: str) -> str | None:
            """Pull every ready chunk out of buf, bounded by limit(): prefer
            a full sentence if one ends within the limit, else force a cut
            at the last word boundary within the limit. Unlike a plain
            "split on sentence end, else split on char count" fallback,
            this bounds *every* chunk regardless of how the LLM streams —
            some servers batch many tokens (or a whole sentence) into one
            delta, which otherwise produces a chunk far past MAX_CHUNK_CHARS
            (observed: a 172-char / 9s-audio chunk from one big delta).
            Returns the unconsumed remainder, or None if enqueue() signaled
            to stop (turn went stale / cancelled).
            """
            while True:
                lim = limit()
                m = config.SENTENCE_END.search(buf)
                if m and m.start() <= lim:
                    piece, buf = buf[: m.start()], buf[m.end():]
                elif len(buf) >= lim:
                    cut = buf.rfind(" ", 0, lim)
                    if cut <= 0:
                        break  # no split point yet; wait for more text
                    piece, buf = buf[:cut], buf[cut + 1:]
                else:
                    break
                if not enqueue(piece):
                    return None
            return buf

        if config.REALTIME_LOG:
            print(f"      [LLM] triggered      \"{text}\"")

        messages = self.conversation[-config.HISTORY_TURNS * 2 :] + [
            {"role": "user", "content": text}
        ]

        try:
            for piece in self.llm.stream(messages, self.cancel):
                if self.cancel.is_set() or turn != self.current_turn():
                    return
                full_response.append(piece)
                buf += piece
                buf = drain(buf)
                if buf is None:
                    return
        except Exception as e:
            print(f"   [llm failed: {e}]")
            return

        if buf.strip():
            enqueue(buf)

        if self.llm.last_ttft is not None:
            self.m_ttft.append(self.llm.last_ttft)
        if self.llm.last_total is not None:
            self.m_llm.append(self.llm.last_total)
            if config.VERBOSE_TIMING:
                print(f"      llm: ttft {self.llm.last_ttft or 0:.2f}s / "
                      f"total {self.llm.last_total:.2f}s")

        if full_response:
            joined = "".join(full_response)
            self.conversation.append({"role": "user", "content": text})
            self.conversation.append({"role": "assistant", "content": joined})
            self._on_event("bot_text", {"text": joined})

        if spoken:
            print(f"BOT:  {' '.join(spoken)}")

    def tts_thread(self):
        """TTS consumer stage: synthesizes queued sentence chunks in order
        and pushes them to playback, decoupled from respond()'s LLM loop.
        """
        while not self.stop.is_set():
            try:
                turn, chunk_no, chunk, stt_t, turn_start = self.tts_q.get(timeout=0.2)
            except queue.Empty:
                continue

            if self.cancel.is_set() or turn != self.current_turn():
                continue

            if config.REALTIME_LOG:
                preview = chunk if len(chunk) <= 40 else chunk[:37] + "..."
                print(f"      [TTS] triggered #{chunk_no}   \"{preview}\"")

            t0 = time.perf_counter()
            try:
                audio = self.tts(chunk)
            except Exception as e:
                print(f"   [tts failed: {e}]")
                continue
            synth = time.perf_counter() - t0

            if self.cancel.is_set() or turn != self.current_turn():
                continue
            if audio.size == 0:
                continue

            audio_sec = audio.size / self.tts.sample_rate
            rtf = synth / audio_sec if audio_sec else 0.0
            self.m_tts.append(synth)
            self.m_rtf.append(rtf)

            self.audio_out.push(audio)

            if config.VERBOSE_TIMING:
                flag = "" if rtf < 1.0 else "   SLOWER THAN REALTIME"
                print(f"      tts#{chunk_no}: synth {synth:.2f}s / "
                      f"audio {audio_sec:.2f}s  rtf={rtf:.2f}{flag}")

            if chunk_no == 1:
                ttfa = time.perf_counter() - turn_start
                self.m_ttfa.append(ttfa)
                ttft = self.llm.last_ttft or 0.0
                other = ttfa - stt_t - ttft - synth
                print(f"      TTFA {ttfa:.2f}s = stt {stt_t:.2f} + "
                      f"llm_ttft {ttft:.2f} + tts {synth:.2f} + other {other:.2f}")

    def run(self, device: int | None = None):
        threads = self.start()

        print("Listening — speak any time, including over the bot. Ctrl+C to stop.\n")

        def _on_frame(indata, frames, time_info, status):
            if status:
                print("Audio status:", status)
            self.feed_audio(indata.flatten().astype(np.float32))

        mic = sd.InputStream(
            device=device,
            samplerate=config.MIC_RATE,
            channels=1,
            dtype="float32",
            blocksize=config.FRAME,
            callback=_on_frame,
        )

        with mic:
            try:
                while True:
                    time.sleep(0.2)
            except KeyboardInterrupt:
                print("\nStopping...")
                self.shutdown(threads)
                self.report()

    def report(self):
        def line(label, vals, unit="s"):
            if not vals:
                print(f"  {label:14s} -")
                return
            print(f"  {label:14s} avg={statistics.mean(vals):.2f}{unit}  "
                  f"min={min(vals):.2f}{unit}  max={max(vals):.2f}{unit}  "
                  f"n={len(vals)}")

        print("\n--- session summary ---")
        print(f"  echo mode      {self.echo.mode} "
              f"(barge-in {'on' if self.echo.barge_in else 'off'})")
        line("stt", self.m_asr)
        line("llm_ttft", self.m_ttft)
        line("llm_total", self.m_llm)
        line("tts_synth", self.m_tts)
        line("tts_rtf", self.m_rtf, unit="x")
        line("ttfa", self.m_ttfa)
        print(f"  interruptions  {self.interrupts}")
        print(f"  underruns      {self.audio_out.underruns}  (TTS fell behind playback mid-turn)")
        if self.m_rtf and statistics.mean(self.m_rtf) >= 1.0:
            print("\n  TTS is slower than realtime — consider a faster engine")
        print("Stopped cleanly.")

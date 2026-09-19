"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { WS_URL } from "@/lib/config";
import { apiFetch } from "@/lib/apiFetch";
import { streamChat } from "@/lib/chatStream";
import { floatToPcm16, pcm16ToFloat, rms } from "@/lib/audio";
import type {
  ConnState,
  Latency,
  SessionMode,
  StoredTurn,
  TranscriptLine,
  Usage,
  VoiceInputMode,
} from "@/lib/types";

const LOUD_RMS = 0.01;
const SPEAKING_HANGOVER_MS = 400;

export interface ConnectConfig {
  mode: SessionMode;
  ttsEngine: string;
  ttsVoice: string;
  llmModel: string;
  micDeviceId: string;
  voiceInputMode: VoiceInputMode;
  triggerWord: string;
  vadThreshold: string;
  vadMinSilenceMs: string;
  vadSpeechPadMs: string;
  provider?: string;
  resumeSessionId: string | null;
}

const STATE_TEXT: Record<ConnState, string> = {
  idle: "Disconnected",
  connecting: "Connecting…",
  listening: "Listening",
  speaking: "Speaking",
  muted: "Muted",
  error: "Error",
};

let lineIdSeq = 0;
const nextLineId = () => `line-${++lineIdSeq}`;

// Mirrors the original index.html's single big imperative closure, just
// reorganized into a hook. Mutable session/audio state lives in refs (not
// React state) since it's read from WebSocket/AudioContext callbacks that
// must always see the latest value, not one captured at render time.
export function useVoiceSession() {
  const [state, setStateValue] = useState<ConnState>("idle");
  const [statusLabel, setStatusLabel] = useState("Not connected");
  const [connected, setConnected] = useState(false);
  const [lines, setLines] = useState<TranscriptLine[]>([]);
  const [lineCountLabel, setLineCountLabel] = useState("");
  const [micMuted, setMicMutedValue] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [mode, setModeValue] = useState<SessionMode>("text");
  const [activeVoiceInputMode, setActiveVoiceInputMode] = useState<VoiceInputMode>("vad");
  const [replying, setReplying] = useState(false); // a text reply is streaming in
  const [historyVersion, setHistoryVersion] = useState(0); // bumps when the server names/creates a chat

  const wsRef = useRef<WebSocket | null>(null);
  // Text chat has no socket: each message is one streamed POST. These carry
  // the chat's identity and the in-flight request between messages.
  const textSessionIdRef = useRef<string | null>(null);
  const streamAbortRef = useRef<AbortController | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const playCtxRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const outputSampleRateRef = useRef(16000);
  const nextPlayTimeRef = useRef(0);

  const turnCountRef = useRef(0);
  const totalTokensRef = useRef(0);
  const totalCostRef = useRef(0);
  const totalCostKnownRef = useRef(false);

  const lastLoudAtRef = useRef(0);
  const speakingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const pttHeldRef = useRef(false);
  const micMutedRef = useRef(false);
  const activeVoiceInputModeRef = useRef<VoiceInputMode>("vad");
  const connectedRef = useRef(false);
  const currentStateRef = useRef<ConnState>("idle");
  // Mirrors `mode` state, but read synchronously inside ws.onmessage
  // handlers -- those can fire before a setState from connect() has
  // flushed to a re-render, so the closured `mode` state value isn't
  // safe to rely on there.
  const modeRef = useRef<SessionMode>("text");

  const setState = useCallback((next: ConnState, label: string) => {
    currentStateRef.current = next;
    setStateValue(next);
    setStatusLabel(label);
  }, []);

  const refreshIdleState = useCallback(() => {
    if (!connectedRef.current) return;
    if (micMutedRef.current) setState("muted", "Microphone is muted");
    else setState("listening", "Listening — just start talking");
  }, [setState]);

  const updateSessionTotals = useCallback((usage: Usage) => {
    if (usage.total_tokens != null) totalTokensRef.current += usage.total_tokens;
    if (usage.cost_usd != null) {
      totalCostRef.current += usage.cost_usd;
      totalCostKnownRef.current = true;
    }
    const tokenPart = totalTokensRef.current ? `${totalTokensRef.current.toLocaleString()} tokens` : "";
    const costPart = totalCostKnownRef.current ? `$${totalCostRef.current.toFixed(4)}` : "";
    const combined = [tokenPart, costPart].filter(Boolean).join(" · ");
    setLineCountLabel(
      combined ||
        (turnCountRef.current
          ? `${turnCountRef.current}${turnCountRef.current === 1 ? " message" : " messages"}`
          : "")
    );
  }, []);

  const addLine = useCallback(
    (who: TranscriptLine["who"], text: string, usage?: Usage, latency?: Latency) => {
      setLines((prev) => [...prev, { id: nextLineId(), who, text, usage, latency }]);
      if (usage) updateSessionTotals(usage);
      if (who === "you" || who === "bot") {
        turnCountRef.current++;
        if (!usage) {
          setLineCountLabel(
            `${turnCountRef.current}${turnCountRef.current === 1 ? " message" : " messages"}`
          );
        }
      }
    },
    [updateSessionTotals]
  );

  const resetTranscript = useCallback(() => {
    setLines([]);
    turnCountRef.current = 0;
    totalTokensRef.current = 0;
    totalCostRef.current = 0;
    totalCostKnownRef.current = false;
    setLineCountLabel("");
  }, []);

  // Loads a resumed session's stored turns into the transcript BEFORE
  // connect() is called with resumeSessionId set, mirroring
  // continueSession() in the original client.
  const loadHistory = useCallback((turns: StoredTurn[]) => {
    resetTranscript();
    setLines(
      turns.map((turn) => ({
        id: nextLineId(),
        who: turn.role === "user" ? "you" : "bot",
        text: turn.text,
        usage: turn.usage,
      }))
    );
  }, [resetTranscript]);

  const stopSpeakingWatchdog = useCallback(() => {
    if (speakingTimerRef.current) clearInterval(speakingTimerRef.current);
    speakingTimerRef.current = null;
  }, []);

  const startSpeakingWatchdog = useCallback(() => {
    speakingTimerRef.current = setInterval(() => {
      if (!connectedRef.current) return;
      if (Date.now() - lastLoudAtRef.current > SPEAKING_HANGOVER_MS) refreshIdleState();
    }, 120);
  }, [refreshIdleState]);

  const noteIncomingAudio = useCallback(
    (float32: Float32Array) => {
      if (rms(float32) > LOUD_RMS) {
        lastLoudAtRef.current = Date.now();
        if (connectedRef.current) setState("speaking", "The assistant is talking");
      }
    },
    [setState]
  );

  const disconnect = useCallback(
    (reason?: string) => {
      connectedRef.current = false;
      setConnected(false);
      streamAbortRef.current?.abort();
      streamAbortRef.current = null;
      setReplying(false);
      pttHeldRef.current = false;
      micMutedRef.current = false;
      setMicMutedValue(false);
      stopSpeakingWatchdog();
      if (processorRef.current) {
        processorRef.current.disconnect();
        processorRef.current = null;
      }
      if (sourceRef.current) {
        sourceRef.current.disconnect();
        sourceRef.current = null;
      }
      if (micStreamRef.current) {
        micStreamRef.current.getTracks().forEach((t) => t.stop());
        micStreamRef.current = null;
      }
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
        wsRef.current = null;
      }
      setState("idle", reason || "Not connected");
    },
    [setState, stopSpeakingWatchdog]
  );

  const handleEvent = useCallback(
    (msg: Record<string, unknown>) => {
      if (msg.event === "ready") {
        outputSampleRateRef.current = msg.output_sample_rate as number;
        const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        playCtxRef.current = new Ctx({ sampleRate: outputSampleRateRef.current });
        nextPlayTimeRef.current = 0;
        connectedRef.current = true;
        setConnected(true);
        setSessionId(msg.session_id as string);
        setState(
          "listening",
          activeVoiceInputModeRef.current === "push_to_talk"
            ? "Hold Space to talk"
            : "Listening — just start talking"
        );
        startSpeakingWatchdog();
      } else if (msg.event === "user_text") {
        addLine("you", msg.text as string);
      } else if (msg.event === "bot_text") {
        addLine("bot", msg.text as string, msg.usage as Usage | undefined, msg.latency as Latency | undefined);
      } else if (msg.event === "interrupted") {
        addLine("system", "Interrupted");
      } else if (msg.event === "error") {
        disconnect((msg.message as string) || "Server rejected the connection");
      }
    },
    [addLine, disconnect, setState, startSpeakingWatchdog]
  );

  const connect = useCallback(
    async (config: ConnectConfig, resuming: boolean) => {
      // Voice only. Text chat never connects: each message is one streamed request (sendText).
      modeRef.current = "voice";
      setModeValue("voice");
      streamAbortRef.current?.abort();

      // A WebSocket can't be retried after the fact, so make sure the access token
      // is fresh first: any authenticated call refreshes it if it has expired.
      await apiFetch("/auth/me").catch(() => undefined);

      setState("connecting", "Opening the microphone…");
      const audioConstraints: MediaTrackConstraints | boolean = config.micDeviceId
        ? { deviceId: { exact: config.micDeviceId } }
        : true;
      try {
        micStreamRef.current = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });
      } catch {
        setState("error", "Microphone access was denied");
        return;
      }

      activeVoiceInputModeRef.current = config.voiceInputMode;
      setActiveVoiceInputMode(activeVoiceInputModeRef.current);
      pttHeldRef.current = false;

      const params = new URLSearchParams();
      params.set("mode", "voice");
      params.set("tts", config.ttsEngine);
      if (config.ttsVoice) params.set("voice", config.ttsVoice);
      if (config.llmModel) params.set("llm_model", config.llmModel);   // unset: the provider picks its own default
      if (config.provider) params.set("provider", config.provider);
      params.set("voice_input_mode", activeVoiceInputModeRef.current);
      if (config.triggerWord.trim()) params.set("trigger_word", config.triggerWord.trim());
      if (config.vadThreshold.trim()) params.set("vad_threshold", config.vadThreshold.trim());
      if (config.vadMinSilenceMs.trim()) params.set("vad_min_silence_ms", config.vadMinSilenceMs.trim());
      if (config.vadSpeechPadMs.trim()) params.set("vad_speech_pad_ms", config.vadSpeechPadMs.trim());
      if (config.resumeSessionId) params.set("resume_session_id", config.resumeSessionId);

      const ws = new WebSocket(`${WS_URL}/ws?${params.toString()}`);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () => {
        // A resumed session's prior turns were already rendered by
        // loadHistory() before connect() was even called -- clearing
        // here would wipe them out right as the connection opens.
        if (!resuming) resetTranscript();
      };
      ws.onclose = () => {
        if (connectedRef.current) disconnect("Connection closed");
      };
      ws.onerror = () => setState("error", "Connection error");
      ws.onmessage = (event: MessageEvent) => {
        if (typeof event.data === "string") {
          handleEvent(JSON.parse(event.data));
          return;
        }
        const floatData = pcm16ToFloat(event.data as ArrayBuffer);
        noteIncomingAudio(floatData);
        const playCtx = playCtxRef.current;
        if (playCtx && floatData.length) {
          const buffer = playCtx.createBuffer(1, floatData.length, outputSampleRateRef.current);
          buffer.copyToChannel(floatData as Float32Array<ArrayBuffer>, 0);
          const src = playCtx.createBufferSource();
          src.buffer = buffer;
          src.connect(playCtx.destination);
          // Scheduled on a running timeline instead of "play now" on
          // every ~40ms block, since network delivery isn't perfectly
          // metronomic -- see the original index.html's comment on this.
          const startAt = Math.max(playCtx.currentTime, nextPlayTimeRef.current);
          src.start(startAt);
          nextPlayTimeRef.current = startAt + buffer.duration;
        }
      };

      const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const audioCtx = new Ctx({ sampleRate: 16000 });
      audioCtxRef.current = audioCtx;
      const source = audioCtx.createMediaStreamSource(micStreamRef.current!);
      sourceRef.current = source;
      const processor = audioCtx.createScriptProcessor(512, 1, 1);
      processorRef.current = processor;
      processor.onaudioprocess = (e: AudioProcessingEvent) => {
        if (micMutedRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
        if (activeVoiceInputModeRef.current === "push_to_talk" && !pttHeldRef.current) return;
        wsRef.current.send(floatToPcm16(e.inputBuffer.getChannelData(0)));
      };
      source.connect(processor);
      processor.connect(audioCtx.destination);
    },
    [disconnect, handleEvent, noteIncomingAudio, resetTranscript, setState]
  );

  const setMicMuted = useCallback(
    (value: boolean) => {
      micMutedRef.current = value;
      setMicMutedValue(value);
      if (micStreamRef.current) micStreamRef.current.getAudioTracks().forEach((t) => (t.enabled = !value));
      if (currentStateRef.current !== "speaking") refreshIdleState();
    },
    [refreshIdleState]
  );

  const toggleMute = useCallback(() => setMicMuted(!micMutedRef.current), [setMicMuted]);

  // Replace one transcript line in place (the streaming bot reply).
  const patchLine = useCallback((id: string, patch: (line: TranscriptLine) => TranscriptLine) => {
    setLines((prev) => prev.map((l) => (l.id === id ? patch(l) : l)));
  }, []);

  /** Start (or continue) a text chat. There is nothing to connect to: this only sets which
   * chat new messages belong to. `clear` empties the transcript for a fresh chat. */
  const startTextChat = useCallback(
    (resumeSessionId: string | null, clear: boolean) => {
      streamAbortRef.current?.abort();
      streamAbortRef.current = null;
      setReplying(false);
      modeRef.current = "text";
      setModeValue("text");
      textSessionIdRef.current = resumeSessionId;
      if (clear) resetTranscript();
      setSessionId(resumeSessionId);
      setState("idle", "Type a message below");
    },
    [resetTranscript, setState]
  );

  const sendText = useCallback(
    (text: string, config: { provider?: string; llmModel: string }) => {
      const value = text.trim();
      if (!value || modeRef.current !== "text" || streamAbortRef.current) return;

      addLine("you", value);
      const botId = nextLineId();
      setLines((prev) => [...prev, { id: botId, who: "bot", text: "" }]);
      turnCountRef.current++;

      const controller = new AbortController();
      streamAbortRef.current = controller;
      setReplying(true);

      streamChat(
        {
          message: value,
          session_id: textSessionIdRef.current,
          provider: config.provider,
          llm_model: config.llmModel || undefined,
        },
        {
          onSession: (id) => {
            if (textSessionIdRef.current !== id) setHistoryVersion((v) => v + 1); // a new chat exists
            textSessionIdRef.current = id;
            setSessionId(id);
          },
          onToken: (piece) => patchLine(botId, (l) => ({ ...l, text: l.text + piece })),
          onDone: (d) => {
            const usage = d.usage as Usage | undefined;
            const latency = d.latency as Latency | undefined;
            patchLine(botId, (l) => ({ ...l, text: d.text, usage, latency }));
            if (usage) updateSessionTotals(usage);
          },
          onTitle: () => setHistoryVersion((v) => v + 1),
          onError: (message) => {
            // Drop the empty bubble if nothing had arrived; keep partial text if it had.
            setLines((prev) => prev.filter((l) => l.id !== botId || l.text !== ""));
            addLine("error", message);
          },
        },
        controller.signal
      )
        .catch((e: unknown) => {
          if ((e as Error).name === "AbortError") return; // user disconnected mid-reply
          setLines((prev) => prev.filter((l) => l.id !== botId || l.text !== ""));
          addLine("error", `Connection problem: ${(e as Error).message}`);
        })
        .finally(() => {
          if (streamAbortRef.current === controller) streamAbortRef.current = null;
          setReplying(false);
        });
    },
    [addLine, patchLine, updateSessionTotals]
  );

  const pttStart = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    pttHeldRef.current = true;
    wsRef.current.send(JSON.stringify({ event: "ptt_start" }));
    setState("listening", "Recording — release Space to send");
  }, [setState]);

  const pttStop = useCallback(() => {
    if (!pttHeldRef.current) return;
    pttHeldRef.current = false;
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ event: "ptt_stop" }));
    }
    if (currentStateRef.current !== "speaking") setState("listening", "Hold Space to talk");
  }, [setState]);

  const reportError = useCallback((message: string) => setState("error", message), [setState]);

  useEffect(() => () => disconnect(), [disconnect]);

  return {
    state,
    statusLabel,
    stateText: STATE_TEXT[state],
    connected,
    lines,
    lineCountLabel,
    micMuted,
    sessionId,
    mode,
    activeVoiceInputMode,
    replying,
    historyVersion,
    isPttActive: activeVoiceInputMode === "push_to_talk",
    connect,
    disconnect,
    toggleMute,
    startTextChat,
    sendText,
    pttStart,
    pttStop,
    isPttHeld: () => pttHeldRef.current,
    loadHistory,
    resetTranscript,
    reportError,
  };
}

export type VoiceSession = ReturnType<typeof useVoiceSession>;

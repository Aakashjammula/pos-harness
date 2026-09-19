"use client";

import { hasLlm } from "@/lib/llm";
import { useCallback, useEffect, useRef, useState } from "react";
import { deleteSession, fetchOptions, fetchSession, fetchSessions } from "@/lib/api";
import { AuthGuard } from "@/components/AuthGuard";
import type { CurrentUser } from "@/lib/auth";
import { fetchConfiguredProviders } from "@/lib/credentials";
import { DEFAULT_SETTINGS, type ApiKeyFields, type OptionsResponse, type SessionMode, type SessionSummary, type Settings } from "@/lib/types";
import { useVoiceSession } from "@/hooks/useVoiceSession";
import { Sidebar } from "@/components/Sidebar";
import { ChatPanel } from "@/components/ChatPanel";
import { SettingsPanel } from "@/components/SettingsPanel";

function isTypingTarget(target: EventTarget | null): boolean {
  const tag = (target as HTMLElement | null)?.tagName;
  return tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA";
}

export default function Home() {
  return <AuthGuard>{(user) => <VoiceAgent user={user} />}</AuthGuard>;
}

function VoiceAgent({ user }: { user: CurrentUser }) {
  const session = useVoiceSession();
  const [mode, setMode] = useState<SessionMode>("text");
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [options, setOptions] = useState<OptionsResponse | null>(null);
  const [optionsError, setOptionsError] = useState(false);
  const [mics, setMics] = useState<MediaDeviceInfo[]>([]);
  const [sessions, setSessions] = useState<SessionSummary[] | null>(null);
  const [sessionsError, setSessionsError] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [configured, setConfigured] = useState<string[]>([]);

  // --- initial data: /options, mic list, session history ---

  useEffect(() => {
    fetchOptions()
      .then((opts) => {
        setOptions(opts);
        setSettings((s) => ({
          ...s,
          ttsEngine: opts.defaults.tts_engine,
          llmModel: opts.defaults.llm_model,
        }));
      })
      .catch(() => setOptionsError(true));
  }, []);

  const refreshMics = useCallback(async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      setMics(devices.filter((d) => d.kind === "audioinput"));
    } catch {
      // enumerateDevices can fail (insecure context, unsupported browser) --
      // leave the mic list empty, "System default" still works fine.
    }
  }, []);

  useEffect(() => {
    // Fetch-on-mount: refreshMics is async, so any setState it performs
    // happens after a microtask, not synchronously within this effect.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    refreshMics();
    if (navigator.mediaDevices) navigator.mediaDevices.ondevicechange = refreshMics;
    return () => {
      if (navigator.mediaDevices) navigator.mediaDevices.ondevicechange = null;
    };
  }, [refreshMics]);

  const refreshHistoryList = useCallback(async () => {
    try {
      const list = await fetchSessions();
      setSessions(list);
      setSessionsError(false);
    } catch {
      setSessionsError(true);
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    refreshHistoryList();
  }, [refreshHistoryList]);

  const refreshConfigured = useCallback(async () => {
    try {
      setConfigured(await fetchConfiguredProviders());
    } catch {
      // leave the previous list in place -- a transient failure here
      // shouldn't blank out dots the user just saw as configured
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    refreshConfigured();
  }, [refreshConfigured]);

  // hash-based settings route, so #/settings survives refresh/back-forward
  useEffect(() => {
    const applyRoute = () => setSettingsOpen(window.location.hash === "#/settings");
    applyRoute();
    window.addEventListener("hashchange", applyRoute);
    return () => window.removeEventListener("hashchange", applyRoute);
  }, []);

  const openSettings = useCallback(() => {
    window.location.hash = "#/settings";
  }, []);
  const closeSettings = useCallback(() => {
    window.location.hash = "";
  }, []);

  // Once connected, leave Settings automatically (mirrors the original's
  // ws "ready" handler forcing location.hash back to "").
  useEffect(() => {
    if (session.connected && settingsOpen) closeSettings();
  }, [session.connected, settingsOpen, closeSettings]);

  // Refresh the sidebar once a session becomes ready, and again ~1.5s
  // after each bot reply (the server generates a title off-thread, after
  // the first exchange, so the row's label may only be ready shortly
  // after this).
  const prevConnected = useRef(false);
  useEffect(() => {
    if (session.connected && !prevConnected.current) refreshHistoryList();
    prevConnected.current = session.connected;
  }, [session.connected, refreshHistoryList]);

  const prevLineCount = useRef(0);
  useEffect(() => {
    if (session.lines.length > prevLineCount.current) {
      const last = session.lines[session.lines.length - 1];
      if (last?.who === "bot") {
        const t = setTimeout(() => refreshHistoryList().catch(() => {}), 1500);
        return () => clearTimeout(t);
      }
    }
    prevLineCount.current = session.lines.length;
  }, [session.lines, refreshHistoryList]);

  // --- settings field updates ---

  const onSettingsChange = useCallback((patch: Partial<Settings>) => {
    setSettings((s) => ({ ...s, ...patch }));
  }, []);
  const onKeysChange = useCallback((patch: Partial<ApiKeyFields>) => {
    setSettings((s) => ({ ...s, keys: { ...s.keys, ...patch } }));
  }, []);

  // --- connect / disconnect ---

  const doConnect = useCallback(
    async (resumeSessionId: string | null, connectMode: SessionMode) => {
      await session.connect(
        {
          mode: connectMode,
          ttsEngine: settings.ttsEngine,
          ttsVoice: settings.ttsVoice,
          llmModel: settings.llmModel,
          micDeviceId: settings.micDeviceId,
          voiceInputMode: settings.voiceInputMode,
          triggerWord: settings.triggerWord,
          vadThreshold: settings.vadThreshold,
          vadMinSilenceMs: settings.vadMinSilenceMs,
          vadSpeechPadMs: settings.vadSpeechPadMs,
          provider: settings.provider || undefined,
          resumeSessionId,
        },
        resumeSessionId !== null
      );
      refreshMics(); // refresh with real labels now that mic permission was (maybe) granted
    },
    [settings, session, refreshMics]
  );

  const handleConnect = useCallback(() => {
    doConnect(null, mode);
  }, [doConnect, mode]);

  const handleDisconnect = useCallback(() => {
    session.disconnect();
  }, [session]);

  // Switching Voice/Text mid-session resumes the SAME session under the
  // new mode -- a quick disconnect+reconnect, not an in-place switch.
  const handleModeChange = useCallback(
    (next: SessionMode) => {
      if (next === mode) return;
      if (session.connected) {
        const sessionId = session.sessionId;
        session.disconnect();
        setMode(next);
        doConnect(sessionId, next);
      } else {
        setMode(next);
      }
    },
    [mode, session, doConnect]
  );

  const continueSession = useCallback(
    async (id: string) => {
      const detail = await fetchSession(id);
      if (session.connected) session.disconnect();
      setMode(detail.session.mode);
      session.loadHistory(detail.turns);
      await doConnect(id, detail.session.mode);
    },
    [session, doConnect]
  );

  const handleDeleteSession = useCallback(
    async (id: string) => {
      await deleteSession(id);
      refreshHistoryList();
    },
    [refreshHistoryList]
  );

  // --- push-to-talk (hold Space) ---

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.code !== "Space" || e.repeat) return;
      if (!session.connected || mode === "text" || !session.isPttActive) return;
      if (isTypingTarget(e.target)) return;
      e.preventDefault();
      session.pttStart();
    }
    function onKeyUp(e: KeyboardEvent) {
      if (e.code !== "Space") return;
      if (!session.isPttHeld()) return;
      session.pttStop();
    }
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [session, mode]);

  const modelChipLabel = !options
    ? "Model: —"
    : hasLlm(options, configured)
      ? `Model: ${settings.provider || options.provider.name || "provider"} · ${settings.llmModel || "default"}`
      : "No LLM configured";
  const fieldsDisabled = session.connected || session.state === "connecting";

  return (
    <div className="flex min-h-screen items-stretch max-[900px]:flex-col">
      <Sidebar
        sessions={sessions}
        loadError={sessionsError}
        onSelect={continueSession}
        onDelete={handleDeleteSession}
        userEmail={user.email}
      />
      {settingsOpen ? (
        <SettingsPanel
          settings={settings}
          onSettingsChange={onSettingsChange}
          onKeysChange={onKeysChange}
          options={optionsError ? null : options}
          mics={mics}
          disabled={fieldsDisabled}
          onBack={closeSettings}
          mode={mode}
          configured={configured}
          onCredentialsChanged={refreshConfigured}
        />
      ) : (
        <ChatPanel
          mode={mode}
          onModeChange={handleModeChange}
          connected={session.connected}
          connecting={session.state === "connecting"}
          state={session.state}
          stateText={session.stateText}
          statusLabel={session.statusLabel}
          lines={session.lines}
          lineCountLabel={session.lineCountLabel}
          micMuted={session.micMuted}
          onToggleMute={session.toggleMute}
          onConnect={handleConnect}
          onDisconnect={handleDisconnect}
          onOpenSettings={openSettings}
          modelChipLabel={modelChipLabel}
          onSendText={session.sendText}
        />
      )}
    </div>
  );
}

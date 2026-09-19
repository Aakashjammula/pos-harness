"use client";

import { effectiveModel, hasLlm, LLM_PROVIDERS } from "@/lib/llm";
import { useProviderModels } from "@/hooks/useProviderModels";
import { useTools } from "@/hooks/useTools";
import { ToolsPanel } from "@/components/ToolsPanel";
import { hashForTab, SettingsShell, type SettingsTab, tabFromHash } from "@/components/SettingsShell";
import { AccountTab } from "@/components/settings/AccountTab";
import { SecurityTab } from "@/components/settings/SecurityTab";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { deleteSession, fetchOptions, fetchSession, fetchSessions } from "@/lib/api";
import { AuthGuard } from "@/components/AuthGuard";
import type { CurrentUser } from "@/lib/auth";
import { fetchCredentialSummary } from "@/lib/credentials";
import { loadSettings, saveSettings } from "@/lib/settingsStore";
import { type ApiKeyFields, type OptionsResponse, type SessionMode, type SessionSummary, type Settings } from "@/lib/types";
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
  // What you last chose survives a reload. This subtree only renders after the sign-in check,
  // on the client, so reading browser storage here cannot cause a hydration mismatch.
  const [storedSettings, setSettings] = useState<Settings>(() => loadSettings(user.id));
  const [options, setOptions] = useState<OptionsResponse | null>(null);
  const [optionsError, setOptionsError] = useState(false);
  const [mics, setMics] = useState<MediaDeviceInfo[]>([]);
  const [sessions, setSessions] = useState<SessionSummary[] | null>(null);
  const [sessionsError, setSessionsError] = useState(false);
  const [settingsTab, setSettingsTab] = useState<SettingsTab | null>(null);
  const [toolsVersion, setToolsVersion] = useState(0);
  const [configured, setConfigured] = useState<string[]>([]);
  const [hints, setHints] = useState<Record<string, string>>({});
  // With exactly one LLM provider saved and none picked, use it: nothing to choose between.
  const onlySaved = useMemo(() => configured.filter((p) => LLM_PROVIDERS.includes(p)), [configured]);
  const settings = useMemo<Settings>(
    () => ({
      ...storedSettings,
      provider: storedSettings.provider || ((onlySaved.length === 1 ? onlySaved[0] : "") as Settings["provider"]),
    }),
    [storedSettings, onlySaved]
  );
  const [credentialsVersion, setCredentialsVersion] = useState(0);

  // --- initial data: /options, mic list, session history ---

  useEffect(() => {
    fetchOptions()
      .then((opts) => {
        setOptions(opts);
        // Server defaults only fill what you have not chosen yet.
        setSettings((s) => ({
          ...s,
          ttsEngine: s.ttsEngine || opts.defaults.tts_engine,
          llmModel: s.llmModel || opts.defaults.llm_model,
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
    setCredentialsVersion((v) => v + 1); // a saved/removed key can change which models exist
    try {
      const saved = await fetchCredentialSummary();
      setConfigured(saved.configured);
      setHints(saved.hints);
      // Fill the non-secret fields from what is saved, so the form does not look empty after a reload.
      const pub = saved.public;
      setSettings((s) => ({
        ...s,
        keys: {
          ...s.keys,
          localBaseUrl: s.keys.localBaseUrl || pub.local?.LOCAL_BASE_URL || "",
          azureEndpoint: s.keys.azureEndpoint || pub.azure?.AZURE_OPENAI_ENDPOINT || "",
          azureDeployment: s.keys.azureDeployment || pub.azure?.AZURE_OPENAI_DEPLOYMENT || "",
          bedrockRegion: s.keys.bedrockRegion || pub.bedrock?.AWS_REGION || "",
        },
      }));
    } catch {
      // leave the previous list in place -- a transient failure here
      // shouldn't blank out dots the user just saw as configured
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    refreshConfigured();
  }, [refreshConfigured]);

  useEffect(() => {
    saveSettings(user.id, storedSettings);
  }, [user.id, storedSettings]);

  // hash-based settings route, so #/settings survives refresh/back-forward
  useEffect(() => {
    const applyRoute = () => setSettingsTab(tabFromHash(window.location.hash));
    applyRoute();
    window.addEventListener("hashchange", applyRoute);
    return () => window.removeEventListener("hashchange", applyRoute);
  }, []);

  const openSettings = useCallback((tab: SettingsTab = "account") => {
    window.location.hash = hashForTab(tab);
  }, []);
  const closeSettings = useCallback(() => {
    window.location.hash = "";
  }, []);

  // Once connected, leave Settings automatically (mirrors the original's
  // ws "ready" handler forcing location.hash back to "").
  useEffect(() => {
    if (session.connected && settingsTab) closeSettings();
  }, [session.connected, settingsTab, closeSettings]);

  // Refresh the sidebar once a session becomes ready, and again ~1.5s
  // after each bot reply (the server generates a title off-thread, after
  // the first exchange, so the row's label may only be ready shortly
  // after this).
  const prevConnected = useRef(false);
  useEffect(() => {
    if (session.connected && !prevConnected.current) refreshHistoryList();
    prevConnected.current = session.connected;
  }, [session.connected, refreshHistoryList]);

  // A streamed text chat is created, then titled, after its first reply; both
  // arrive as events, so refresh the sidebar when the hook says so.
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- sets state only when the fetch completes
    if (session.historyVersion > 0) refreshHistoryList().catch(() => {});
  }, [session.historyVersion, refreshHistoryList]);

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

  const providerModels = useProviderModels(
    settings.provider,
    configured,
    credentialsVersion,
    options?.provider.name ?? ""
  );
  const toolsState = useTools(toolsVersion);
  const onToolsChanged = useCallback(() => {
    setToolsVersion((v) => v + 1);
    refreshConfigured(); // a saved/removed tool key changes the saved-credential list too
  }, [refreshConfigured]);
  const llmModel = effectiveModel(
    settings.provider,
    settings.llmModel,
    providerModels.models,
    providerModels.listable
  );

  const doConnect = useCallback(
    async (resumeSessionId: string | null, connectMode: SessionMode) => {
      await session.connect(
        {
          mode: connectMode,
          ttsEngine: settings.ttsEngine,
          ttsVoice: settings.ttsVoice,
          llmModel,
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
    [settings, llmModel, session, refreshMics]
  );

  // Switching to voice carries the current chat along: connecting resumes it.
  const resumeOnConnect = useRef<string | null>(null);

  const handleConnect = useCallback(() => {
    const resume = resumeOnConnect.current;
    resumeOnConnect.current = null;
    doConnect(resume, "voice");
  }, [doConnect]);

  const handleDisconnect = useCallback(() => {
    session.disconnect();
  }, [session]);

  // Text needs no connection, so switching modes only changes what the panel offers. The chat
  // (its session id and transcript) carries over; voice still needs a Connect for the microphone.
  const handleModeChange = useCallback(
    (next: SessionMode) => {
      if (next === mode) return;
      const sessionId = session.sessionId;
      if (session.connected) session.disconnect();
      setMode(next);
      if (next === "text") session.startTextChat(sessionId, false);
      else resumeOnConnect.current = sessionId;
    },
    [mode, session]
  );

  const handleNewChat = useCallback(() => {
    if (session.connected) session.disconnect();
    resumeOnConnect.current = null;
    session.startTextChat(null, true);
  }, [session]);

  const handleSendText = useCallback(
    (text: string) => session.sendText(text, { provider: settings.provider || undefined, llmModel }),
    [session, settings.provider, llmModel]
  );

  const continueSession = useCallback(
    async (id: string) => {
      const detail = await fetchSession(id);
      if (session.connected) session.disconnect();
      setMode(detail.session.mode);
      session.loadHistory(detail.turns);
      if (detail.session.mode === "text") session.startTextChat(id, false);
      else await doConnect(id, "voice");
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
      ? `Model: ${settings.provider || options.provider.name || "provider"} · ${llmModel || "default"}`
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
        onOpenSettings={() => openSettings("account")}
      />
      {settingsTab ? (
        <SettingsShell tab={settingsTab} onTab={openSettings} onBack={closeSettings}>
          {settingsTab === "account" && <AccountTab />}
          {settingsTab === "security" && <SecurityTab />}
          {settingsTab === "model" && (
            <SettingsPanel
              embedded
              settings={settings}
              onSettingsChange={onSettingsChange}
              onKeysChange={onKeysChange}
              options={optionsError ? null : options}
              mics={mics}
              disabled={fieldsDisabled}
              onBack={closeSettings}
              mode={mode}
              configured={configured}
              providerModels={providerModels}
              hints={hints}
              llmModel={llmModel}
              tools={toolsState.tools}
              onCredentialsChanged={refreshConfigured}
            />
          )}
          {settingsTab === "tools" && (
            <ToolsPanel
              embedded
              tools={toolsState.tools}
              loading={toolsState.loading}
              error={toolsState.error}
              onChanged={onToolsChanged}
              onBack={closeSettings}
            />
          )}
        </SettingsShell>
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
          onOpenSettings={() => openSettings("model")}
          onOpenTools={() => openSettings("tools")}
          modelChipLabel={modelChipLabel}
          onSendText={handleSendText}
          onNewChat={handleNewChat}
          replying={session.replying}
        />
      )}
    </div>
  );
}

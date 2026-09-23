"use client";

import { useCallback, useEffect, useState } from "react";
import { useChatSession } from "@/hooks/useChatSession";
import { useWorkspaceFolder } from "@/hooks/useWorkspaceFolder";
import { deleteSession, fetchSession, fetchSessions, type SessionSummary } from "@/lib/sessions";
import { fetchModels, type ModelInfo } from "@/lib/models";
import { Sidebar } from "@/components/Sidebar";
import { ChatPanel } from "@/components/ChatPanel";
import { SettingsPage } from "@/components/SettingsPage";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { TracePage } from "@/components/TracePage";

const LEVELS = ["Low", "Medium", "High"];

export default function Home() {
  const session = useChatSession();
  const workspace = useWorkspaceFolder();
  const [level, setLevel] = useState(LEVELS[2]);
  const [showSettings, setShowSettings] = useState(false);
  const [sessions, setSessions] = useState<SessionSummary[] | null>(null);
  // The chat awaiting a delete confirmation; null when no dialog is open.
  const [pendingDelete, setPendingDelete] = useState<SessionSummary | null>(null);
  const [showTrace, setShowTrace] = useState(false);
  // The models the backend can call, and which one is selected. Names come
  // from the backend's .env; everything else is looked up there.
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [model, setModel] = useState("");
  // Set when the backend has no usable provider credentials.
  const [configError, setConfigError] = useState<string | null>(null);

  // Every folder's chats, not just the open one -- the sidebar groups them
  // by folder, so picking a chat also says which folder it belongs to.
  const refreshSessions = useCallback(async () => {
    try {
      setSessions(await fetchSessions());
    } catch {
      setSessions([]); // a transient failure shows an empty list, not a permanent spinner
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- sets state only when the fetch completes
    refreshSessions();
  }, [refreshSessions]);

  useEffect(() => {
    let cancelled = false;
    fetchModels()
      .then((d) => {
        if (cancelled) return;
         
        setModels(d.models);
        setModel(d.default);
        setConfigError(d.config_error);
      })
      .catch(() => {
        if (!cancelled) setConfigError("Can't reach the backend. Is it running?");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- sets state only when the fetch completes
    if (session.historyVersion > 0) refreshSessions();
  }, [session.historyVersion, refreshSessions]);

  // How full the model's context is. Not `input_tokens` -- that sums every
  // round of a turn, so a turn with 11 tool calls reports ~121k for a thread
  // that is really ~17k. `context_tokens` is the last round's input, which is
  // the thread as it now stands.
  const lastUsage = [...session.lines].reverse().find((l) => l.usage)?.usage;
  const contextUsed = lastUsage?.context_tokens ?? undefined;
  const contextWindow = lastUsage?.context_window ?? undefined;
  // What this chat has cost so far, summed over its replies.
  const chatCostUsd = session.lines.reduce((sum, l) => sum + (l.usage?.cost_usd ?? 0), 0) || undefined;

  const handleSendText = useCallback(
    (text: string) =>
      session.sendText(text, { folder: workspace.folderPath, reasoningEffort: level, model }),
    [session, workspace.folderPath, level, model]
  );

  // Opening a past chat also switches to the folder it belongs to -- otherwise
  // the next message in it would run against whichever folder happened to be
  // open, and get stored under that one.
  const handleSelectSession = useCallback(
    async (summary: SessionSummary) => {
      try {
        const { turns } = await fetchSession(summary.id);
        workspace.setFolder(summary.folder);
        session.loadHistory(summary.id, turns);
      } catch {
        // leave the current transcript alone if the reload fails
      }
    },
    [session, workspace]
  );

  const handleNewChatInFolder = useCallback(
    (folder: string) => {
      workspace.setFolder(folder);
      session.newChat();
    },
    [session, workspace]
  );

  // Deleting is permanent -- the chat, its stored turns and its memory all
  // go -- so it asks first. Deleting the chat you're looking at leaves the
  // transcript pointing at a thread that no longer exists, so start a fresh one.
  const handleConfirmDelete = useCallback(async () => {
    const summary = pendingDelete;
    setPendingDelete(null);
    if (!summary) return;
    try {
      await deleteSession(summary.id);
      if (summary.id === session.threadId) session.newChat();
      await refreshSessions();
    } catch {
      // leave the list alone if the delete failed
    }
  }, [pendingDelete, session, refreshSessions]);

  // A new chat belongs to the folder you're in; with no folder open there's
  // nowhere to put it, so ask for one instead of starting a chat that can't run.
  const handleNewChat = useCallback(() => {
    if (!workspace.folderPath) {
      workspace.openFolder();
      return;
    }
    session.newChat();
  }, [session, workspace]);

  return (
    <div className="flex min-h-screen items-stretch max-[900px]:flex-col">
      <Sidebar
        sessions={sessions}
        activeFolder={workspace.folderPath}
        activeSessionId={session.threadId}
        onSelectSession={handleSelectSession}
        onNewChat={handleNewChat}
        onNewChatInFolder={handleNewChatInFolder}
        onDeleteSession={setPendingDelete}
        onOpenFolder={workspace.openFolder}
        onOpenSettings={() => setShowSettings(true)}
      />
      {showTrace ? (
        <TracePage lines={session.lines} onBack={() => setShowTrace(false)} />
      ) : showSettings ? (
        <SettingsPage
          models={models}
          model={model}
          onModelChange={setModel}
          levels={LEVELS}
          level={level}
          onLevelChange={setLevel}
          onBack={() => setShowSettings(false)}
        />
      ) : (
        <ChatPanel
          lines={session.lines}
          lineCountLabel={session.lineCountLabel}
          onSendText={handleSendText}
          replying={session.replying}
          activity={session.activity}
          contextUsed={contextUsed}
          contextWindow={contextWindow}
          chatCostUsd={chatCostUsd}
          model={model}
          models={models.map((m) => m.name)}
          onModelChange={setModel}
          levels={LEVELS}
          level={level}
          onLevelChange={setLevel}
          folderName={workspace.folderName}
          folderReady={workspace.ready}
          onOpenFolder={workspace.openFolder}
          folderPicking={workspace.picking}
          folderError={workspace.error}
          configError={configError}
          onOpenTrace={() => setShowTrace(true)}
        />
      )}
      {pendingDelete && (
        <ConfirmDialog
          title={`Delete "${pendingDelete.title || "Untitled chat"}"?`}
          body="This removes the chat, its messages and everything the agent remembered from it. It can't be undone."
          confirmLabel="Delete"
          danger
          onConfirm={handleConfirmDelete}
          onCancel={() => setPendingDelete(null)}
        />
      )}
    </div>
  );
}

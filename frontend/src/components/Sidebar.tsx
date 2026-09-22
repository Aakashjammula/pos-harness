"use client";

import { useEffect, useMemo, useState } from "react";
import type { SessionSummary } from "@/lib/sessions";
import { BrandMark, FolderIcon, GearIcon, TrashIcon } from "./icons";

interface SidebarProps {
  sessions: SessionSummary[] | null; // null = still loading / failed
  activeFolder: string | null;
  activeSessionId: string | null;
  onSelectSession: (session: SessionSummary) => void;
  onNewChat: () => void;
  onNewChatInFolder: (folder: string) => void;
  onDeleteSession: (session: SessionSummary) => void;
  onOpenFolder: () => void;
  onOpenSettings: () => void;
}

/** Everything after the last path separator -- what to call a folder in a
 * list that's only so wide. */
function basename(path: string): string {
  return path.replace(/[/\\]+$/, "").split(/[/\\]/).pop() || path;
}

/**
 * Every folder you've chatted in, collapsed by default -- expand one to
 * see its chats. The folder you're currently in starts expanded. Chats
 * with no folder are grouped under "Other".
 */
export function Sidebar({
  sessions,
  activeFolder,
  activeSessionId,
  onSelectSession,
  onNewChat,
  onNewChatInFolder,
  onDeleteSession,
  onOpenFolder,
  onOpenSettings,
}: SidebarProps) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  // Folders in most-recent-chat-first order, since `sessions` arrives newest first.
  const groups = useMemo(() => {
    const byFolder = new Map<string, SessionSummary[]>();
    for (const s of sessions ?? []) {
      const key = s.folder ?? "";
      const existing = byFolder.get(key);
      if (existing) existing.push(s);
      else byFolder.set(key, [s]);
    }
    return [...byFolder.entries()];
  }, [sessions]);

  // The folder you're working in opens itself, so its chats are one click away.
  useEffect(() => {
    if (!activeFolder) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- expands the active folder when it changes
    setExpanded((prev) => (prev.has(activeFolder) ? prev : new Set(prev).add(activeFolder)));
  }, [activeFolder]);

  const toggle = (folder: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(folder)) next.delete(folder);
      else next.add(folder);
      return next;
    });

  return (
    <aside className="flex w-[260px] shrink-0 flex-col gap-0.5 overflow-y-auto bg-sidebar-bg p-2 max-[900px]:w-auto max-[900px]:max-h-[220px] max-[900px]:border-b max-[900px]:border-border">
      <div className="flex items-center gap-2 px-2 pt-2 pb-3.5">
        <BrandMark className="h-5 w-auto shrink-0" />
        <span className="font-mono text-sm font-semibold tracking-tight">pos</span>
      </div>

      <button
        type="button"
        onClick={onNewChat}
        className="mx-2 rounded-lg border border-border bg-surface-sunken px-2.5 py-1.5 text-left text-[13px] font-medium text-text-muted transition-colors hover:border-text-muted hover:text-text"
      >
        + New chat
      </button>

      <button
        type="button"
        onClick={onOpenFolder}
        className="mx-2 mt-1.5 flex items-center gap-2 rounded-lg px-2.5 py-1.5 text-left text-[13px] font-medium text-text-muted transition-colors hover:bg-surface-sunken hover:text-text"
      >
        <FolderIcon className="h-4 w-4 shrink-0" />
        Open folder
      </button>

      <div className="mt-2 flex flex-col gap-0.5">
        {sessions === null ? (
          <div className="px-2 py-1 text-[12.5px] text-text-faint">Loading…</div>
        ) : groups.length === 0 ? (
          <div className="px-2 py-1 text-[12.5px] text-text-faint">No chats yet. Open a folder to start one.</div>
        ) : (
          groups.map(([folder, items]) => {
            const key = folder || "__other__";
            const isOpen = expanded.has(folder);
            return (
              <div key={key}>
                <div className="flex items-center justify-between gap-1 px-2 py-1">
                  <button
                    type="button"
                    onClick={() => toggle(folder)}
                    aria-expanded={isOpen}
                    title={folder || "Chats with no folder"}
                    className={`flex min-w-0 flex-1 items-center gap-1 text-left text-[12px] transition-colors hover:text-text ${
                      folder === activeFolder ? "font-medium text-text" : "text-text-faint"
                    }`}
                  >
                    <span className="shrink-0 text-[10px] leading-none">{isOpen ? "▾" : "▸"}</span>
                    <span className="truncate">{folder ? basename(folder) : "Other"}</span>
                    <span className="shrink-0 text-text-faint">({items.length})</span>
                  </button>
                  {folder && (
                    <button
                      type="button"
                      onClick={() => onNewChatInFolder(folder)}
                      aria-label={`New chat in ${basename(folder)}`}
                      title={`New chat in ${basename(folder)}`}
                      className="shrink-0 rounded px-1 text-[15px] leading-none text-text-faint hover:text-text"
                    >
                      +
                    </button>
                  )}
                </div>
                {isOpen &&
                  items.map((s) => (
                    // A row, not a button: the delete control is a button of
                    // its own, and buttons can't be nested.
                    <div
                      key={s.id}
                      className={`group flex items-center rounded-lg pr-1 pl-6 transition-colors hover:bg-surface-sunken ${
                        s.id === activeSessionId ? "bg-surface-sunken" : ""
                      }`}
                    >
                      <button
                        type="button"
                        onClick={() => onSelectSession(s)}
                        aria-current={s.id === activeSessionId ? "true" : undefined}
                        className={`min-w-0 flex-1 truncate py-1.5 text-left text-[13px] ${
                          s.id === activeSessionId ? "text-text" : "text-text-muted"
                        }`}
                      >
                        {s.title || "Untitled chat"}
                      </button>
                      <button
                        type="button"
                        onClick={() => onDeleteSession(s)}
                        aria-label={`Delete ${s.title || "Untitled chat"}`}
                        title="Delete chat"
                        // Hidden until the row is hovered or the button itself
                        // is focused, so it stays reachable by keyboard.
                        className="shrink-0 rounded p-1 text-text-faint opacity-0 transition-opacity group-hover:opacity-100 hover:text-danger focus-visible:opacity-100"
                      >
                        <TrashIcon className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  ))}
              </div>
            );
          })
        )}
      </div>

      <div className="mt-auto border-t border-border px-2 pt-2">
        <button
          type="button"
          onClick={onOpenSettings}
          className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-[13px] text-text-muted hover:bg-surface-sunken hover:text-text"
        >
          <GearIcon className="h-4 w-4 shrink-0" />
          Settings
        </button>
      </div>
    </aside>
  );
}

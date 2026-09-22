"use client";

import { useCallback, useEffect, useState } from "react";
import { API_URL } from "@/lib/config";

const STORAGE_KEY = "pos.folderPath.v1";

/**
 * Like Codex/Claude Code: work happens inside a folder you explicitly
 * open. The native OS folder dialog is opened by the *backend*
 * (POST /fs/pick): the browser's own File System Access API can only hand
 * back a folder's name, never a real path -- deliberately, for security --
 * which is no use to the agent's filesystem tool.
 *
 * The chosen folder is remembered in this browser, so a reload keeps your
 * folder instead of resetting. `ready` is false until that restore has
 * run, so the UI can avoid flashing the "no folder open" state first.
 */
export function useWorkspaceFolder() {
  const [folderPath, setFolderPath] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [picking, setPicking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Read on mount rather than in useState's initializer: this component is
  // prerendered on the server, where localStorage doesn't exist, and reading
  // it during that first render would mismatch on hydration.
  useEffect(() => {
    let saved: string | null = null;
    try {
      saved = localStorage.getItem(STORAGE_KEY);
    } catch {
      // private mode / blocked storage: just start with no folder open
    }
    // eslint-disable-next-line react-hooks/set-state-in-effect -- restoring persisted state on mount
    if (saved) setFolderPath(saved);
    setReady(true); // unblocks first paint once the restore has settled
  }, []);

  const folderName = folderPath ? folderPath.replace(/[/\\]+$/, "").split(/[/\\]/).pop() || folderPath : null;

  /** Switches to a folder we already know the path of (e.g. opening a past
   * chat that belongs to a different folder). */
  const setFolder = useCallback((path: string | null) => {
    setFolderPath(path);
    try {
      if (path) localStorage.setItem(STORAGE_KEY, path);
      else localStorage.removeItem(STORAGE_KEY);
    } catch {
      // storage blocked: the folder still works for this session
    }
  }, []);

  const openFolder = useCallback(async () => {
    setError(null);
    setPicking(true);
    try {
      const res = await fetch(`${API_URL}/fs/pick`, { method: "POST" });
      if (!res.ok) throw new Error(`Couldn't open the folder dialog (HTTP ${res.status}).`);
      const data = (await res.json()) as { path: string | null };
      if (data.path) setFolder(data.path);
      // no path back = the dialog was cancelled; leave the current folder as-is
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't reach the backend.");
    } finally {
      setPicking(false);
    }
  }, [setFolder]);

  return { folderPath, folderName, ready, picking, error, openFolder, setFolder };
}

"use client";

import { useCallback, useEffect, useState } from "react";
import { fetchDir, type FileEntry } from "@/lib/files";

interface FileTreeProps {
  folder: string;
  /** Bumps when files may have changed; every expanded level reloads. */
  version: number;
  activePath: string | null;
  onOpenFile: (path: string) => void;
  onPreviewKey: (key: string) => void;
}

/**
 * The open folder as a tree, loaded one level at a time.
 *
 * Only expanded folders are ever fetched, so a node_modules beside the
 * code costs nothing until someone opens it.
 */
export function FileTree({ folder, version, activePath, onOpenFile, onPreviewKey }: FileTreeProps) {
  // Each loaded folder's entries, by path ("" is the root).
  const [listings, setListings] = useState<Record<string, FileEntry[]>>({});
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set([""]));
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(
    async (path: string) => {
      try {
        const listing = await fetchDir(folder, path);
        setListings((prev) => ({ ...prev, [path]: listing.entries }));
        onPreviewKey(listing.preview_key);
        setError(null);
      } catch (e) {
        if (path === "") setError((e as Error).message);
        // A folder deleted while expanded just closes.
        else setExpanded((prev) => new Set([...prev].filter((p) => p !== path)));
      }
    },
    [folder, onPreviewKey]
  );

  // Reload everything open, on mount and whenever files may have changed.
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- sets state only when the fetch completes
    for (const path of expanded) load(path);
    // `expanded` is read, not watched: toggling loads its own level below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [load, version]);

  function toggle(path: string) {
    const opening = !expanded.has(path);
    setExpanded((prev) => {
      const next = new Set(prev);
      if (opening) next.add(path);
      else next.delete(path);
      return next;
    });
    if (opening) load(path);
  }

  function renderLevel(path: string, depth: number) {
    const entries = listings[path];
    if (!entries) {
      return (
        <div className="py-0.5 text-[12px] text-text-faint" style={{ paddingLeft: 8 + depth * 12 }}>
          …
        </div>
      );
    }
    if (entries.length === 0 && depth === 0) {
      return <div className="px-2 py-1 text-[12px] text-text-faint">Empty folder.</div>;
    }
    return entries.map((entry) => {
      const isDir = entry.type === "dir";
      const open = expanded.has(entry.path);
      const active = entry.path === activePath;
      return (
        <div key={entry.path}>
          <button
            type="button"
            disabled={entry.locked}
            onClick={() => (isDir ? toggle(entry.path) : onOpenFile(entry.path))}
            title={entry.locked ? `${entry.name} — not shown` : entry.path}
            className={`flex w-full min-w-0 items-center gap-1 rounded py-[3px] pr-2 text-left text-[12.5px] disabled:cursor-not-allowed ${
              active ? "bg-surface-sunken text-text" : "text-text-muted hover:bg-surface-sunken hover:text-text"
            } ${entry.heavy || entry.locked ? "opacity-50" : ""}`}
            style={{ paddingLeft: 8 + depth * 12 }}
          >
            <span className="w-3 shrink-0 text-center text-[9px] leading-none text-text-faint">
              {isDir ? (open ? "▾" : "▸") : ""}
            </span>
            <span className="min-w-0 truncate">{entry.name}</span>
            {entry.locked && <span className="shrink-0 text-[10px] text-text-faint">🔒</span>}
          </button>
          {isDir && open && renderLevel(entry.path, depth + 1)}
        </div>
      );
    });
  }

  if (error) {
    return (
      <div role="alert" className="p-3 text-[12px] text-danger">
        {error}
      </div>
    );
  }
  return <div className="py-1">{renderLevel("", 0)}</div>;
}

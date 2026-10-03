"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useRef, useState } from "react";
import { fetchFile, previewUrl } from "@/lib/files";
import { FileTree } from "./FileTree";

// CodeMirror is browser-only and the heaviest thing in the UI, so it gets
// its own bundle, fetched the first time a file is opened.
const CodeView = dynamic(() => import("./CodeView"), {
  ssr: false,
  loading: () => <div className="p-3 text-[12px] text-text-faint">Loading viewer…</div>,
});

const WIDTH_KEY = "pos.filePanelWidth.v1";
const MIN_WIDTH = 360;
const DEFAULT_WIDTH = 640;

/** A file to show, from outside the panel (e.g. a click in the chat). The
 * nonce makes asking for the same file twice still count. */
export interface OpenRequest {
  path: string;
  nonce: number;
}

interface FilePanelProps {
  folder: string;
  /** Bumps when the agent may have changed files. */
  version: number;
  openRequest: OpenRequest | null;
  onClose: () => void;
}

type Loaded = { content: string } | { error: string };

const isHtml = (path: string) => /\.html?$/i.test(path);
const baseName = (path: string) => path.split("/").pop() || path;

function readWidth(): number {
  try {
    const saved = Number(localStorage.getItem(WIDTH_KEY));
    return saved >= MIN_WIDTH ? saved : DEFAULT_WIDTH;
  } catch {
    return DEFAULT_WIDTH; // storage blocked: the default is fine
  }
}

/**
 * The open folder beside the chat: a tree, and the files you open from it
 * as tabs. HTML files can also run, sandboxed, in a preview.
 *
 * Rendered with `key={folder}`, so switching folders starts it afresh
 * rather than showing one folder's tabs over another's tree.
 */
export function FilePanel({ folder, version, openRequest, onClose }: FilePanelProps) {
  const [tabs, setTabs] = useState<string[]>([]);
  const [active, setActive] = useState<string | null>(null);
  const [loaded, setLoaded] = useState<Record<string, Loaded>>({});
  const [previewing, setPreviewing] = useState<Record<string, boolean>>({});
  const [previewKey, setPreviewKey] = useState<string | null>(null);
  const [width, setWidth] = useState(DEFAULT_WIDTH);
  const dragging = useRef(false);

  useEffect(() => {
    setWidth(readWidth());
  }, []);

  const load = useCallback(
    async (path: string) => {
      try {
        const content = await fetchFile(folder, path);
        setLoaded((prev) => ({ ...prev, [path]: { content } }));
      } catch (e) {
        setLoaded((prev) => ({ ...prev, [path]: { error: (e as Error).message } }));
      }
    },
    [folder],
  );

  const openFile = useCallback(
    (path: string) => {
      setTabs((prev) => (prev.includes(path) ? prev : [...prev, path]));
      setActive(path);
      load(path);
    },
    [load],
  );

  useEffect(() => {
    if (openRequest) openFile(openRequest.path);
  }, [openRequest, openFile]);

  // The agent may have just rewritten what is on screen.
  useEffect(() => {
    if (version === 0) return;
    for (const path of tabs) load(path);
    // Reload on `version` only; opening a tab loads it itself.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [version]);

  function closeTab(path: string) {
    const index = tabs.indexOf(path);
    const next = tabs.filter((t) => t !== path);
    setTabs(next);
    if (active === path) setActive(next[Math.min(index, next.length - 1)] ?? null);
  }

  // Dragging the left edge resizes; the width is remembered in this browser.
  function startDrag(e: React.PointerEvent) {
    dragging.current = true;
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  }
  function onDrag(e: React.PointerEvent) {
    if (!dragging.current) return;
    const next = Math.max(MIN_WIDTH, Math.min(window.innerWidth - 480, window.innerWidth - e.clientX));
    setWidth(next);
  }
  function endDrag() {
    if (!dragging.current) return;
    dragging.current = false;
    try {
      localStorage.setItem(WIDTH_KEY, String(width));
    } catch {
      // storage blocked: the width still holds for this session
    }
  }

  const current = active ? loaded[active] : undefined;
  const showPreview = active !== null && isHtml(active) && previewing[active] && previewKey !== null;

  return (
    <aside
      className="relative flex h-screen shrink-0 border-l border-border bg-bg max-[900px]:h-[60vh] max-[900px]:w-full!"
      style={{ width }}
      aria-label="Files"
    >
      <div
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize the file panel"
        onPointerDown={startDrag}
        onPointerMove={onDrag}
        onPointerUp={endDrag}
        className="absolute inset-y-0 -left-1 z-10 w-2 cursor-col-resize hover:bg-accent/30 max-[900px]:hidden"
      />

      {/* explorer */}
      <div className="flex w-[220px] shrink-0 flex-col border-r border-border bg-sidebar-bg">
        <div className="flex items-center justify-between px-3 py-3">
          <span className="text-[11px] font-semibold tracking-wide text-text-faint uppercase">Explorer</span>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close the file panel"
            className="rounded px-1 text-[13px] text-text-faint hover:text-text"
          >
            ✕
          </button>
        </div>
        <div className="chat-scroll min-h-0 flex-1 overflow-y-auto">
          <FileTree
            folder={folder}
            version={version}
            activePath={active}
            onOpenFile={openFile}
            onPreviewKey={setPreviewKey}
          />
        </div>
      </div>

      {/* viewer */}
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="chat-scroll flex min-h-[41px] items-stretch overflow-x-auto border-b border-border">
          {tabs.map((path) => (
            <div
              key={path}
              className={`group flex shrink-0 items-center gap-1.5 border-r border-border pr-1.5 pl-3 text-[12.5px] ${
                path === active ? "bg-bg text-text" : "bg-sidebar-bg text-text-muted hover:text-text"
              }`}
            >
              <button type="button" onClick={() => setActive(path)} title={path} className="max-w-[180px] truncate py-2">
                {baseName(path)}
              </button>
              <button
                type="button"
                onClick={() => closeTab(path)}
                aria-label={`Close ${baseName(path)}`}
                className="rounded px-1 text-[11px] text-text-faint opacity-0 group-hover:opacity-100 hover:text-text"
              >
                ✕
              </button>
            </div>
          ))}
        </div>

        {active && (
          <div className="flex items-center gap-2 border-b border-border px-3 py-1.5 text-[11.5px] text-text-faint">
            <span className="min-w-0 flex-1 truncate font-mono">{active}</span>
            {isHtml(active) && previewKey && (
              <>
                <div className="flex overflow-hidden rounded-md border border-border">
                  {(["Code", "Preview"] as const).map((label) => {
                    const on = (label === "Preview") === !!previewing[active];
                    return (
                      <button
                        key={label}
                        type="button"
                        onClick={() => setPreviewing((p) => ({ ...p, [active]: label === "Preview" }))}
                        className={`px-2 py-0.5 ${on ? "bg-surface-sunken text-text" : "hover:text-text"}`}
                      >
                        {label}
                      </button>
                    );
                  })}
                </div>
                <a
                  href={previewUrl(previewKey, active)}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-text"
                >
                  Open in new tab ↗
                </a>
              </>
            )}
          </div>
        )}

        <div className="min-h-0 flex-1 overflow-hidden">
          {!active ? (
            <div className="p-6 text-center text-[13px] text-text-faint">
              Pick a file on the left, or click one the agent touched in the chat.
            </div>
          ) : showPreview ? (
            // The page runs as itself but sandboxed: no allow-same-origin, so it
            // gets an opaque origin and cannot reach this app. The server sends
            // the same sandbox as a CSP, which also covers "open in new tab".
            <iframe
              key={`${active}-${version}`}
              title={`Preview of ${active}`}
              src={previewUrl(previewKey!, active)}
              sandbox="allow-scripts allow-modals allow-forms allow-popups"
              className="h-full w-full border-0 bg-white"
            />
          ) : !current ? (
            <div className="p-3 text-[12px] text-text-faint">Loading…</div>
          ) : "error" in current ? (
            <div role="alert" className="p-6 text-center text-[13px] text-text-faint">
              {current.error}
            </div>
          ) : (
            <CodeView path={active} content={current.content} />
          )}
        </div>
      </div>
    </aside>
  );
}

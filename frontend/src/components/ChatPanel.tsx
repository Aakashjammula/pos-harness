"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { TranscriptLine } from "@/lib/types";
import { formatUsageLine } from "@/lib/format";
import { Dropdown } from "./Dropdown";
import { PlusMenu } from "./PlusMenu";
import { FolderIcon } from "./icons";
import { ContextMeter } from "./ContextMeter";
import { Markdown } from "./Markdown";
import { ActivityTrail, ThinkingRow } from "./ActivityTrail";
import { ComposerMenu, type MenuItem } from "./ComposerMenu";
import { searchFiles } from "@/lib/files";

interface ChatPanelProps {
  lines: TranscriptLine[];
  lineCountLabel: string;
  onSendText: (text: string) => void;
  replying: boolean; // a reply is still streaming in
  stopping: boolean; // Stop was pressed; the reply is winding down
  onStop: () => void;
  activity: string | null; // e.g. "read_file /skills/pdf/SKILL.md" while a tool runs
  contextUsed?: number;
  contextWindow?: number;
  chatCostUsd?: number;
  model: string;
  models: string[]; // what the backend offers; one entry means no picker
  onModelChange: (model: string) => void;
  levels: string[];
  level: string;
  onLevelChange: (level: string) => void;
  folderName: string | null; // null = no folder open yet -- chat is locked until then
  folderPath: string | null; // the same folder's real path, for @-mention search
  onNewChat: () => void;
  folderReady: boolean; // false until the saved folder has been restored
  onOpenFolder: () => void;
  folderPicking: boolean; // the native OS dialog is open, waiting on you
  folderError: string | null;
  configError: string | null; // provider not set up; chatting will fail
  onOpenTrace: () => void;
  /** Opens a project file in the file panel. */
  onOpenFile: (path: string) => void;
  filesOpen: boolean;
  onToggleFiles: () => void;
}

/** What "/" offers at the start of the box. */
const COMMANDS: MenuItem[] = [
  { key: "new", label: "/new", detail: "Start a new chat in this folder", icon: "+" },
  { key: "files", label: "/files", detail: "Show or hide the file panel", icon: "▤" },
  { key: "model", label: "/model", detail: "Switch model", icon: "◇" },
  { key: "help", label: "/help", detail: "What / and @ can do", icon: "?" },
];

/** A popup the composer is showing: what it lists, the text it is
 * completing, and where that text starts in the box. */
interface MenuState {
  kind: "file" | "command" | "model";
  query: string;
  start: number;
}

/** Which popup, if any, the text before the cursor calls for: "/word" at
 * the very start, or "@word" after a space or at the start. */
function menuFor(value: string, caret: number): MenuState | null {
  const before = value.slice(0, caret);
  const command = /^\/(\S*)$/.exec(before);
  if (command) return { kind: "command", query: command[1], start: 0 };
  const mention = /(?:^|\s)@(\S*)$/.exec(before);
  if (mention) return { kind: "file", query: mention[1], start: caret - mention[1].length - 1 };
  return null;
}

/** The part of a reply that belongs in the bubble.
 *
 * `line.text` accumulates every round's tokens, so on a turn whose earlier
 * rounds the activity trail has already printed, only the last round's text
 * is left to show. Without rounds -- an older turn, or one still streaming
 * -- it is all of it.
 */
function finalText(line: TranscriptLine): string {
  const rounds = line.usage?.rounds;
  if (!rounds || rounds.length < 2) return line.text;
  return rounds[rounds.length - 1].text || line.text;
}

export function ChatPanel({
  lines,
  lineCountLabel,
  onSendText,
  replying,
  stopping,
  onStop,
  activity,
  contextUsed,
  contextWindow,
  chatCostUsd,
  model,
  models,
  onModelChange,
  levels,
  level,
  onLevelChange,
  folderName,
  folderPath,
  onNewChat,
  folderReady,
  onOpenFolder,
  folderPicking,
  folderError,
  configError,
  onOpenTrace,
  onOpenFile,
  filesOpen,
  onToggleFiles,
}: ChatPanelProps) {
  const transcriptRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [textValue, setTextValue] = useState("");
  // A message sent while a reply was still running; it goes when that ends.
  const [queued, setQueued] = useState<string | null>(null);
  // The "/" or "@" popup, its file results, and the highlighted row.
  const [menu, setMenu] = useState<MenuState | null>(null);
  const [fileItems, setFileItems] = useState<MenuItem[]>([]);
  const [menuActive, setMenuActive] = useState(0);
  const [showHelp, setShowHelp] = useState(false);

  const menuItems: MenuItem[] = useMemo(() => {
    if (!menu) return [];
    if (menu.kind === "command") return COMMANDS.filter((c) => c.label.startsWith(`/${menu.query}`));
    if (menu.kind === "model") return models.map((m) => ({ key: m, label: m, icon: m === model ? "✓" : "" }));
    return fileItems;
  }, [menu, models, model, fileItems]);

  // Files come from the server, a moment after typing pauses.
  const fileQuery = menu?.kind === "file" ? menu.query : null;
  useEffect(() => {
    if (fileQuery === null || !folderPath) return;
    const controller = new AbortController();
    const timer = setTimeout(() => {
      searchFiles(folderPath, fileQuery, controller.signal)
        .then((entries) =>
          setFileItems(
            entries.map((e) => ({
              key: e.path,
              label: e.type === "dir" ? `${e.path}/` : e.path,
              icon: e.type === "dir" ? "▸" : "·",
            }))
          )
        )
        .catch(() => {
          // aborted by the next keystroke, or the folder went away: keep the old list
        });
    }, 80);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [fileQuery, folderPath]);

  function updateText(value: string, caret: number) {
    setTextValue(value);
    const next = menuFor(value, caret);
    // The model list was opened by a command, not by typing; typing leaves it.
    setMenu((prev) => (prev?.kind === "model" && !next ? prev : next));
    setMenuActive(0);
  }

  function pick(item: MenuItem) {
    if (!menu) return;
    if (menu.kind === "model") {
      onModelChange(item.key);
      setMenu(null);
      return;
    }
    if (menu.kind === "command") {
      setTextValue("");
      setMenu(null);
      if (item.key === "new") onNewChat();
      else if (item.key === "files") onToggleFiles();
      else if (item.key === "help") setShowHelp(true);
      else if (item.key === "model") setMenu({ kind: "model", query: "", start: 0 });
      return;
    }
    // A file: replace "@what-was-typed" with its path. A folder stays open,
    // listing what is inside it, the way Claude Code drills down.
    const el = textareaRef.current;
    const caret = el?.selectionStart ?? textValue.length;
    const isDir = item.label.endsWith("/");
    const inserted = isDir ? `@${item.label}` : `@${item.label} `;
    const value = textValue.slice(0, menu.start) + inserted + textValue.slice(caret);
    const at = menu.start + inserted.length;
    setTextValue(value);
    setMenu(isDir ? { kind: "file", query: item.label, start: menu.start } : null);
    setMenuActive(0);
    requestAnimationFrame(() => {
      el?.focus();
      el?.setSelectionRange(at, at);
    });
  }
  const [files, setFiles] = useState<File[]>([]);
  const [toolsEnabled, setToolsEnabled] = useState(false);
  // Shown after the reasoning level is changed part-way through a chat.
  // OpenAI lists reasoning.effort among the settings that invalidate a
  // cached prefix, so the next message reprocesses the whole thread at the
  // uncached rate. Cleared once that message is sent.
  const [levelChanged, setLevelChanged] = useState(false);
  const started = lines.some((l) => l.usage);

  function handleLevelChange(next: string) {
    if (next !== level && started) setLevelChanged(true);
    onLevelChange(next);
  }
  // Locked until a folder is open. While the saved folder is still being
  // restored we lock too, but stay quiet about it -- otherwise every reload
  // flashes "open a folder" for a moment before the real one appears.
  const locked = !folderName;
  const settled = folderReady;

  useEffect(() => {
    const el = transcriptRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines]);

  // Esc stops a reply, as in Claude Code -- from anywhere on the page.
  useEffect(() => {
    if (!replying) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !e.defaultPrevented) onStop();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [replying, onStop]);

  // Grows with the message, up to a cap, rather than a fixed one-line box.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [textValue]);

  function handleSend() {
    const value = textValue.trim();
    if (!value || locked) return;
    setTextValue("");
    // Mid-reply, a message waits its turn rather than being dropped -- or,
    // as when this button turned into Stop under the cursor, cutting the
    // reply short.
    if (replying) {
      setQueued(value);
      return;
    }
    onSendText(value);
    setLevelChanged(false);
    setFiles([]); // nothing to actually upload to yet -- clears with the message
  }

  // The queued message goes as soon as the reply in progress is over --
  // however it ended.
  useEffect(() => {
    if (replying || queued === null) return;
    onSendText(queued);
    // eslint-disable-next-line react-hooks/set-state-in-effect -- the queue empties as it sends
    setQueued(null);
    setLevelChanged(false);
  }, [replying, queued, onSendText]);

  function handleFilesSelected(e: React.ChangeEvent<HTMLInputElement>) {
    const picked = Array.from(e.target.files ?? []);
    if (picked.length) setFiles((prev) => [...prev, ...picked]);
    e.target.value = ""; // lets picking the same file again register as a change
  }

  function removeFile(index: number) {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  }

  return (
    <div className="flex h-screen flex-1 flex-col">
      {/* topbar */}
      <div className="flex items-center justify-end gap-3 border-b border-border px-6 py-3.5">
        {folderName && (
          <button
            type="button"
            onClick={onToggleFiles}
            aria-pressed={filesOpen}
            title={filesOpen ? "Hide the file panel" : "Browse this folder's files"}
            className={`rounded-full border px-2.5 py-1 text-[12.5px] font-medium ${
              filesOpen
                ? "border-text-muted text-text"
                : "border-border text-text-muted hover:border-text-muted hover:text-text"
            }`}
          >
            Files
          </button>
        )}
        {lines.some((l) => l.usage) && (
          <button
            type="button"
            onClick={onOpenTrace}
            title="See every step this chat took"
            className="rounded-full border border-border px-2.5 py-1 text-[12.5px] font-medium text-text-muted hover:border-text-muted hover:text-text"
          >
            Trace
          </button>
        )}
        {lineCountLabel && (
          <div className="flex items-center gap-1.5 rounded-full border border-border px-2.5 py-1 text-[12.5px] font-medium text-text-muted">
            <span className="font-mono text-[11px] text-text-faint">{lineCountLabel}</span>
          </div>
        )}
      </div>

      {/* transcript */}
      <div className="chat-scroll flex-1 overflow-y-auto" ref={transcriptRef}>
        <div className="mx-auto max-w-[720px] px-6">
          <div className="py-6 pb-3">
            {lines.length === 0 && settled && (
              <div className="pt-[20vh] text-center text-sm text-text-faint">
                {locked ? "Open a folder to start chatting." : "Ask something."}
              </div>
            )}
            {lines.map((line) => {
              const statsLine = formatUsageLine(line.usage, line.latency);
              const isYou = line.who === "you";
              return (
                <div key={line.id}>
                  <div className={`mb-4 flex ${isYou ? "justify-end" : line.who === "system" ? "justify-center mb-2.5" : ""}`}>
                    {line.who === "system" ? (
                      <span className="text-[12.5px] italic text-text-faint">{line.text}</span>
                    ) : line.who === "error" ? (
                      <span
                        role="alert"
                        className="max-w-full whitespace-pre-wrap break-words rounded-lg border border-danger/40 bg-danger-tint px-3.5 py-2.5 text-[13.5px] text-danger"
                      >
                        {line.text}
                      </span>
                    ) : line.who === "bot" && line.text === "" && replying ? (
                      <span className="grid w-full min-w-0 gap-1">
                      {line.liveThinking && <ThinkingRow text={line.liveThinking} live />}
                      <span role="status" aria-label="The assistant is working" className="flex items-center gap-2 py-2">
                        <span className="flex items-center gap-1.5">
                          {[0, 1, 2].map((i) => (
                            <span
                              key={i}
                              className="h-2 w-2 animate-pulse rounded-full bg-text-faint"
                              style={{ animationDelay: `${i * 0.2}s` }}
                            />
                          ))}
                        </span>
                        {activity && (
                          <span className="truncate font-mono text-[11.5px] text-text-faint">{activity}</span>
                        )}
                      </span>
                      </span>
                    ) : (
                      <span
                        className={`break-words text-[15px] leading-relaxed ${
                          isYou
                            ? "max-w-[80%] whitespace-pre-wrap rounded-[18px] bg-user-bubble px-4 py-2.5"
                            : // min-w-0 so a long command inside the trail clips
                              // rather than stretching the whole transcript.
                              "grid w-full min-w-0 gap-3"
                        }`}
                      >
                        {/* What it did on the way to this answer: its own
                            commentary, and each batch of tool calls. */}
                        {/* Until the turn is stored, its thinking has no
                            rounds to sit in -- show what has streamed so far. */}
                        {!isYou && !line.usage && line.liveThinking && (
                          <ThinkingRow text={line.liveThinking} live={replying} />
                        )}
                        {!isYou && (
                          <ActivityTrail
                            rounds={line.usage?.rounds}
                            // While streaming there is no stored trace yet, so
                            // the rows come from what has been announced so far.
                            toolCalls={line.usage?.tool_calls ?? line.liveCalls}
                            onOpenFile={onOpenFile}
                            root={folderPath}
                          />
                        )}
                        {/* Only the assistant writes markdown. Your own
                            message stays literal -- a line starting with
                            "#" or "-" is text you typed, not a heading.
                            `line.text` is every round's tokens concatenated,
                            so when the trail has already shown the earlier
                            rounds only the last one belongs here. */}
                        {isYou ? (
                          line.text
                        ) : line.usage?.finish_reason === "stopped" && !finalText(line).trim() ? (
                          <span className="text-[13.5px] italic text-text-faint">Stopped.</span>
                        ) : line.usage && !finalText(line).trim() ? (
                          // A finished turn with nothing to say. Seen with Qwen
                          // on LM Studio: a round that ends inside its thinking.
                          <span className="text-[13.5px] italic text-text-faint">
                            The model stopped without writing an answer
                            {line.usage.rounds?.some((r) => r.reasoning?.trim())
                              ? " — open “Thought” above to see what it was working on."
                              : "."}{" "}
                            Send “continue” to let it carry on.
                          </span>
                        ) : (
                          <Markdown>{finalText(line)}</Markdown>
                        )}
                      </span>
                    )}
                  </div>
                  {statsLine && (
                    // Per-turn tokens and cost. The full breakdown moved to
                    // the Trace button in the topbar; this stays because it
                    // is the number you want without leaving the chat.
                    <div
                      className={`-mt-2.5 mb-4 font-mono text-[11px] text-text-faint ${isYou ? "text-right" : ""}`}
                    >
                      {statsLine}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* composer */}
      <div className="shrink-0 border-t border-border px-6 pt-3.5 pb-5">
        {queued !== null && (
          <div className="mx-auto mb-2 flex max-w-[720px]">
            <span
              role="status"
              className="flex max-w-full items-center gap-1.5 rounded-full border border-border bg-surface-sunken px-3 py-1 text-[12px] text-text-muted"
            >
              <span className="shrink-0 text-text-faint">Queued:</span>
              <span className="min-w-0 truncate">{queued}</span>
              <button
                type="button"
                onClick={() => setQueued(null)}
                aria-label="Don't send the queued message"
                className="shrink-0 text-text-faint hover:text-danger"
              >
                ✕
              </button>
            </span>
          </div>
        )}
        {files.length > 0 && (
          <div className="mx-auto mb-2 flex max-w-[720px] flex-wrap gap-1.5">
            {files.map((f, i) => (
              <span
                key={`${f.name}-${i}`}
                className="flex items-center gap-1.5 rounded-full border border-border bg-surface-sunken px-3 py-1 text-[12px] text-text-muted"
              >
                {f.name}
                <button
                  type="button"
                  onClick={() => removeFile(i)}
                  aria-label={`Remove ${f.name}`}
                  className="text-text-faint hover:text-danger"
                >
                  ✕
                </button>
              </span>
            ))}
          </div>
        )}
        {/* which folder this chat runs in -- required before you can chat at all.
            Clicking this opens the real native OS folder dialog (backend-driven,
            via POST /fs/pick) -- not a browser picker, which can't give a real path. */}
        <div className="mx-auto mb-2 flex max-w-[720px] items-center gap-1.5">
          <button
            type="button"
            onClick={onOpenFolder}
            disabled={folderPicking}
            title={folderName ?? "Open a folder to start chatting"}
            className={`flex items-center gap-1.5 rounded-full border px-3 py-1 text-[12px] font-medium transition-colors disabled:cursor-wait ${
              locked && settled
                ? "border-accent-tint bg-accent-tint text-accent hover:opacity-85"
                : "border-border bg-surface-sunken text-text-muted hover:border-text-muted hover:text-text"
            }`}
          >
            <FolderIcon className="h-3.5 w-3.5" />
            {folderPicking ? "Waiting for folder…" : (folderName ?? (settled ? "Open folder" : "…"))}
          </button>
          {folderError && (
            <span role="alert" className="text-[12px] text-danger">
              {folderError}
            </span>
          )}
          {configError && (
            <span role="alert" className="text-[12px] text-danger">
              {configError}
            </span>
          )}
        </div>
        {/* one rounded box: message at top, add/model/level along the bottom
            -- mic/dictation go in that same bottom-right row later */}
        {showHelp && (
          <div className="relative mx-auto mb-2 max-w-[720px] rounded-xl border border-border bg-surface-sunken px-4 py-3 text-[12.5px] text-text-muted">
            <button
              type="button"
              onClick={() => setShowHelp(false)}
              aria-label="Close help"
              className="absolute top-2 right-3 text-text-faint hover:text-text"
            >
              ✕
            </button>
            <div className="mb-1.5 font-medium text-text">In the message box</div>
            <div>
              <span className="font-mono text-text">@</span> — mention a file or folder. The agent gets its path,
              and a small file&apos;s contents too.
            </div>
            <div>
              <span className="font-mono text-text">/</span> — a command at the start:{" "}
              {COMMANDS.map((c) => c.label).join(", ")}.
            </div>
            <div>
              <span className="font-mono text-text">Esc</span> stops a reply; typing while it runs queues your
              next message.
            </div>
          </div>
        )}
        <div className="relative mx-auto max-w-[720px] rounded-[26px] border border-border bg-surface-sunken px-4 pt-3.5 pb-2.5">
          {menu && (
            <ComposerMenu
              title={menu.kind === "file" ? "Files" : menu.kind === "model" ? "Model" : "Commands"}
              items={menuItems}
              active={Math.min(menuActive, Math.max(menuItems.length - 1, 0))}
              empty={menu.kind === "file" ? "No matching files." : "Nothing matches."}
              onPick={pick}
              onHover={setMenuActive}
            />
          )}
          <textarea
            ref={textareaRef}
            value={textValue}
            onChange={(e) => updateText(e.target.value, e.target.selectionStart)}
            onKeyDown={(e) => {
              if (menu) {
                const count = menuItems.length;
                if (e.key === "ArrowDown" && count) {
                  e.preventDefault();
                  setMenuActive((i) => (i + 1) % count);
                  return;
                }
                if (e.key === "ArrowUp" && count) {
                  e.preventDefault();
                  setMenuActive((i) => (i - 1 + count) % count);
                  return;
                }
                if ((e.key === "Enter" && !e.shiftKey) || e.key === "Tab") {
                  if (count) {
                    e.preventDefault();
                    pick(menuItems[Math.min(menuActive, count - 1)]);
                    return;
                  }
                }
                if (e.key === "Escape") {
                  // Marks it handled, so the Esc-to-stop listener leaves the reply alone.
                  e.preventDefault();
                  setMenu(null);
                  return;
                }
              }
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder={
              !settled
                ? ""
                : locked
                  ? "Open a folder to start chatting…"
                  : replying
                    ? "Replying… (type to queue a message, Esc to stop)"
                    : "Write a message…"
            }
            rows={1}
            disabled={locked}
            autoFocus
            className="block w-full resize-none bg-transparent text-[14.5px] leading-relaxed outline-none placeholder:text-text-faint disabled:cursor-not-allowed"
          />
          {levelChanged && (
            <div role="status" className="mt-2 text-[11.5px] text-text-faint">
              Changing the reasoning level resets the prompt cache — your next message reprocesses the
              whole conversation and costs more than usual.
            </div>
          )}
          <div className="mt-2 flex items-center justify-between">
            <input ref={fileInputRef} type="file" multiple hidden onChange={handleFilesSelected} />
            <PlusMenu
              onAddFile={() => fileInputRef.current?.click()}
              toolsEnabled={toolsEnabled}
              onToggleTools={() => setToolsEnabled((v) => !v)}
              disabled={locked}
            />
            <div className="flex items-center gap-1">
              {models.length > 1 ? (
                <Dropdown
                  value={model}
                  options={models}
                  onChange={onModelChange}
                  triggerClassName="text-text"
                  placement="top"
                />
              ) : (
                <span className="rounded-md px-1.5 py-1 text-[12.5px] font-medium text-text">{model}</span>
              )}
              <Dropdown value={level} options={levels} onChange={handleLevelChange} triggerClassName="text-text-muted" placement="top" />
              <ContextMeter used={contextUsed} window={contextWindow} costUsd={chatCostUsd} />
              {/* Stop only when there is nothing typed. With a message in
                  the box this stays Send (it queues), so clicking where
                  Send was a moment ago can't cut the reply short. */}
              {replying && !textValue.trim() ? (
                <button
                  type="button"
                  onClick={onStop}
                  disabled={stopping}
                  aria-label={stopping ? "Stopping" : "Stop (Esc)"}
                  title={stopping ? "Stopping…" : "Stop (Esc)"}
                  className="ml-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-text text-bg transition-opacity hover:opacity-85 disabled:cursor-wait disabled:opacity-40"
                >
                  <span className="block h-2.5 w-2.5 rounded-[2px] bg-bg" />
                </button>
              ) : (
                <button
                  type="button"
                  onClick={handleSend}
                  disabled={locked}
                  aria-label="Send"
                  className="ml-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-text text-bg transition-opacity hover:opacity-85 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  ↑
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

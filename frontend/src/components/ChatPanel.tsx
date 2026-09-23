"use client";

import { useEffect, useRef, useState } from "react";
import type { TranscriptLine } from "@/lib/types";
import { formatUsageLine } from "@/lib/format";
import { Dropdown } from "./Dropdown";
import { PlusMenu } from "./PlusMenu";
import { FolderIcon } from "./icons";
import { ContextMeter } from "./ContextMeter";
import { Markdown } from "./Markdown";

interface ChatPanelProps {
  lines: TranscriptLine[];
  lineCountLabel: string;
  onSendText: (text: string) => void;
  replying: boolean; // a reply is still streaming in
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
  folderReady: boolean; // false until the saved folder has been restored
  onOpenFolder: () => void;
  folderPicking: boolean; // the native OS dialog is open, waiting on you
  folderError: string | null;
  configError: string | null; // provider not set up; chatting will fail
  onOpenTrace: () => void;
}

export function ChatPanel({
  lines,
  lineCountLabel,
  onSendText,
  replying,
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
  folderReady,
  onOpenFolder,
  folderPicking,
  folderError,
  configError,
  onOpenTrace,
}: ChatPanelProps) {
  const transcriptRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [textValue, setTextValue] = useState("");
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

  // Grows with the message, up to a cap, rather than a fixed one-line box.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [textValue]);

  function handleSend() {
    const value = textValue.trim();
    if (!value || replying || locked) return;
    onSendText(value);
    setTextValue("");
    setLevelChanged(false);
    setFiles([]); // nothing to actually upload to yet -- clears with the message
  }

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
                    ) : (
                      <span
                        className={`break-words text-[15px] leading-relaxed ${
                          isYou
                            ? "max-w-[80%] whitespace-pre-wrap rounded-[18px] bg-user-bubble px-4 py-2.5"
                            : "max-w-full"
                        }`}
                      >
                        {/* Only the assistant writes markdown. Your own
                            message stays literal -- a line starting with
                            "#" or "-" is text you typed, not a heading. */}
                        {isYou ? line.text : <Markdown>{line.text}</Markdown>}
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
        <div className="mx-auto max-w-[720px] rounded-[26px] border border-border bg-surface-sunken px-4 pt-3.5 pb-2.5">
          <textarea
            ref={textareaRef}
            value={textValue}
            onChange={(e) => setTextValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder={
              !settled ? "" : locked ? "Open a folder to start chatting…" : replying ? "Replying…" : "Write a message…"
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
              <button
                type="button"
                onClick={handleSend}
                disabled={replying || locked}
                aria-label="Send"
                className="ml-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-text text-bg transition-opacity hover:opacity-85 disabled:cursor-not-allowed disabled:opacity-40"
              >
                ↑
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

"use client";

import { useEffect, useRef, useState } from "react";
import { formatUsageLine, truncateToolResult } from "@/lib/format";
import type { ConnState, SessionMode, TranscriptLine } from "@/lib/types";
import { StateIcon } from "./icons";

interface ChatPanelProps {
  mode: SessionMode;
  onModeChange: (mode: SessionMode) => void;
  connected: boolean;
  connecting: boolean;
  state: ConnState;
  stateText: string;
  statusLabel: string;
  lines: TranscriptLine[];
  lineCountLabel: string;
  micMuted: boolean;
  onToggleMute: () => void;
  onConnect: () => void;
  onDisconnect: () => void;
  onOpenSettings: () => void;
  onOpenTools: () => void;
  modelChipLabel: string;
  onSendText: (text: string) => void;
}

export function ChatPanel({
  mode,
  onModeChange,
  connected,
  connecting,
  state,
  stateText,
  statusLabel,
  lines,
  lineCountLabel,
  micMuted,
  onToggleMute,
  onConnect,
  onDisconnect,
  onOpenSettings,
  onOpenTools,
  modelChipLabel,
  onSendText,
}: ChatPanelProps) {
  const transcriptRef = useRef<HTMLDivElement>(null);
  const [textValue, setTextValue] = useState("");
  const textMode = mode === "text";

  useEffect(() => {
    const el = transcriptRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines]);

  const led =
    state === "listening" || state === "speaking"
      ? "bg-accent shadow-[0_0_0_3px_var(--accent-tint)]"
      : state === "error"
        ? "bg-danger shadow-[0_0_0_3px_var(--danger-tint)]"
        : "bg-text-faint";

  const tileClass =
    state === "listening" || state === "speaking"
      ? "bg-accent-tint text-accent"
      : state === "error"
        ? "bg-danger-tint text-danger"
        : state === "muted"
          ? "bg-surface-sunken text-text-faint"
          : "bg-surface-sunken text-text-faint";

  function handleSend() {
    const value = textValue.trim();
    if (!value) return;
    onSendText(value);
    setTextValue("");
  }

  return (
    <div className="flex h-screen flex-1 flex-col">
      {/* topbar */}
      <div className="flex items-center justify-between gap-3 border-b border-border px-6 py-3.5">
        <div className="flex gap-0.5 rounded-lg bg-surface-sunken p-[3px]">
          <button
            type="button"
            disabled={connecting}
            onClick={() => onModeChange("voice")}
            className={`rounded-md px-3.5 py-1.5 text-[12.5px] font-semibold transition-colors ${
              mode === "voice" ? "bg-bg text-text shadow-[var(--shadow)]" : "text-text-muted hover:text-text"
            } disabled:cursor-not-allowed disabled:opacity-60`}
          >
            Voice
          </button>
          <button
            type="button"
            disabled={connecting}
            onClick={() => onModeChange("text")}
            className={`rounded-md px-3.5 py-1.5 text-[12.5px] font-semibold transition-colors ${
              mode === "text" ? "bg-bg text-text shadow-[var(--shadow)]" : "text-text-muted hover:text-text"
            } disabled:cursor-not-allowed disabled:opacity-60`}
          >
            Text
          </button>
        </div>
        <div className="flex items-center gap-1.5 rounded-full border border-border px-2.5 py-1 text-[12.5px] font-medium text-text-muted">
          <span className={`h-[7px] w-[7px] shrink-0 rounded-full transition-colors ${led}`} />
          <span>{stateText}</span>
          {lineCountLabel && (
            <span className="font-mono text-[11px] text-text-faint before:content-['·_']">{lineCountLabel}</span>
          )}
        </div>
      </div>

      {/* transcript */}
      <div className="chat-scroll flex-1 overflow-y-auto" ref={transcriptRef}>
        <div className="mx-auto max-w-[720px] px-6">
          <div className="py-6 pb-3">
            {lines.length === 0 && (
              <div className="pt-[20vh] text-center text-sm text-text-faint">
                Ask something, or say it out loud once you connect.
              </div>
            )}
            {lines.map((line) => {
              const statsLine = formatUsageLine(line.usage, line.latency);
              const toolCalls = line.usage?.tool_calls;
              const isYou = line.who === "you";
              return (
                <div key={line.id}>
                  <div className={`mb-4 flex ${isYou ? "justify-end" : line.who === "system" ? "justify-center mb-2.5" : ""}`}>
                    {line.who === "system" ? (
                      <span className="text-[12.5px] italic text-text-faint">{line.text}</span>
                    ) : (
                      <span
                        className={`whitespace-pre-wrap break-words text-[15px] leading-relaxed ${
                          isYou
                            ? "max-w-[80%] rounded-[18px] bg-user-bubble px-4 py-2.5"
                            : "max-w-full"
                        }`}
                      >
                        {line.text}
                      </span>
                    )}
                  </div>
                  {statsLine && (
                    <div className={`-mt-2.5 mb-4 font-mono text-[11px] text-text-faint ${isYou ? "text-right" : ""}`}>
                      {statsLine}
                    </div>
                  )}
                  {toolCalls && toolCalls.length > 0 && (
                    <details className={`-mt-2.5 mb-4 font-mono text-[11px] text-text-faint ${isYou ? "text-right" : ""}`}>
                      <summary className="cursor-pointer list-none hover:text-text-muted">
                        tools: {toolCalls.map((c) => c.name).join(", ")}
                      </summary>
                      {toolCalls.map((call, i) => (
                        <div key={i} className="grid grid-cols-[auto_auto_1fr] items-baseline gap-2 pt-1.5 pl-3">
                          <span className="font-semibold text-text-muted">{call.name}</span>
                          <span className="text-text-faint">{JSON.stringify(call.args ?? {})}</span>
                          <span className="whitespace-pre-wrap break-words text-text-muted">
                            {truncateToolResult(call.result)}
                          </span>
                        </div>
                      ))}
                    </details>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* composer */}
      <div className="shrink-0 border-t border-border px-6 pt-3.5 pb-5">
        <div className="mx-auto mb-2 flex max-w-[720px] justify-center gap-2">
          <button
            type="button"
            onClick={onOpenSettings}
            className="rounded-full border border-border bg-surface-sunken px-3.5 py-1 text-xs font-medium text-text-muted transition-colors hover:border-text-muted hover:text-text"
          >
            {modelChipLabel}
          </button>
          <button
            type="button"
            onClick={onOpenTools}
            className="rounded-full border border-border bg-surface-sunken px-3.5 py-1 text-xs font-medium text-text-muted transition-colors hover:border-text-muted hover:text-text"
          >
            Tools
          </button>
        </div>
        <div className="mx-auto flex max-w-[720px] items-center gap-3">
          <div className={`flex h-[38px] w-[38px] shrink-0 items-center justify-center rounded-full transition-colors ${tileClass}`}>
            <StateIcon state={state} className={`h-[18px] w-[18px] ${state === "speaking" ? "tile-pulse animate-[tile-pulse_1s_ease-in-out_infinite]" : ""}`} />
          </div>
          <div className="min-w-0 flex-1">
            <div className="text-[13.5px] font-semibold">{stateText}</div>
            <div className="mt-px overflow-hidden text-ellipsis whitespace-nowrap text-xs text-text-muted">
              {statusLabel}
            </div>
          </div>
          <div className="flex shrink-0 gap-2">
            {!connected ? (
              <button
                type="button"
                onClick={onConnect}
                disabled={connecting}
                className="rounded-full bg-text px-[18px] py-2.5 text-[13.5px] font-semibold text-bg transition-opacity hover:opacity-85 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-40"
              >
                Connect
              </button>
            ) : (
              <button
                type="button"
                onClick={onDisconnect}
                className="rounded-full border border-border bg-transparent px-[18px] py-2.5 text-[13.5px] font-semibold text-text-muted transition-colors hover:border-danger hover:text-danger"
              >
                Disconnect
              </button>
            )}
            <button
              type="button"
              onClick={onToggleMute}
              disabled={!connected || textMode}
              className={`rounded-full border px-4 py-2.5 text-[13.5px] font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-45 ${
                micMuted
                  ? "border-danger bg-danger-tint text-danger"
                  : "border-border bg-transparent text-text hover:border-text-muted"
              }`}
            >
              {micMuted ? "Unmute" : "Mute"}
            </button>
          </div>
        </div>
        {textMode && connected && (
          <div className="mx-auto mt-3 flex max-w-[720px] items-center gap-2 rounded-[26px] bg-surface-sunken py-1.5 pr-1.5 pl-4.5">
            <input
              type="text"
              value={textValue}
              onChange={(e) => setTextValue(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
              placeholder="Message the assistant…"
              autoFocus
              className="flex-1 bg-transparent py-2 text-[14.5px] outline-none placeholder:text-text-faint"
            />
            <button
              type="button"
              onClick={handleSend}
              className="shrink-0 rounded-full bg-text px-[18px] py-2.5 text-[13.5px] font-semibold text-bg transition-opacity hover:opacity-85"
            >
              Send
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

"use client";

import { useState } from "react";
import { fetchSystemPrompt } from "@/lib/tools";

/**
 * Read-only view of the agent's system prompt. It's fetched on first open
 * rather than with the page, since most visits to Settings aren't for this.
 * Editing it means editing `backend/src/pos/prompt.py` -- there's no write
 * path here on purpose.
 */
export function SystemPromptView() {
  const [open, setOpen] = useState(false);
  const [text, setText] = useState<string | null>(null);
  const [error, setError] = useState(false);
  const [copied, setCopied] = useState(false);

  async function toggle() {
    const next = !open;
    setOpen(next);
    if (next && text === null && !error) {
      try {
        setText(await fetchSystemPrompt());
      } catch {
        setError(true);
      }
    }
  }

  async function copy() {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard blocked -- the text is on screen and selectable anyway
    }
  }

  return (
    <div className="grid gap-2.5">
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={toggle}
          aria-expanded={open}
          className="rounded-lg border border-border px-2.5 py-1.5 text-[12.5px] font-medium text-text hover:bg-surface-sunken"
        >
          {open ? "Hide prompt" : "View prompt"}
        </button>
        {open && text && (
          <button
            type="button"
            onClick={copy}
            className="rounded-lg px-2.5 py-1.5 text-[12.5px] font-medium text-text-muted hover:bg-surface-sunken hover:text-text"
          >
            {copied ? "Copied" : "Copy"}
          </button>
        )}
      </div>

      {open && (
        <>
          {error ? (
            <div className="text-[12.5px] text-text-faint">Couldn&apos;t reach the backend.</div>
          ) : text === null ? (
            <div className="text-[12.5px] text-text-faint">Loading…</div>
          ) : (
            <pre className="m-0 max-h-[340px] overflow-auto whitespace-pre-wrap break-words rounded-lg border border-border bg-surface-sunken p-3 font-mono text-[11.5px] leading-[1.6] text-text-muted">
              {text}
            </pre>
          )}
          <p className="m-0 text-[11.5px] text-text-faint">
            The file and skill tools add their own instructions on top of this at request time.
          </p>
        </>
      )}
    </div>
  );
}

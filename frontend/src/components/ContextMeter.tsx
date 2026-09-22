"use client";

import { useRef, useState } from "react";
import { useClickOutside } from "@/hooks/useClickOutside";
import { formatTokens } from "@/lib/format";

interface ContextMeterProps {
  used?: number;
  window?: number;
  costUsd?: number;
}

/** A ring that fills as the conversation grows; click it for the numbers. */
export function ContextMeter({ used, window: limit, costUsd }: ContextMeterProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useClickOutside(rootRef, () => setOpen(false), open);

  const ratio = used != null && limit ? Math.min(1, used / limit) : 0;
  const full = ratio > 0.8;

  // A stroked circle, drawn from the top: the dash covers `ratio` of the
  // circumference, the gap covers the rest.
  const radius = 6;
  const circumference = 2 * Math.PI * radius;

  return (
    <div ref={rootRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-label="Context window usage"
        aria-expanded={open}
        title={used != null && limit ? `${used.toLocaleString()} / ${limit.toLocaleString()} tokens` : "Context window"}
        className="flex h-7 w-7 items-center justify-center rounded-full text-text-muted transition-colors hover:bg-border hover:text-text"
      >
        <svg viewBox="0 0 16 16" className="h-[15px] w-[15px] -rotate-90">
          <circle cx="8" cy="8" r={radius} fill="none" stroke="currentColor" strokeWidth="2" opacity="0.25" />
          {ratio > 0 && (
            <circle
              cx="8"
              cy="8"
              r={radius}
              fill="none"
              stroke={full ? "var(--danger)" : "var(--accent)"}
              strokeWidth="2"
              strokeLinecap="round"
              strokeDasharray={`${ratio * circumference} ${circumference}`}
            />
          )}
        </svg>
      </button>

      {open && (
        <div className="absolute right-0 bottom-full z-10 mb-2 w-[272px] rounded-xl border border-border bg-bg p-3.5 shadow-[var(--shadow)]">
          <div className="flex items-baseline justify-between gap-3">
            <span className="text-[13px] text-text-muted">Context window</span>
            <span className="font-mono text-[12px] text-text">
              {used != null ? formatTokens(used) : "0"}
              {limit ? ` / ${formatTokens(limit)}` : ""}
            </span>
          </div>
          <div className="mt-2 h-[3px] w-full overflow-hidden rounded-full bg-border">
            <div
              className={`h-full rounded-full ${full ? "bg-danger" : "bg-accent"}`}
              style={{ width: `${ratio * 100}%` }}
            />
          </div>
          <div className="mt-1.5 text-[11.5px] text-text-faint">
            {used != null && limit ? `${(ratio * 100).toFixed(1)}% used` : "Send a message to measure"}
          </div>

          {costUsd != null && (
            <>
              <div className="my-3 border-t border-border" />
              <div className="flex items-baseline justify-between gap-3">
                <span className="text-[13px] text-text-muted">This chat</span>
                <span className="font-mono text-[12px] text-text">${costUsd.toFixed(4)}</span>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

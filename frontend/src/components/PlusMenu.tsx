"use client";

import { useRef, useState } from "react";
import { useClickOutside } from "@/hooks/useClickOutside";

interface PlusMenuProps {
  onAddFile: () => void;
  toolsEnabled: boolean;
  onToggleTools: () => void;
  disabled?: boolean;
}

/** The composer's "+" button: a menu of things to add to the message, not a
 * direct action -- room for more than just files (tools now, more later). */
export function PlusMenu({ onAddFile, toolsEnabled, onToggleTools, disabled }: PlusMenuProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useClickOutside(rootRef, () => setOpen(false), open);

  return (
    <div ref={rootRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        disabled={disabled}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label="Add to message"
        className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-lg leading-none text-text-muted transition-colors hover:bg-border hover:text-text disabled:cursor-not-allowed disabled:opacity-40"
      >
        +
      </button>
      {open && (
        <div
          role="menu"
          className="absolute bottom-full left-0 z-10 mb-1.5 min-w-[170px] overflow-hidden rounded-lg border border-border bg-bg py-1 shadow-[var(--shadow)]"
        >
          <button
            type="button"
            role="menuitem"
            onClick={() => {
              onAddFile();
              setOpen(false);
            }}
            className="block w-full px-3 py-1.5 text-left text-[13px] text-text-muted transition-colors hover:bg-surface-sunken hover:text-text"
          >
            Add file
          </button>
          <button
            type="button"
            role="menuitemcheckbox"
            aria-checked={toolsEnabled}
            onClick={onToggleTools}
            className="flex w-full items-center justify-between px-3 py-1.5 text-left text-[13px] text-text-muted transition-colors hover:bg-surface-sunken hover:text-text"
          >
            Tools
            <span className={`text-[11px] ${toolsEnabled ? "text-accent" : "text-text-faint"}`}>
              {toolsEnabled ? "On" : "Off"}
            </span>
          </button>
        </div>
      )}
    </div>
  );
}

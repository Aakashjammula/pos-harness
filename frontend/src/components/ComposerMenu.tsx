"use client";

import { useEffect, useRef } from "react";

export interface MenuItem {
  key: string;
  label: string;
  /** Shown faint beside the label: a command's purpose, a file's folder. */
  detail?: string;
  icon?: string;
}

interface ComposerMenuProps {
  title: string;
  items: MenuItem[];
  active: number;
  /** Shown when there is nothing to list. */
  empty: string;
  onPick: (item: MenuItem) => void;
  onHover: (index: number) => void;
}

/**
 * The popup above the composer for "@" (files) and "/" (commands).
 *
 * Purely presentational: ChatPanel owns which items are listed and which is
 * highlighted, since the keys that move through them arrive at the
 * textarea, which keeps focus the whole time.
 */
export function ComposerMenu({ title, items, active, empty, onPick, onHover }: ComposerMenuProps) {
  const listRef = useRef<HTMLDivElement>(null);

  // Keep the highlighted row in view as the arrow keys move it.
  useEffect(() => {
    listRef.current?.querySelector<HTMLElement>(`[data-index="${active}"]`)?.scrollIntoView({ block: "nearest" });
  }, [active]);

  return (
    <div
      className="absolute inset-x-0 bottom-full mb-2 overflow-hidden rounded-xl border border-border bg-bg shadow-[var(--shadow)]"
      role="listbox"
      aria-label={title}
    >
      <div className="border-b border-border px-3 py-1.5 text-[11px] font-semibold tracking-wide text-text-faint uppercase">
        {title}
      </div>
      <div ref={listRef} className="chat-scroll max-h-64 overflow-y-auto py-1">
        {items.length === 0 ? (
          <div className="px-3 py-2 text-[12.5px] text-text-faint">{empty}</div>
        ) : (
          items.map((item, index) => (
            <button
              key={item.key}
              type="button"
              role="option"
              aria-selected={index === active}
              data-index={index}
              // mousedown, not click: a click would blur the textarea first.
              onMouseDown={(e) => {
                e.preventDefault();
                onPick(item);
              }}
              onMouseEnter={() => onHover(index)}
              className={`flex w-full min-w-0 items-baseline gap-2 px-3 py-1.5 text-left text-[13px] ${
                index === active ? "bg-surface-sunken text-text" : "text-text-muted"
              }`}
            >
              {item.icon && <span className="w-4 shrink-0 text-center text-text-faint">{item.icon}</span>}
              <span className="shrink-0 font-mono text-[12.5px]">{item.label}</span>
              {item.detail && <span className="min-w-0 truncate text-[12px] text-text-faint">{item.detail}</span>}
            </button>
          ))
        )}
      </div>
    </div>
  );
}

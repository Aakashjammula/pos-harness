"use client";

import { useRef, useState } from "react";
import { useClickOutside } from "@/hooks/useClickOutside";

interface DropdownProps {
  value: string;
  options: string[];
  onChange: (value: string) => void;
  triggerClassName?: string; // text color/weight for the trigger button
  placement?: "top" | "bottom"; // which side of the trigger the menu opens on (default: bottom)
}

/**
 * A themed dropdown standing in for a native <select>. A native select's
 * closed state can be styled, but its open menu is drawn by the OS/browser
 * itself -- no CSS reaches it -- so against a dark app it renders with the
 * platform's own light chrome and flickers between that and the hover
 * state. This draws the whole menu itself instead.
 */
export function Dropdown({ value, options, onChange, triggerClassName, placement = "bottom" }: DropdownProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useClickOutside(rootRef, () => setOpen(false), open);

  return (
    <div ref={rootRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="listbox"
        aria-expanded={open}
        className={`rounded-md px-1.5 py-1 text-[12.5px] font-medium outline-none transition-colors hover:bg-border ${triggerClassName ?? "text-text"}`}
      >
        {value}
      </button>
      {open && (
        <div
          role="listbox"
          className={`absolute right-0 z-10 min-w-[150px] overflow-hidden rounded-lg border border-border bg-bg py-1 shadow-[var(--shadow)] ${
            placement === "top" ? "bottom-full mb-1.5" : "top-full mt-1.5"
          }`}
        >
          {options.map((opt) => (
            <button
              key={opt}
              type="button"
              role="option"
              aria-selected={opt === value}
              onClick={() => {
                onChange(opt);
                setOpen(false);
              }}
              className={`block w-full px-3 py-1.5 text-left text-[13px] transition-colors hover:bg-surface-sunken ${
                opt === value ? "font-medium text-text" : "text-text-muted"
              }`}
            >
              {opt}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

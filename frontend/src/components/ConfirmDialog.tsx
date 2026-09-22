"use client";

import { useEffect, useRef } from "react";
import { useClickOutside } from "@/hooks/useClickOutside";

interface ConfirmDialogProps {
  title: string;
  body?: string;
  confirmLabel?: string;
  /** Styles the confirm button as destructive. */
  danger?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

/**
 * A modal confirm, in place of the browser's own `confirm()` -- which can't
 * be styled and, in some browsers, offers to suppress itself. Rendered only
 * while a confirmation is pending, so mounting it is what opens it.
 */
export function ConfirmDialog({
  title,
  body,
  confirmLabel = "Confirm",
  danger = false,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  const cardRef = useRef<HTMLDivElement>(null);
  const cancelRef = useRef<HTMLButtonElement>(null);

  // Escape and outside clicks both cancel -- the same behaviour as every
  // other popover in the app.
  useClickOutside(cardRef, onCancel, true);

  // Focus starts on Cancel, not Confirm: a stray Enter shouldn't delete
  // anything.
  useEffect(() => {
    cancelRef.current?.focus();
  }, []);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
      <div
        ref={cardRef}
        role="alertdialog"
        aria-modal="true"
        aria-label={title}
        className="w-full max-w-[380px] rounded-xl border border-border bg-bg p-4 shadow-[var(--shadow)]"
      >
        <h2 className="m-0 text-[14.5px] font-semibold text-text">{title}</h2>
        {body && <p className="mt-1.5 mb-0 text-[12.5px] leading-[1.5] text-text-muted">{body}</p>}
        <div className="mt-4 flex justify-end gap-2">
          <button
            ref={cancelRef}
            type="button"
            onClick={onCancel}
            className="rounded-lg border border-border px-3 py-1.5 text-[12.5px] font-medium text-text-muted hover:bg-surface-sunken hover:text-text"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onConfirm}
            className={`rounded-lg px-3 py-1.5 text-[12.5px] font-medium text-white ${
              danger ? "bg-danger hover:opacity-90" : "bg-accent hover:opacity-90"
            }`}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}

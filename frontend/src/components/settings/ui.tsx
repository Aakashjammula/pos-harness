"use client";

import type { ReactNode } from "react";

export const inputClass =
  "w-full rounded-lg border border-transparent bg-surface-sunken px-2.5 py-2 text-[13px] text-text outline-none transition-[border-color,box-shadow] focus:border-accent focus:shadow-[0_0_0_3px_var(--accent-tint)] disabled:cursor-not-allowed disabled:opacity-50";

/** A titled block with its own actions, so saving one section never touches another. */
export function Section({
  title,
  description,
  danger,
  children,
}: {
  title: string;
  description?: string;
  danger?: boolean;
  children: ReactNode;
}) {
  return (
    <section
      className={`grid gap-3 rounded-xl border p-4 ${danger ? "border-danger/40" : "border-border"}`}
      aria-label={title}
    >
      <div className="grid gap-0.5">
        <h3 className={`m-0 text-[14px] font-semibold ${danger ? "text-danger" : "text-text"}`}>{title}</h3>
        {description && <p className="m-0 text-[12.5px] text-text-muted">{description}</p>}
      </div>
      {children}
    </section>
  );
}

export function TextField({
  label,
  id,
  hint,
  ...input
}: { label: string; id: string; hint?: string } & React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <div className="grid gap-1.5">
      <label htmlFor={id} className="text-[12.5px] text-text-muted">
        {label}
      </label>
      <input id={id} className={inputClass} {...input} />
      {hint && <span className="text-[11.5px] text-text-faint">{hint}</span>}
    </div>
  );
}

export function PrimaryButton({ children, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      type="button"
      className="rounded-lg bg-accent px-3 py-1.5 text-[12.5px] font-medium text-white hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-50"
      {...props}
    >
      {children}
    </button>
  );
}

export function QuietButton({ children, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      type="button"
      className="rounded-lg px-3 py-1.5 text-[12.5px] font-medium text-text-muted hover:bg-surface-sunken hover:text-text disabled:cursor-not-allowed disabled:opacity-50"
      {...props}
    >
      {children}
    </button>
  );
}

export function DangerButton({ children, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      type="button"
      className="rounded-lg bg-danger px-3 py-1.5 text-[12.5px] font-medium text-white hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
      {...props}
    >
      {children}
    </button>
  );
}

/** Feedback under a section: an error is an alert, success is a quiet status line. */
export function Notice({ kind, children }: { kind: "error" | "ok"; children: ReactNode }) {
  return kind === "error" ? (
    <p role="alert" className="m-0 text-xs text-red-500">
      {children}
    </p>
  ) : (
    <p role="status" className="m-0 text-xs text-text-muted">
      {children}
    </p>
  );
}

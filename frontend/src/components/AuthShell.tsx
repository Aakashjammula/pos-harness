"use client";

import { BrandMark } from "./icons";

// Auth pages are forced dark regardless of the viewer's system theme --
// a deliberate "boot screen" moment for the harness, distinct from the
// rest of the app which follows prefers-color-scheme. Values mirror
// globals.css's own dark block so the two never drift apart silently.
const authDarkVars = {
  "--bg": "#212121",
  "--sidebar-bg": "#171717",
  "--surface-sunken": "#2f2f2f",
  "--border": "#383838",
  "--text": "#ececec",
  "--text-muted": "#9b9b9b",
  "--text-faint": "#6e6e6e",
  "--accent": "#19c37d",
  "--accent-hover": "#29d68e",
  "--accent-tint": "#123529",
  "--danger": "#f0665f",
  "--danger-tint": "#3a1c1c",
  "--user-bubble": "#2f2f2f",
  "--shadow": "0 4px 16px -6px rgba(0, 0, 0, 0.5)",
  // Everything inside the shell is sized in em against this one value, so
  // the whole form (type, inputs, spacing, column width) scales with the
  // window instead of sitting in a fixed-pixel card.
  fontSize: "clamp(15px, 0.55vw + 11px, 20px)",
} as React.CSSProperties;

export function BrandWordmark() {
  return (
    <div className="flex items-center gap-[0.65em]">
      <BrandMark className="h-[1.6em] w-auto shrink-0" />
      <span className="font-mono text-[1.1em] font-semibold tracking-tight text-text">pos</span>
    </div>
  );
}

const STAGES = [
  { name: "Voice detection", detail: "Hears when you start and stop talking." },
  { name: "Speech to text", detail: "Turns your voice into words, on your CPU." },
  { name: "Language model", detail: "Local or cloud. Your pick, per conversation." },
  { name: "Text to speech", detail: "Speaks the reply, and stops if you cut in." },
];

// 32 bars with a deterministic spread of heights and delays, so the
// waveform looks organic without any randomness (SSR-safe, no hydration
// mismatch).
const BARS = Array.from({ length: 32 }, (_, i) => ({
  h: 22 + Math.round(58 * Math.abs(Math.sin(i * 0.61) * Math.cos(i * 0.23))),
  delay: -((i * 137) % 1400),
}));

function Waveform() {
  return (
    <div aria-hidden className="flex h-[7em] items-center gap-[0.32em]">
      {BARS.map((b, i) => (
        <span
          key={i}
          className="auth-bar w-[0.32em] flex-1 rounded-full bg-accent"
          style={{ height: `${b.h}%`, animationDelay: `${b.delay}ms` }}
        />
      ))}
    </div>
  );
}

function BrandPanel() {
  return (
    <div className="flex h-full flex-col justify-between gap-[3em] p-[3em]">
      <BrandWordmark />

      <div className="grid gap-[2.2em]">
        <div className="grid gap-[0.7em]">
          <h2 className="max-w-[14em] font-mono text-[2.1em] font-semibold leading-[1.15] tracking-tight text-text text-balance">
            Talk to it.
            <br />
            It talks back.
          </h2>
          <p className="max-w-[28em] text-[0.95em] leading-relaxed text-text-muted">
            A voice agent that runs on your machine, with the model of your choice, and keeps one
            thread going whether you speak or type.
          </p>
        </div>

        <Waveform />

        <ol className="grid gap-[0.9em]">
          {STAGES.map((s) => (
            <li key={s.name} className="grid grid-cols-[0.5em_1fr] items-baseline gap-x-[0.9em]">
              <span aria-hidden className="h-[0.5em] w-[0.5em] translate-y-[-0.05em] rounded-full bg-accent" />
              <span className="text-[0.95em] text-text">
                {s.name}
                <span className="text-text-muted"> — {s.detail}</span>
              </span>
            </li>
          ))}
        </ol>
      </div>

      <p className="text-[0.8em] text-text-faint">Nothing leaves your machine unless you pick a cloud model.</p>
    </div>
  );
}

// Shared frame for login, signup and magic-link. Desktop: brand panel left,
// form right, both filling the viewport. Below `lg` the panel drops away and
// the form takes the full width under a compact wordmark.
export function AuthShell({ children }: { children: React.ReactNode }) {
  return (
    <main style={authDarkVars} className="flex min-h-screen bg-bg text-text">
      <div className="hidden w-[46%] max-w-[44em] shrink-0 border-r border-border bg-sidebar-bg lg:block">
        <BrandPanel />
      </div>

      <div className="flex flex-1 flex-col items-center justify-center gap-[2.5em] px-[1.5em] py-[3em]">
        <div className="lg:hidden">
          <BrandWordmark />
        </div>
        <div className="w-full max-w-[26em]">{children}</div>
      </div>
    </main>
  );
}

const inputClass =
  "w-full rounded-[0.6em] border border-border bg-surface-sunken px-[0.9em] py-[0.8em] text-[1em] text-text outline-none transition-[border-color,box-shadow] placeholder:text-text-faint focus:border-accent focus:shadow-[0_0_0_0.2em_var(--accent-tint)]";

export function AuthHeading({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div className="mb-[1.8em] grid gap-[0.5em]">
      <h1 className="font-mono text-[2em] font-semibold leading-tight tracking-tight text-text text-balance">
        {title}
      </h1>
      <p className="text-[0.95em] leading-relaxed text-text-muted">{subtitle}</p>
    </div>
  );
}

export function Field({
  id,
  label,
  ...props
}: { id: string; label: string } & React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <div className="grid gap-[0.45em]">
      <label htmlFor={id} className="text-[0.9em] font-medium text-text-muted">
        {label}
      </label>
      <input id={id} className={inputClass} {...props} />
    </div>
  );
}

export function PasswordField({
  id,
  label,
  shown,
  onToggle,
  ...props
}: {
  id: string;
  label: string;
  shown: boolean;
  onToggle: () => void;
} & Omit<React.InputHTMLAttributes<HTMLInputElement>, "type">) {
  return (
    <div className="grid gap-[0.45em]">
      <label htmlFor={id} className="text-[0.9em] font-medium text-text-muted">
        {label}
      </label>
      <div className="relative">
        <input id={id} type={shown ? "text" : "password"} className={`${inputClass} pr-[4.5em]`} {...props} />
        <button
          type="button"
          onClick={onToggle}
          aria-pressed={shown}
          className="absolute inset-y-0 right-[0.4em] my-[0.4em] rounded-[0.4em] px-[0.7em] text-[0.85em] font-medium text-text-muted hover:text-text focus-visible:outline-2 focus-visible:outline-accent"
        >
          {shown ? "Hide" : "Show"}
        </button>
      </div>
    </div>
  );
}

export function ErrorNote({ children }: { children: React.ReactNode }) {
  return (
    <div role="alert" className="rounded-[0.6em] bg-danger-tint px-[0.9em] py-[0.7em] text-[0.9em] leading-relaxed text-danger">
      {children}
    </div>
  );
}

export function PrimaryButton({
  children,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...props}
      className="w-full rounded-[0.6em] bg-accent px-[1em] py-[0.85em] text-[1em] font-semibold text-white transition-colors hover:bg-accent-hover focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:opacity-60"
    >
      {children}
    </button>
  );
}

export function SecondaryButton({
  children,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...props}
      className="w-full rounded-[0.6em] bg-surface-sunken px-[1em] py-[0.85em] text-[1em] font-medium text-text transition-colors hover:bg-border focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
    >
      {children}
    </button>
  );
}

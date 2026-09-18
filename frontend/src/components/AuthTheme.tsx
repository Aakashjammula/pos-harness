"use client";

import { BrandMark } from "./icons";

// Auth pages are forced dark regardless of the viewer's system theme --
// a deliberate "boot screen" moment for the harness, distinct from the
// rest of the app which follows prefers-color-scheme. Values mirror
// globals.css's own dark block so the two never drift apart silently.
export const authDarkVars = {
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
} as React.CSSProperties;

export function BrandWordmark() {
  return (
    <div className="flex items-center gap-2.5">
      <BrandMark className="h-6 w-auto shrink-0" />
      <span className="font-mono text-[16px] font-semibold tracking-tight text-text">pos</span>
    </div>
  );
}

// Textured backdrop for auth pages -- a faint dot grid (the "harness" is a
// technical tool, not a consumer app) plus one soft glow in the logo's own
// blue, so the mark isn't the only place that color appears. Decorative
// only: aria-hidden, and it never intercepts clicks.
export function AuthBackdrop() {
  return (
    <div aria-hidden className="pointer-events-none absolute inset-0 overflow-hidden">
      <div
        className="absolute inset-0 opacity-[0.05]"
        style={{
          backgroundImage: "radial-gradient(var(--border) 1px, transparent 1px)",
          backgroundSize: "24px 24px",
        }}
      />
      <div
        className="absolute -top-[280px] left-1/2 h-[560px] w-[820px] -translate-x-1/2 rounded-full opacity-[0.14] blur-[110px]"
        style={{ background: "#297AFF" }}
      />
    </div>
  );
}

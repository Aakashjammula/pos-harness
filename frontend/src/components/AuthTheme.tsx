"use client";

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
      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-accent">
        <svg viewBox="0 0 24 24" fill="#fff" className="h-4 w-4">
          <path d="M12 15a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3Zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-2.08A7 7 0 0 0 19 12h-2Z" />
        </svg>
      </span>
      <span className="font-mono text-[15px] font-semibold tracking-tight text-text">pos</span>
    </div>
  );
}

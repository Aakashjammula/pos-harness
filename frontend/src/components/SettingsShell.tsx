"use client";

import type { ReactNode } from "react";

export type SettingsTab = "account" | "security" | "model" | "tools";

const TABS: { id: SettingsTab; label: string }[] = [
  { id: "account", label: "Account" },
  { id: "security", label: "Security" },
  { id: "model", label: "Model & voice" },
  { id: "tools", label: "Tools" },
];

/** The route (#/settings, #/settings/security, ...) for a tab; #/tools is the older link to the Tools tab. */
export function tabFromHash(hash: string): SettingsTab | null {
  if (hash === "#/tools") return "tools";
  const match = /^#\/settings(?:\/(account|security|model|tools))?$/.exec(hash);
  return match ? ((match[1] as SettingsTab | undefined) ?? "account") : null;
}

export function hashForTab(tab: SettingsTab): string {
  return tab === "account" ? "#/settings" : `#/settings/${tab}`;
}

/** Settings: one page, four sections. Each section saves on its own. */
export function SettingsShell({
  tab,
  onTab,
  onBack,
  children,
}: {
  tab: SettingsTab;
  onTab: (tab: SettingsTab) => void;
  onBack: () => void;
  children: ReactNode;
}) {
  return (
    <div className="flex h-screen flex-1 flex-col">
      <div className="flex shrink-0 items-center gap-3.5 border-b border-border px-6 py-3.5">
        <button
          type="button"
          onClick={onBack}
          className="rounded-lg px-2.5 py-1.5 text-[13px] font-medium text-text-muted hover:bg-surface-sunken hover:text-text"
        >
          ← Back
        </button>
        <div className="text-[14.5px] font-semibold">Settings</div>
      </div>

      <nav aria-label="Settings sections" className="flex shrink-0 gap-1 border-b border-border px-6 pt-2">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            aria-current={t.id === tab ? "page" : undefined}
            onClick={() => onTab(t.id)}
            className={`-mb-px rounded-t-lg border-b-2 px-3.5 py-2 text-[13px] font-medium transition-colors ${
              t.id === tab
                ? "border-accent text-text"
                : "border-transparent text-text-muted hover:text-text"
            }`}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <div className="flex-1 overflow-y-auto px-6 pt-5 pb-10">
        <div className="max-w-[820px]">{children}</div>
      </div>
    </div>
  );
}

"use client";

import { useRouter } from "next/navigation";
import { formatSessionTimestamp } from "@/lib/format";
import { logout } from "@/lib/auth";
import type { SessionSummary } from "@/lib/types";
import { BrandMark } from "./icons";

interface SidebarProps {
  sessions: SessionSummary[] | null; // null = still loading / failed
  loadError: boolean;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  userEmail: string;
  onOpenSettings: () => void;
}

export function Sidebar({ sessions, loadError, onSelect, onDelete, userEmail, onOpenSettings }: SidebarProps) {
  const router = useRouter();
  return (
    <aside className="flex w-[260px] shrink-0 flex-col gap-0.5 overflow-y-auto bg-sidebar-bg p-2 max-[900px]:w-auto max-[900px]:max-h-[220px] max-[900px]:border-b max-[900px]:border-border">
      <div className="flex items-center gap-2 px-2 pt-2 pb-3.5">
        <BrandMark className="h-5 w-auto shrink-0" />
        <span className="font-mono text-sm font-semibold tracking-tight">pos</span>
      </div>
      <div className="px-2 pt-2.5 pb-1.5 text-[11.5px] font-medium text-text-faint">History</div>
      <div className="flex flex-col gap-px">
        {loadError ? (
          <div className="p-2 text-[12.5px] text-text-faint">Couldn&apos;t load history.</div>
        ) : sessions === null ? (
          <div className="p-2 text-[12.5px] text-text-faint">Loading…</div>
        ) : sessions.length === 0 ? (
          <div className="p-2 text-[12.5px] text-text-faint">No past sessions yet.</div>
        ) : (
          sessions.map((s) => (
            <div
              key={s.id}
              onClick={() => onSelect(s.id)}
              className="group flex cursor-pointer items-start gap-1 rounded-lg px-2 py-2.5 text-[13px] leading-snug text-text hover:bg-surface-sunken"
            >
              <div className="min-w-0 flex-1">
                <div className="truncate">{s.title || formatSessionTimestamp(s.created_at)}</div>
                <span className="mt-0.5 block text-[11.5px] text-text-faint">
                  {s.mode} · {s.turn_count} turn{s.turn_count === 1 ? "" : "s"}
                </span>
              </div>
              <span
                title="Delete this chat"
                onClick={(e) => {
                  e.stopPropagation();
                  if (window.confirm("Delete this chat? This can't be undone.")) onDelete(s.id);
                }}
                className="shrink-0 rounded-md px-1.5 py-0.5 text-xs text-text-faint opacity-0 group-hover:opacity-100 hover:bg-danger-tint hover:text-danger"
              >
                ✕
              </span>
            </div>
          ))
        )}
      </div>
      <div className="mt-auto border-t border-border px-2 pt-2">
        <div className="truncate px-2 py-1 text-[12px] text-text-faint">{userEmail}</div>
        <button
          type="button"
          onClick={onOpenSettings}
          className="w-full rounded-lg px-2 py-1.5 text-left text-[13px] text-text-muted hover:bg-surface-sunken hover:text-text"
        >
          Settings
        </button>
        <button
          type="button"
          onClick={async () => {
            await logout();
            router.replace("/login"); // replace, so Back does not return to the app
          }}
          className="w-full rounded-lg px-2 py-1.5 text-left text-[13px] text-text-muted hover:bg-surface-sunken hover:text-text"
        >
          Log out
        </button>
      </div>
    </aside>
  );
}

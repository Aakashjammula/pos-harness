"use client";

import type { Tool } from "@/lib/tools";
import type { OptionsResponse } from "@/lib/types";

// The original static/index.html lazy-loaded React Flow from a CDN (unpkg)
// on first Settings visit to draw an interactive provider->tools graph.
// That trick relies on a second, CDN-hosted copy of React sharing the page
// with this app's own React/Next.js bundle via ad hoc UMD globals, which
// doesn't fit cleanly into a bundled Next.js app (and would fight React's
// module singleton assumptions). This renders the same information --
// provider + which tools are wired/enabled -- as a static flex diagram
// instead of pulling in a graph library for one read-only diagram.
export function ConnectionsDiagram({
  options,
  llmConfigured,
  tools,
}: {
  options: OptionsResponse | null;
  llmConfigured: boolean;
  tools: Tool[];
}) {
  if (!options) {
    return <div className="mt-2.5 text-[12.5px] text-text-faint">Loading…</div>;
  }

  const providerLabel = !llmConfigured
    ? "No LLM configured"
    : options.provider.name
      ? `${options.provider.name} · ${options.provider.model}`
      : "Your saved provider";

  return (
    <div className="grid content-start gap-2.5">
      <div className="mt-2 flex min-h-[220px] w-full flex-col items-center gap-8 rounded-xl border border-border p-8">
        <div className="rounded-[10px] border border-border bg-surface-sunken px-[22px] py-3.5 text-center text-[15px] font-semibold text-text">
          {providerLabel}
        </div>
        <div className="flex flex-wrap justify-center gap-4">
          {tools.map((t) => (
            <div key={t.id} className="flex flex-col items-center gap-1.5">
              <span className={`h-6 w-px ${t.active ? "bg-text-faint" : "bg-border"}`} />
              <div
                className={`rounded-[10px] border border-border px-[22px] py-3.5 text-center text-[15px] font-semibold ${
                  t.active ? "bg-surface-sunken text-text" : "bg-surface-sunken text-text opacity-50"
                }`}
              >
                {t.active ? t.label : `${t.label} (off)`}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

"use client";

import { useState } from "react";
import { removeCredential } from "@/lib/credentials";
import { saveToolCredential, setToolEnabled, type Tool } from "@/lib/tools";

interface ToolsPanelProps {
  tools: Tool[];
  loading: boolean;
  error: string | null;
  onChanged: () => void; // a key or switch changed: re-fetch tools and saved-credential state
  onBack: () => void;
}

const inputClass =
  "w-full rounded-lg border border-transparent bg-surface-sunken px-2.5 py-2 text-[13px] text-text outline-none transition-[border-color,box-shadow] focus:border-accent focus:shadow-[0_0_0_3px_var(--accent-tint)]";

/** "tavily_api_key" -> "Tavily API key" */
function fieldLabel(name: string): string {
  const words = name.split("_").map((w) => (w === "api" ? "API" : w === "id" ? "ID" : w));
  const text = words.join(" ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function statusOf(tool: Tool): { text: string; className: string } {
  if (tool.requires_key && !tool.configured) return { text: "Needs API key", className: "text-text-faint" };
  return tool.active
    ? { text: "On", className: "text-accent" }
    : { text: "Off", className: "text-text-faint" };
}

function ToolCard({ tool, onChanged }: { tool: Tool; onChanged: () => void }) {
  const [values, setValues] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const status = statusOf(tool);
  const locked = tool.requires_key && !tool.configured;

  // Run one action, surface its error inline, then refresh everything.
  async function run(action: () => Promise<void>) {
    setBusy(true);
    setMessage(null);
    try {
      await action();
      onChanged();
    } catch (e) {
      setMessage((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="grid gap-3 rounded-xl border border-border p-4">
      <div className="flex items-start justify-between gap-4">
        <div className="grid gap-1">
          <div className="flex items-center gap-2">
            <h3 className="m-0 text-[14px] font-semibold text-text">{tool.label}</h3>
            <span className={`text-[11.5px] font-medium ${status.className}`}>{status.text}</span>
          </div>
          <p className="m-0 text-[12.5px] text-text-muted">{tool.description}</p>
        </div>
        <label className="flex shrink-0 items-center gap-2 text-[12.5px] text-text-muted">
          <input
            type="checkbox"
            role="switch"
            aria-label={`${tool.label} on or off`}
            checked={tool.active}
            disabled={busy || locked}
            onChange={(e) => run(() => setToolEnabled(tool.id, e.target.checked))}
            className="h-4 w-4 accent-[var(--accent)]"
          />
          {tool.active ? "On" : "Off"}
        </label>
      </div>

      {tool.requires_key && tool.credential_provider && (
        <div className="grid gap-2.5">
          {tool.credential_fields.map((field) => (
            <div key={field} className="grid gap-1.5">
              <label htmlFor={`${tool.id}-${field}`} className="text-[12.5px] text-text-muted">
                {fieldLabel(field)}
              </label>
              <input
                id={`${tool.id}-${field}`}
                type="password"
                autoComplete="off"
                placeholder={tool.configured ? "Saved — enter a new value to replace it" : ""}
                className={inputClass}
                value={values[field] ?? ""}
                onChange={(e) => setValues((v) => ({ ...v, [field]: e.target.value }))}
              />
            </div>
          ))}
          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={busy}
              onClick={() =>
                run(async () => {
                  await saveToolCredential(tool.credential_provider!, values);
                  setValues({}); // don't keep a secret in component state once it is saved
                })
              }
              className="rounded-lg bg-accent px-3 py-1.5 text-[12.5px] font-medium text-white hover:bg-accent-hover disabled:opacity-50"
            >
              Save
            </button>
            {tool.configured && (
              <button
                type="button"
                disabled={busy}
                onClick={() => run(() => removeCredential(tool.credential_provider!))}
                className="rounded-lg px-3 py-1.5 text-[12.5px] font-medium text-text-muted hover:bg-surface-sunken hover:text-text disabled:opacity-50"
              >
                Remove key
              </button>
            )}
          </div>
        </div>
      )}

      {message && (
        <p role="alert" className="m-0 text-xs text-red-500">
          {message}
        </p>
      )}
    </section>
  );
}

export function ToolsPanel({ tools, loading, error, onChanged, onBack }: ToolsPanelProps) {
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
        <div className="text-[14.5px] font-semibold">Tools</div>
      </div>

      <div className="flex-1 overflow-y-auto px-6 pt-4 pb-10">
        <div className="grid max-w-[720px] gap-3.5">
          <p className="m-0 text-[12.5px] text-text-muted">
            Tools the assistant may call during a chat. Switch each on or off; a tool that needs an API key stays off
            until you add yours.
          </p>
          {loading && <div className="text-[12.5px] text-text-faint">Loading…</div>}
          {error && (
            <p role="alert" className="m-0 text-xs text-red-500">
              Couldn&apos;t load tools: {error}
            </p>
          )}
          {tools.map((tool) => (
            <ToolCard key={tool.id} tool={tool} onChanged={onChanged} />
          ))}
        </div>
      </div>
    </div>
  );
}

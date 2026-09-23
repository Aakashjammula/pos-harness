"use client";

import type { ModelInfo } from "@/lib/models";
import { Dropdown } from "./Dropdown";
import { SystemPromptView } from "./SystemPromptView";
import { ToolGraph } from "./ToolGraph";
import { UsageDashboard } from "./UsageDashboard";

interface SettingsPageProps {
  models: ModelInfo[];
  model: string;
  onModelChange: (model: string) => void;
  levels: string[];
  level: string;
  onLevelChange: (level: string) => void;
  onBack: () => void;
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-text-faint">{label}</span>
      <span className="font-mono text-[11.5px] text-text-muted">{value}</span>
    </div>
  );
}

function Section({ title, description, children }: { title: string; description?: string; children: React.ReactNode }) {
  return (
    <section className="grid gap-3 rounded-xl border border-border p-4">
      <div className="grid gap-0.5">
        <h3 className="m-0 text-[14px] font-semibold text-text">{title}</h3>
        {description && <p className="m-0 text-[12.5px] text-text-muted">{description}</p>}
      </div>
      {children}
    </section>
  );
}

/**
 * Provider keys, tools, and accounts all lived here, backed by the server.
 * They're gone along with the backend that served them (it's being rebuilt
 * from scratch -- see the UI/backend replan). What's left is the one thing
 * that's purely local: your default model and reasoning level.
 */
export function SettingsPage({
  models,
  model,
  onModelChange,
  levels,
  level,
  onLevelChange,
  onBack,
}: SettingsPageProps) {
  const selected = models.find((m) => m.name === model);
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

      <div className="flex-1 overflow-y-auto px-6 pt-5 pb-10">
        <div className="grid max-w-[620px] gap-4">
          <Section
            title="Model"
            description="Names come from POS_MODELS in the backend's .env. Rates and limits are looked up."
          >
            {models.length > 1 ? (
              <Dropdown
                value={model}
                options={models.map((m) => m.name)}
                onChange={onModelChange}
                triggerClassName="text-text"
              />
            ) : (
              <span className="text-[13px] font-medium text-text">{model || "—"}</span>
            )}
            {selected && (
              <div className="grid gap-1 text-[12px]">
                <Row label="context window" value={selected.context_window?.toLocaleString() ?? "unknown"} />
                <Row label="max output" value={selected.max_output?.toLocaleString() ?? "unknown"} />
                <Row label="input / 1M" value={`$${selected.rates.short.input}`} />
                <Row label="output / 1M" value={`$${selected.rates.short.output}`} />
                <Row label="cached input / 1M" value={`$${selected.rates.short.cached}`} />
                {selected.rates.long_threshold && (
                  <Row
                    label={`above ${selected.rates.long_threshold.toLocaleString()} tokens`}
                    value={`$${selected.rates.long.input} in / $${selected.rates.long.output} out`}
                  />
                )}
                {!selected.known && (
                  <p className="m-0 text-[11.5px] text-danger">
                    No rates found for this name, so costs will show as $0. Set the prices in .env, or use
                    the model&apos;s own name as the deployment name.
                  </p>
                )}
              </div>
            )}
          </Section>
          <Section title="Reasoning level" description="How much the model thinks before answering.">
            <Dropdown value={level} options={levels} onChange={onLevelChange} triggerClassName="text-text" />
          </Section>
          <Section title="Usage" description="Tokens, cost and activity, from your own stored chats.">
            <UsageDashboard />
          </Section>
          <Section
            title="System prompt"
            description="The instructions the agent starts every chat with. Edit it in backend/src/pos/prompt.py."
          >
            <SystemPromptView />
          </Section>
          <Section title="Tools" description="What the agent can currently do. Keys are set in the backend's .env.">
            <ToolGraph />
          </Section>
        </div>
      </div>
    </div>
  );
}

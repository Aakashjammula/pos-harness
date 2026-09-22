"use client";

import { Dropdown } from "./Dropdown";
import { SystemPromptView } from "./SystemPromptView";
import { ToolGraph } from "./ToolGraph";
import { UsageDashboard } from "./UsageDashboard";

interface SettingsPageProps {
  model: string; // fixed -- the backend's one configured deployment, not a picker
  levels: string[];
  level: string;
  onLevelChange: (level: string) => void;
  onBack: () => void;
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
export function SettingsPage({ model, levels, level, onLevelChange, onBack }: SettingsPageProps) {
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
          <Section title="Model" description="Set in the backend's own .env, not from here.">
            <span className="text-[13px] font-medium text-text">{model}</span>
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

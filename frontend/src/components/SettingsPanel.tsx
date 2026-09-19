"use client";

import { hasLlm } from "@/lib/llm";
import { removeCredential, saveCredential, type ProviderModel } from "@/lib/credentials";
import type { Tool } from "@/lib/tools";
import type { ApiKeyFields, KeyProvider, OptionsResponse, Settings, SessionMode } from "@/lib/types";
import { ConnectionsDiagram } from "./ConnectionsDiagram";

interface SettingsPanelProps {
  settings: Settings;
  onSettingsChange: (patch: Partial<Settings>) => void;
  onKeysChange: (patch: Partial<ApiKeyFields>) => void;
  options: OptionsResponse | null;
  mics: MediaDeviceInfo[];
  disabled: boolean;
  onBack: () => void;
  mode: SessionMode;
  configured: string[];
  providerModels: { listable: boolean; models: ProviderModel[]; loading: boolean; error: string | null };
  llmModel: string; // the model a session would use right now (derived, see lib/llm.ts)
  tools: Tool[];
  onCredentialsChanged: () => void;
}

const PROVIDERS: { value: Exclude<KeyProvider, "">; label: string }[] = [
  { value: "local", label: "Local (LM Studio)" },
  { value: "openai", label: "OpenAI" },
  { value: "azure", label: "Azure OpenAI" },
  { value: "anthropic", label: "Anthropic" },
  { value: "gemini", label: "Google Gemini" },
  { value: "bedrock", label: "AWS Bedrock" },
  { value: "openrouter", label: "OpenRouter" },
];

const selectClass =
  "w-full rounded-lg border border-transparent bg-surface-sunken px-2.5 py-2 text-[13px] text-text outline-none transition-[border-color,box-shadow] focus:border-accent focus:shadow-[0_0_0_3px_var(--accent-tint)] disabled:cursor-not-allowed disabled:opacity-50";

function Field({ label, htmlFor, children }: { label: string; htmlFor: string; children: React.ReactNode }) {
  return (
    <div className="grid gap-1.5">
      <label htmlFor={htmlFor} className="text-[12.5px] text-text-muted">
        {label}
      </label>
      {children}
    </div>
  );
}

function SaveRemoveRow({
  configured,
  onSave,
  onRemove,
}: {
  configured: boolean;
  onSave: () => Promise<void>;
  onRemove: () => Promise<void>;
}) {
  return (
    <div className="flex items-center gap-2">
      <button
        type="button"
        onClick={onSave}
        className="rounded-lg bg-accent px-3 py-1.5 text-[12.5px] font-medium text-white hover:bg-accent-hover"
      >
        Save
      </button>
      {configured && (
        <button
          type="button"
          onClick={onRemove}
          className="rounded-lg px-3 py-1.5 text-[12.5px] text-danger hover:bg-danger-tint"
        >
          Remove
        </button>
      )}
    </div>
  );
}

export function SettingsPanel({
  settings,
  onSettingsChange,
  onKeysChange,
  options,
  mics,
  disabled,
  onBack,
  mode,
  configured,
  providerModels,
  llmModel,
  tools,
  onCredentialsChanged,
}: SettingsPanelProps) {
  const textMode = mode === "text";
  const voices = options?.tts[settings.ttsEngine] || [];

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

      <div className="flex-1 overflow-y-auto px-6 pt-2 pb-10">
        <div className="grid max-w-[820px] gap-[18px] py-1 pb-5 [grid-template-columns:repeat(auto-fit,minmax(220px,1fr))]">
          <div className="grid content-start gap-2.5">
            <h3 className="m-0 text-[11px] font-semibold tracking-wide text-text-faint">Model</h3>
            {!textMode && (
              <>
                <Field label="Voice engine" htmlFor="ttsEngine">
                  <select
                    id="ttsEngine"
                    disabled={disabled}
                    className={selectClass}
                    value={settings.ttsEngine}
                    onChange={(e) => onSettingsChange({ ttsEngine: e.target.value, ttsVoice: "" })}
                  >
                    {Object.keys(options?.tts || {}).map((engine) => (
                      <option key={engine} value={engine}>
                        {engine}
                      </option>
                    ))}
                  </select>
                </Field>
                <Field label="Voice" htmlFor="ttsVoice">
                  <select
                    id="ttsVoice"
                    disabled={disabled}
                    className={selectClass}
                    value={settings.ttsVoice}
                    onChange={(e) => onSettingsChange({ ttsVoice: e.target.value })}
                  >
                    {voices.map((voice) => (
                      <option key={voice} value={voice}>
                        {voice}
                      </option>
                    ))}
                  </select>
                </Field>
              </>
            )}
            <Field label="LLM model" htmlFor="llmModel">
              {settings.provider && providerModels.listable ? (
                <select
                  id="llmModel"
                  disabled={disabled || providerModels.loading || providerModels.models.length === 0}
                  className={selectClass}
                  value={llmModel}
                  onChange={(e) => onSettingsChange({ llmModel: e.target.value })}
                >
                  {providerModels.loading && <option value="">Loading models…</option>}
                  {!providerModels.loading && providerModels.models.length === 0 && (
                    <option value="">
                      {configured.includes(settings.provider) ? "No models available" : "Save a key to load models"}
                    </option>
                  )}
                  {providerModels.models.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.label === m.id ? m.id : `${m.label} (${m.id})`}
                    </option>
                  ))}
                </select>
              ) : (
                <select
                  id="llmModel"
                  disabled={disabled}
                  className={selectClass}
                  value={settings.llmModel}
                  onChange={(e) => onSettingsChange({ llmModel: e.target.value })}
                >
                  <option value="">Provider default</option>
                  {(options?.llm_models || []).map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))}
                </select>
              )}
            </Field>
            {providerModels.error && (
              <p role="alert" className="m-0 text-xs text-red-500">
                Couldn&apos;t load models: {providerModels.error}
              </p>
            )}
            {settings.provider === "azure" && (
              <p className="m-0 text-xs text-text-faint">
                Azure deployments can&apos;t be listed with a key — enter your deployment name below.
              </p>
            )}
            {options && !hasLlm(options, configured) && (
              <p className="m-0 text-xs text-text-faint">
                No LLM configured yet. Add your server URL or an API key in the provider settings.
              </p>
            )}
          </div>

          {!textMode && (
            <div className="grid content-start gap-2.5">
              <h3 className="m-0 text-[11px] font-semibold tracking-wide text-text-faint">Input</h3>
              <Field label="Microphone" htmlFor="micSelect">
                <select
                  id="micSelect"
                  disabled={disabled}
                  className={selectClass}
                  value={settings.micDeviceId}
                  onChange={(e) => onSettingsChange({ micDeviceId: e.target.value })}
                >
                  <option value="">System default</option>
                  {mics.map((d, i) => (
                    <option key={d.deviceId || i} value={d.deviceId}>
                      {d.label || `Microphone ${i + 1}`}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="Voice input" htmlFor="voiceInputMode">
                <select
                  id="voiceInputMode"
                  disabled={disabled}
                  className={selectClass}
                  value={settings.voiceInputMode}
                  onChange={(e) => onSettingsChange({ voiceInputMode: e.target.value as Settings["voiceInputMode"] })}
                >
                  <option value="vad">Voice detection (VAD)</option>
                  <option value="wake_word">Wake word</option>
                  <option value="push_to_talk">Push to talk (hold Space)</option>
                </select>
              </Field>
              {settings.voiceInputMode === "wake_word" && (
                <Field label="Wake word" htmlFor="triggerWord">
                  <input
                    id="triggerWord"
                    type="text"
                    disabled={disabled}
                    placeholder="e.g. computer"
                    className={selectClass}
                    value={settings.triggerWord}
                    onChange={(e) => onSettingsChange({ triggerWord: e.target.value })}
                  />
                </Field>
              )}
            </div>
          )}

          {!textMode && settings.voiceInputMode !== "push_to_talk" && (
            <div className="grid content-start gap-2.5">
              <h3 className="m-0 text-[11px] font-semibold tracking-wide text-text-faint">Voice detection</h3>
              <Field label="Sensitivity (0–1)" htmlFor="vadThreshold">
                <input
                  id="vadThreshold"
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  disabled={disabled}
                  className={selectClass}
                  value={settings.vadThreshold}
                  onChange={(e) => onSettingsChange({ vadThreshold: e.target.value })}
                />
              </Field>
              <div className="grid grid-cols-2 gap-2.5">
                <Field label="Pause length (ms)" htmlFor="vadMinSilenceMs">
                  <input
                    id="vadMinSilenceMs"
                    type="number"
                    min={0}
                    step={100}
                    disabled={disabled}
                    className={selectClass}
                    value={settings.vadMinSilenceMs}
                    onChange={(e) => onSettingsChange({ vadMinSilenceMs: e.target.value })}
                  />
                </Field>
                <Field label="Speech padding (ms)" htmlFor="vadSpeechPadMs">
                  <input
                    id="vadSpeechPadMs"
                    type="number"
                    min={0}
                    step={50}
                    disabled={disabled}
                    className={selectClass}
                    value={settings.vadSpeechPadMs}
                    onChange={(e) => onSettingsChange({ vadSpeechPadMs: e.target.value })}
                  />
                </Field>
              </div>
            </div>
          )}
        </div>

        <div className="mt-7 mb-1 text-[11px] font-semibold tracking-wide text-text-faint uppercase">API keys</div>
        <p className="mb-3.5 max-w-[640px] text-[12.5px] leading-relaxed text-text-faint">
          Saved securely on the server, encrypted at rest, and never sent back to the browser. Leave blank
          to use the server&apos;s own environment.
        </p>

        <div className="mb-1 flex max-w-[420px] flex-col gap-0.5">
          {PROVIDERS.map((p) => (
            <label
              key={p.value}
              className="flex cursor-pointer items-center gap-2.5 rounded-lg px-2.5 py-2 text-[13.5px] hover:bg-surface-sunken"
            >
              <input
                type="radio"
                name="keyProvider"
                value={p.value}
                checked={settings.provider === p.value}
                disabled={disabled}
                onChange={() => onSettingsChange({ provider: p.value })}
                className="shrink-0 accent-accent"
              />
              <span className="flex-1">{p.label}</span>
              <span
                className={`h-2 w-2 shrink-0 rounded-full ${
                  configured.includes(p.value) ? "bg-accent" : "bg-text-faint"
                }`}
              />
            </label>
          ))}
        </div>
        {!settings.provider && (
          <p className="mb-3.5 max-w-[640px] text-[12.5px] leading-relaxed text-text-faint">
            Select a provider above before connecting. ● configured &nbsp;○ not configured (optional for
            Local).
          </p>
        )}

        <div className="grid max-w-[820px] gap-[18px] py-1 pb-5 [grid-template-columns:repeat(auto-fit,minmax(220px,1fr))]">
          {settings.provider === "local" && (
            <div className="grid content-start gap-2.5">
              <h3 className="m-0 text-[11px] font-semibold tracking-wide text-text-faint">Local (LM Studio)</h3>
              <Field label="API key (optional)" htmlFor="localApiKey">
                <input
                  id="localApiKey"
                  type="password"
                  autoComplete="off"
                  placeholder="only if Require Authentication is on"
                  className={selectClass}
                  value={settings.keys.localApiKey}
                  onChange={(e) => onKeysChange({ localApiKey: e.target.value })}
                />
              </Field>
              <Field label="Base URL (optional)" htmlFor="localBaseUrl">
                <input
                  id="localBaseUrl"
                  type="text"
                  placeholder="http://your-server:port/v1"
                  className={selectClass}
                  value={settings.keys.localBaseUrl}
                  onChange={(e) => onKeysChange({ localBaseUrl: e.target.value })}
                />
              </Field>
              <SaveRemoveRow
                configured={configured.includes("local")}
                onSave={async () => {
                  await saveCredential("local", settings.keys);
                  onCredentialsChanged();
                }}
                onRemove={async () => {
                  await removeCredential("local");
                  onCredentialsChanged();
                }}
              />
            </div>
          )}
          {settings.provider === "openai" && (
            <div className="grid content-start gap-2.5">
              <h3 className="m-0 text-[11px] font-semibold tracking-wide text-text-faint">OpenAI</h3>
              <Field label="API key" htmlFor="openaiApiKey">
                <input
                  id="openaiApiKey"
                  type="password"
                  autoComplete="off"
                  placeholder="sk-…"
                  className={selectClass}
                  value={settings.keys.openaiApiKey}
                  onChange={(e) => onKeysChange({ openaiApiKey: e.target.value })}
                />
              </Field>
              <SaveRemoveRow
                configured={configured.includes("openai")}
                onSave={async () => {
                  await saveCredential("openai", settings.keys);
                  onCredentialsChanged();
                }}
                onRemove={async () => {
                  await removeCredential("openai");
                  onCredentialsChanged();
                }}
              />
            </div>
          )}
          {settings.provider === "azure" && (
            <div className="grid content-start gap-2.5">
              <h3 className="m-0 text-[11px] font-semibold tracking-wide text-text-faint">Azure OpenAI</h3>
              <Field label="API key" htmlFor="azureApiKey">
                <input
                  id="azureApiKey"
                  type="password"
                  autoComplete="off"
                  className={selectClass}
                  value={settings.keys.azureApiKey}
                  onChange={(e) => onKeysChange({ azureApiKey: e.target.value })}
                />
              </Field>
              <Field label="Endpoint" htmlFor="azureEndpoint">
                <input
                  id="azureEndpoint"
                  type="text"
                  placeholder="https://…openai.azure.com/"
                  className={selectClass}
                  value={settings.keys.azureEndpoint}
                  onChange={(e) => onKeysChange({ azureEndpoint: e.target.value })}
                />
              </Field>
              <Field label="Deployment" htmlFor="azureDeployment">
                <input
                  id="azureDeployment"
                  type="text"
                  className={selectClass}
                  value={settings.keys.azureDeployment}
                  onChange={(e) => onKeysChange({ azureDeployment: e.target.value })}
                />
              </Field>
              <SaveRemoveRow
                configured={configured.includes("azure")}
                onSave={async () => {
                  await saveCredential("azure", settings.keys);
                  onCredentialsChanged();
                }}
                onRemove={async () => {
                  await removeCredential("azure");
                  onCredentialsChanged();
                }}
              />
            </div>
          )}
          {settings.provider === "anthropic" && (
            <div className="grid content-start gap-2.5">
              <h3 className="m-0 text-[11px] font-semibold tracking-wide text-text-faint">Anthropic</h3>
              <Field label="API key" htmlFor="anthropicApiKey">
                <input
                  id="anthropicApiKey"
                  type="password"
                  autoComplete="off"
                  placeholder="sk-ant-…"
                  className={selectClass}
                  value={settings.keys.anthropicApiKey}
                  onChange={(e) => onKeysChange({ anthropicApiKey: e.target.value })}
                />
              </Field>
              <SaveRemoveRow
                configured={configured.includes("anthropic")}
                onSave={async () => {
                  await saveCredential("anthropic", settings.keys);
                  onCredentialsChanged();
                }}
                onRemove={async () => {
                  await removeCredential("anthropic");
                  onCredentialsChanged();
                }}
              />
            </div>
          )}
          {settings.provider === "gemini" && (
            <div className="grid content-start gap-2.5">
              <h3 className="m-0 text-[11px] font-semibold tracking-wide text-text-faint">Google Gemini</h3>
              <Field label="API key" htmlFor="geminiApiKey">
                <input
                  id="geminiApiKey"
                  type="password"
                  autoComplete="off"
                  className={selectClass}
                  value={settings.keys.geminiApiKey}
                  onChange={(e) => onKeysChange({ geminiApiKey: e.target.value })}
                />
              </Field>
              <SaveRemoveRow
                configured={configured.includes("gemini")}
                onSave={async () => {
                  await saveCredential("gemini", settings.keys);
                  onCredentialsChanged();
                }}
                onRemove={async () => {
                  await removeCredential("gemini");
                  onCredentialsChanged();
                }}
              />
            </div>
          )}
          {settings.provider === "bedrock" && (
            <div className="grid content-start gap-2.5">
              <h3 className="m-0 text-[11px] font-semibold tracking-wide text-text-faint">AWS Bedrock</h3>
              <Field label="Access key ID" htmlFor="bedrockAccessKeyId">
                <input
                  id="bedrockAccessKeyId"
                  type="text"
                  autoComplete="off"
                  placeholder="AKIA…"
                  className={selectClass}
                  value={settings.keys.bedrockAccessKeyId}
                  onChange={(e) => onKeysChange({ bedrockAccessKeyId: e.target.value })}
                />
              </Field>
              <Field label="Secret access key" htmlFor="bedrockSecretAccessKey">
                <input
                  id="bedrockSecretAccessKey"
                  type="password"
                  autoComplete="off"
                  className={selectClass}
                  value={settings.keys.bedrockSecretAccessKey}
                  onChange={(e) => onKeysChange({ bedrockSecretAccessKey: e.target.value })}
                />
              </Field>
              <Field label="Region (optional)" htmlFor="bedrockRegion">
                <input
                  id="bedrockRegion"
                  type="text"
                  placeholder="us-east-1"
                  className={selectClass}
                  value={settings.keys.bedrockRegion}
                  onChange={(e) => onKeysChange({ bedrockRegion: e.target.value })}
                />
              </Field>
              <SaveRemoveRow
                configured={configured.includes("bedrock")}
                onSave={async () => {
                  await saveCredential("bedrock", settings.keys);
                  onCredentialsChanged();
                }}
                onRemove={async () => {
                  await removeCredential("bedrock");
                  onCredentialsChanged();
                }}
              />
            </div>
          )}
          {settings.provider === "openrouter" && (
            <div className="grid content-start gap-2.5">
              <h3 className="m-0 text-[11px] font-semibold tracking-wide text-text-faint">OpenRouter</h3>
              <Field label="API key" htmlFor="openrouterApiKey">
                <input
                  id="openrouterApiKey"
                  type="password"
                  autoComplete="off"
                  placeholder="sk-or-…"
                  className={selectClass}
                  value={settings.keys.openrouterApiKey}
                  onChange={(e) => onKeysChange({ openrouterApiKey: e.target.value })}
                />
              </Field>
              <SaveRemoveRow
                configured={configured.includes("openrouter")}
                onSave={async () => {
                  await saveCredential("openrouter", settings.keys);
                  onCredentialsChanged();
                }}
                onRemove={async () => {
                  await removeCredential("openrouter");
                  onCredentialsChanged();
                }}
              />
            </div>
          )}
        </div>

        <div className="mt-7 mb-1 text-[11px] font-semibold tracking-wide text-text-faint uppercase">
          Connections
        </div>
        <ConnectionsDiagram options={options} llmConfigured={hasLlm(options, configured)} tools={tools} />
      </div>
    </div>
  );
}

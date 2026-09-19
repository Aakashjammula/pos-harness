import { apiFetch } from "./apiFetch";
import type { ApiKeyFields, KeyProvider } from "./types";

const PROVIDER_PAYLOAD: Record<string, (k: ApiKeyFields) => Record<string, string>> = {
  local: (k) => ({ local_api_key: k.localApiKey, local_base_url: k.localBaseUrl }),
  openai: (k) => ({ openai_api_key: k.openaiApiKey }),
  azure: (k) => ({
    azure_api_key: k.azureApiKey,
    azure_endpoint: k.azureEndpoint,
    azure_deployment: k.azureDeployment,
  }),
  anthropic: (k) => ({ anthropic_api_key: k.anthropicApiKey }),
  gemini: (k) => ({ gemini_api_key: k.geminiApiKey }),
  bedrock: (k) => ({
    bedrock_access_key_id: k.bedrockAccessKeyId,
    bedrock_secret_access_key: k.bedrockSecretAccessKey,
    bedrock_region: k.bedrockRegion,
  }),
  openrouter: (k) => ({ openrouter_api_key: k.openrouterApiKey }),
};

/** What is already saved. Keys never come back; `hints` are their last four characters and
 * `public` holds the non-secret settings (server URL, Azure endpoint/deployment, AWS region). */
export interface CredentialSummary {
  configured: string[];
  public: Record<string, Record<string, string>>;
  hints: Record<string, string>;
}

export async function fetchCredentialSummary(): Promise<CredentialSummary> {
  const res = await apiFetch("/credentials");
  if (!res.ok) throw new Error("Couldn't load saved credentials.");
  const body = await res.json();
  return { configured: body.configured, public: body.public ?? {}, hints: body.hints ?? {} };
}

export async function saveCredential(provider: Exclude<KeyProvider, "">, keys: ApiKeyFields): Promise<void> {
  const res = await apiFetch(`/credentials/${provider}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(PROVIDER_PAYLOAD[provider](keys)),
  });
  if (res.status === 422) throw new Error("Fill in at least one field before saving.");
  if (!res.ok) throw new Error("Couldn't save. Try again.");
}

export async function removeCredential(provider: string): Promise<void> {
  await apiFetch(`/credentials/${provider}`, { method: "DELETE" });
}

export interface ProviderModel {
  id: string;
  label: string;
  chat?: boolean; // answers in text. false = image/audio/music/agent/etc., hidden unless "show all"
  context_window?: number; // input tokens, when the provider says (OpenAI does not)
}

/** The models this user's saved credential can use, asked of the provider by
 * the backend (the key never reaches the browser). Throws with the reason. */
export async function fetchProviderModels(provider: string): Promise<ProviderModel[]> {
  const res = await apiFetch(`/credentials/${provider}/models`);
  if (!res.ok) {
    let detail = "";
    try {
      const body = (await res.json()).detail;
      if (typeof body === "string") detail = body; // a validation error carries a list, not text
    } catch {
      // not JSON -- fall through to the generic message
    }
    throw new Error(detail || `Couldn't load models (HTTP ${res.status}).`);
  }
  return (await res.json()).models;
}

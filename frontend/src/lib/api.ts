import { API_URL } from "./config";
import type {
  ApiKeyFields,
  KeyProvider,
  OptionsResponse,
  SessionDetail,
  SessionSummary,
} from "./types";

export async function fetchOptions(): Promise<OptionsResponse> {
  const res = await fetch(`${API_URL}/options`);
  if (!res.ok) throw new Error(`GET /options failed: ${res.status}`);
  return res.json();
}

export async function fetchSessions(): Promise<SessionSummary[]> {
  const res = await fetch(`${API_URL}/sessions`);
  if (!res.ok) throw new Error(`GET /sessions failed: ${res.status}`);
  return res.json();
}

export async function fetchSession(id: string): Promise<SessionDetail> {
  const res = await fetch(`${API_URL}/sessions/${id}`);
  if (!res.ok) throw new Error(`GET /sessions/${id} failed: ${res.status}`);
  return res.json();
}

export async function deleteSession(id: string): Promise<void> {
  await fetch(`${API_URL}/sessions/${id}`, { method: "DELETE" });
}

const PROVIDER_LABELS: Record<Exclude<KeyProvider, "" | "local">, string> = {
  openai: "OpenAI",
  azure: "Azure",
  anthropic: "Anthropic",
  gemini: "Gemini",
  bedrock: "AWS Bedrock",
  openrouter: "OpenRouter",
};

// A provider counts as "configured" once its REQUIRED field(s) are
// filled -- Local's fields are both optional, so it's always considered
// configured (nothing needed to use the server's own default local setup).
export function isProviderConfigured(provider: KeyProvider, keys: ApiKeyFields): boolean {
  switch (provider) {
    case "local":
      return true;
    case "openai":
      return !!keys.openaiApiKey.trim();
    case "azure":
      return !!(keys.azureApiKey.trim() && keys.azureEndpoint.trim() && keys.azureDeployment.trim());
    case "anthropic":
      return !!keys.anthropicApiKey.trim();
    case "gemini":
      return !!keys.geminiApiKey.trim();
    case "bedrock":
      return !!(keys.bedrockAccessKeyId.trim() && keys.bedrockSecretAccessKey.trim());
    case "openrouter":
      return !!keys.openrouterApiKey.trim();
    default:
      return false;
  }
}

export function keyProviderValidationError(provider: KeyProvider, keys: ApiKeyFields): string | null {
  if (!provider) return "Select a provider in Settings before connecting.";
  if (provider !== "local" && !isProviderConfigured(provider, keys)) {
    return `Fill in your ${PROVIDER_LABELS[provider]} API key in Settings, or switch Provider back to Local.`;
  }
  return null;
}

// Only the selected provider's fields are ever sent -- switching Provider
// back to Local means "no override" even if a key is still sitting in a
// now-hidden field from an earlier selection. Held in server memory only
// for the lifetime of one connection (see backend's POST /session-keys).
export async function fetchKeyTokenIfNeeded(
  provider: KeyProvider,
  keys: ApiKeyFields
): Promise<string | null> {
  const body: Record<string, string> = {};
  if (provider === "local") {
    if (keys.localApiKey.trim()) body.local_api_key = keys.localApiKey.trim();
    if (keys.localBaseUrl.trim()) body.local_base_url = keys.localBaseUrl.trim();
  } else if (provider === "openai" && keys.openaiApiKey.trim()) {
    body.openai_api_key = keys.openaiApiKey.trim();
  } else if (provider === "azure") {
    if (keys.azureApiKey.trim()) body.azure_api_key = keys.azureApiKey.trim();
    if (keys.azureEndpoint.trim()) body.azure_endpoint = keys.azureEndpoint.trim();
    if (keys.azureDeployment.trim()) body.azure_deployment = keys.azureDeployment.trim();
  } else if (provider === "anthropic" && keys.anthropicApiKey.trim()) {
    body.anthropic_api_key = keys.anthropicApiKey.trim();
  } else if (provider === "gemini" && keys.geminiApiKey.trim()) {
    body.gemini_api_key = keys.geminiApiKey.trim();
  } else if (provider === "bedrock") {
    if (keys.bedrockAccessKeyId.trim()) body.bedrock_access_key_id = keys.bedrockAccessKeyId.trim();
    if (keys.bedrockSecretAccessKey.trim()) body.bedrock_secret_access_key = keys.bedrockSecretAccessKey.trim();
    if (keys.bedrockRegion.trim()) body.bedrock_region = keys.bedrockRegion.trim();
  } else if (provider === "openrouter" && keys.openrouterApiKey.trim()) {
    body.openrouter_api_key = keys.openrouterApiKey.trim();
  }
  // Web search is independent of the LLM provider choice.
  if (keys.tavilyApiKey.trim()) body.tavily_api_key = keys.tavilyApiKey.trim();
  if (Object.keys(body).length === 0) return null;

  const res = await fetch(`${API_URL}/session-keys`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`session-keys request failed: ${res.status}`);
  const data = await res.json();
  return data.key_token as string;
}

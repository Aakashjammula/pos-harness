import { API_URL } from "./config";
import type {
  ApiKeyFields,
  KeyProvider,
  OptionsResponse,
  SessionDetail,
  SessionSummary,
} from "./types";

export async function fetchOptions(): Promise<OptionsResponse> {
  const res = await fetch(`${API_URL}/options`, { credentials: "include" });
  if (!res.ok) throw new Error(`GET /options failed: ${res.status}`);
  return res.json();
}

export async function fetchSessions(): Promise<SessionSummary[]> {
  const res = await fetch(`${API_URL}/sessions`, { credentials: "include" });
  if (!res.ok) throw new Error(`GET /sessions failed: ${res.status}`);
  return res.json();
}

export async function fetchSession(id: string): Promise<SessionDetail> {
  const res = await fetch(`${API_URL}/sessions/${id}`, { credentials: "include" });
  if (!res.ok) throw new Error(`GET /sessions/${id} failed: ${res.status}`);
  return res.json();
}

export async function deleteSession(id: string): Promise<void> {
  await fetch(`${API_URL}/sessions/${id}`, { method: "DELETE", credentials: "include" });
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

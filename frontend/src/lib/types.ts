export type SessionMode = "voice" | "text";

export type VoiceInputMode = "vad" | "wake_word" | "push_to_talk";

export type KeyProvider =
  | ""
  | "local"
  | "openai"
  | "azure"
  | "anthropic"
  | "gemini"
  | "bedrock"
  | "openrouter";

export type ConnState =
  | "idle"
  | "connecting"
  | "listening"
  | "speaking"
  | "muted"
  | "error";

export interface ToolStatus {
  name: string;
  label: string;
  enabled: boolean;
}

export interface OptionsResponse {
  tts: Record<string, string[]>;
  llm_models: string[];
  defaults: { tts_engine: string; llm_model: string };
  provider: { name: string; model: string };
  llm_configured: boolean;
  tools: ToolStatus[];
}

export interface ToolCall {
  name: string;
  args?: Record<string, unknown>;
  result?: unknown;
}

export interface Usage {
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
  context_window?: number;
  cost_usd?: number;
  tool_calls?: ToolCall[];
}

export interface Latency {
  ttft: number;
  total: number;
}

export interface SessionSummary {
  id: string;
  mode: SessionMode;
  turn_count: number;
  title: string | null;
  created_at: string;
}

export interface StoredTurn {
  role: "user" | "assistant";
  text: string;
  usage?: Usage;
}

export interface SessionDetail {
  session: { id: string; mode: SessionMode; created_at: string; title: string | null };
  turns: StoredTurn[];
}

export interface TranscriptLine {
  id: string;
  who: "you" | "bot" | "system";
  text: string;
  usage?: Usage;
  latency?: Latency;
}

export interface ApiKeyFields {
  localApiKey: string;
  localBaseUrl: string;
  openaiApiKey: string;
  azureApiKey: string;
  azureEndpoint: string;
  azureDeployment: string;
  anthropicApiKey: string;
  geminiApiKey: string;
  bedrockAccessKeyId: string;
  bedrockSecretAccessKey: string;
  bedrockRegion: string;
  openrouterApiKey: string;
  tavilyApiKey: string;
}

export const EMPTY_KEY_FIELDS: ApiKeyFields = {
  localApiKey: "",
  localBaseUrl: "",
  openaiApiKey: "",
  azureApiKey: "",
  azureEndpoint: "",
  azureDeployment: "",
  anthropicApiKey: "",
  geminiApiKey: "",
  bedrockAccessKeyId: "",
  bedrockSecretAccessKey: "",
  bedrockRegion: "",
  openrouterApiKey: "",
  tavilyApiKey: "",
};

export interface Settings {
  ttsEngine: string;
  ttsVoice: string;
  llmModel: string;
  micDeviceId: string;
  voiceInputMode: VoiceInputMode;
  triggerWord: string;
  vadThreshold: string;
  vadMinSilenceMs: string;
  vadSpeechPadMs: string;
  provider: KeyProvider;
  keys: ApiKeyFields;
}

export const DEFAULT_SETTINGS: Settings = {
  ttsEngine: "",
  ttsVoice: "",
  llmModel: "",
  micDeviceId: "",
  voiceInputMode: "vad",
  triggerWord: "",
  vadThreshold: "0.5",
  vadMinSilenceMs: "1200",
  vadSpeechPadMs: "300",
  provider: "",
  keys: EMPTY_KEY_FIELDS,
};

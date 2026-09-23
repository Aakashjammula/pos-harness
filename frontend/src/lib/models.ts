import { API_URL } from "./config";

export interface ModelRates {
  input: number;
  cached: number;
  cache_write: number;
  output: number;
}

export interface ModelInfo {
  name: string;
  context_window: number | null;
  max_output: number | null;
  reasoning: boolean;
  /** False when neither the catalogue nor .env knew this model -- its rates
   * are zero, so costs will read $0.0000. */
  known: boolean;
  rates: { short: ModelRates; long: ModelRates; long_threshold: number | null };
}

export interface ModelList {
  provider: string;
  default: string;
  models: ModelInfo[];
  /** Null when the backend is usable; otherwise names what to set in .env. */
  config_error: string | null;
}

export async function fetchModels(): Promise<ModelList> {
  const res = await fetch(`${API_URL}/models`);
  if (!res.ok) throw new Error(`GET /models failed: ${res.status}`);
  return res.json();
}

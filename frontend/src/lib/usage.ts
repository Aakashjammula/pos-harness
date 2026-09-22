import { API_URL } from "./config";

export interface DayUsage {
  day: string;
  input: number;
  output: number;
  messages: number;
  cost: number;
}

export interface ModelUsage {
  model: string;
  input: number;
  output: number;
  messages: number;
  cost: number;
  share: number;
}

export interface UsageSummary {
  range: string;
  sessions: number;
  messages: number;
  active_days: number;
  peak_hour: number | null;
  favorite_model: string | null;
  totals: {
    input: number;
    output: number;
    total: number;
    cost: number;
    cache_read: number;
    cache_creation: number;
  };
  by_day: DayUsage[];
  by_model: ModelUsage[];
}

export async function fetchUsage(range: string): Promise<UsageSummary> {
  const res = await fetch(`${API_URL}/usage?range=${encodeURIComponent(range)}`);
  if (!res.ok) throw new Error(`GET /usage failed: ${res.status}`);
  return res.json();
}

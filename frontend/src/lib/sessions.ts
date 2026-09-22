import { API_URL } from "./config";
import type { ToolCall, Usage } from "./types";

export interface SessionSummary {
  id: string;
  folder: string | null;
  title: string | null;
  created_at: string;
}

export async function fetchSessions(folder?: string | null): Promise<SessionSummary[]> {
  const url = folder ? `${API_URL}/sessions?folder=${encodeURIComponent(folder)}` : `${API_URL}/sessions`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GET /sessions failed: ${res.status}`);
  return res.json();
}

export interface StoredTurn {
  role: "user" | "assistant";
  text: string;
  model: string | null;
  finish_reason: string | null;
  reasoning_effort: string | null;
  usage: Usage | null;
  // Stored alongside usage rather than inside it -- the API returns them as
  // a sibling field.
  tool_calls: ToolCall[] | null;
  created_at: string;
}

export async function fetchSession(id: string): Promise<{ turns: StoredTurn[] }> {
  const res = await fetch(`${API_URL}/sessions/${id}`);
  if (!res.ok) throw new Error(`GET /sessions/${id} failed: ${res.status}`);
  return res.json();
}

export async function deleteSession(id: string): Promise<void> {
  const res = await fetch(`${API_URL}/sessions/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`DELETE /sessions/${id} failed: ${res.status}`);
}

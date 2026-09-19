import { API_URL } from "./config";
import { apiFetch } from "./apiFetch";
import type { OptionsResponse, SessionDetail, SessionSummary } from "./types";

export async function fetchOptions(): Promise<OptionsResponse> {
  const res = await fetch(`${API_URL}/options`, { credentials: "include" });
  if (!res.ok) throw new Error(`GET /options failed: ${res.status}`);
  return res.json();
}

export async function fetchSessions(): Promise<SessionSummary[]> {
  const res = await apiFetch("/sessions");
  if (!res.ok) throw new Error(`GET /sessions failed: ${res.status}`);
  return res.json();
}

export async function fetchSession(id: string): Promise<SessionDetail> {
  const res = await apiFetch(`/sessions/${id}`);
  if (!res.ok) throw new Error(`GET /sessions/${id} failed: ${res.status}`);
  return res.json();
}

export async function deleteSession(id: string): Promise<void> {
  await apiFetch(`/sessions/${id}`, { method: "DELETE" });
}

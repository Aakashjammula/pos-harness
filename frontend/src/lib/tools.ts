import { API_URL } from "./config";

export interface ToolInfo {
  name: string;
  description: string;
  source: "filesystem" | "web" | string;
  active: boolean;
}

export interface SkillInfo {
  name: string;
  description: string;
}

export async function fetchTools(): Promise<{ tools: ToolInfo[]; skills: SkillInfo[]; model: string }> {
  const res = await fetch(`${API_URL}/tools`);
  if (!res.ok) throw new Error(`GET /tools failed: ${res.status}`);
  return res.json();
}

export async function fetchSystemPrompt(): Promise<string> {
  const res = await fetch(`${API_URL}/system-prompt`);
  if (!res.ok) throw new Error(`GET /system-prompt failed: ${res.status}`);
  const data = await res.json();
  return data?.prompt ?? "";
}

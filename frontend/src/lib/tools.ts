import { API_URL } from "./config";

export interface Tool {
  id: string;
  label: string;
  description: string;
  requires_key: boolean;
  credential_provider: string | null;
  credential_fields: string[];
  configured: boolean; // its key is present (saved by this user, or set on the server)
  enabled: boolean; // this user's own switch
  active: boolean; // enabled and configured: will actually be bound
}

async function detail(res: Response, fallback: string): Promise<string> {
  try {
    return (await res.json()).detail || fallback;
  } catch {
    return fallback;
  }
}

export async function fetchTools(): Promise<Tool[]> {
  const res = await fetch(`${API_URL}/tools`, { credentials: "include" });
  if (!res.ok) throw new Error(await detail(res, `Couldn't load tools (HTTP ${res.status}).`));
  return (await res.json()).tools;
}

export async function setToolEnabled(id: string, enabled: boolean): Promise<void> {
  const res = await fetch(`${API_URL}/tools/${id}`, {
    method: "PUT",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
  if (!res.ok) throw new Error(await detail(res, "Couldn't update the tool."));
}

/** Save a tool's key(s). Blank fields are dropped and an all-blank form is
 * rejected here, so it never costs a round trip that ends in a 422. */
export async function saveToolCredential(provider: string, fields: Record<string, string>): Promise<void> {
  const cleaned = Object.fromEntries(
    Object.entries(fields)
      .map(([k, v]) => [k, v.trim()] as const)
      .filter(([, v]) => v)
  );
  if (Object.keys(cleaned).length === 0) throw new Error("Enter the key before saving.");
  const res = await fetch(`${API_URL}/credentials/${provider}`, {
    method: "PUT",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cleaned),
  });
  if (!res.ok) throw new Error(await detail(res, "Couldn't save. Try again."));
}

import { API_URL } from "./config";

export interface CurrentUser {
  id: string;
  email: string;
}

async function post(path: string, body?: unknown): Promise<Response> {
  return fetch(`${API_URL}${path}`, {
    method: "POST",
    credentials: "include",
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
}

export async function login(email: string, password: string): Promise<CurrentUser> {
  const res = await post("/auth/login", { email, password });
  if (res.status === 401) throw new Error("Invalid email or password.");
  if (!res.ok) throw new Error("Couldn't sign you in. Try again.");
  return res.json();
}

// Always resolves -- the backend returns 200 regardless of whether the
// email is known or a link was just sent a moment ago, so there's
// nothing meaningful to branch on here besides a network failure.
export async function requestMagicLink(email: string): Promise<void> {
  const res = await post("/auth/magic-link/request", { email });
  if (!res.ok) throw new Error("Couldn't send the link. Try again.");
}

export async function verifyMagicLink(token: string): Promise<CurrentUser> {
  const res = await post("/auth/magic-link/verify", { token });
  if (res.status === 401) throw new Error("That link is invalid or has expired.");
  if (!res.ok) throw new Error("Couldn't sign you in. Try again.");
  return res.json();
}

export async function logout(): Promise<void> {
  await post("/auth/logout");
}

export async function fetchMe(): Promise<CurrentUser | null> {
  const res = await fetch(`${API_URL}/auth/me`, { credentials: "include" });
  if (res.status === 401) return null;
  if (!res.ok) throw new Error("Couldn't load your account.");
  return res.json();
}

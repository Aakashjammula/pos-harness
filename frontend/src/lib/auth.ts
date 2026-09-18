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

export async function signup(email: string, password: string): Promise<CurrentUser> {
  const res = await post("/auth/signup", { email, password });
  if (res.status === 409) throw new Error("That email is already registered.");
  if (res.status === 422) throw new Error("Password must be at least 8 characters.");
  if (!res.ok) throw new Error("Couldn't create your account. Try again.");
  return res.json();
}

export async function login(email: string, password: string): Promise<CurrentUser> {
  const res = await post("/auth/login", { email, password });
  if (res.status === 401) throw new Error("Invalid email or password.");
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

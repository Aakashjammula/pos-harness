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

export interface SignupInput {
  name: string;
  username: string;
  email: string;
  /** Omitted entirely for a magic-link-only account. */
  password?: string;
}

export interface SignupResult extends CurrentUser {
  /** True when no password was given: the account exists but this
   * browser is NOT signed in -- the link in their inbox is. */
  magic_link_sent: boolean;
}

export async function signup(input: SignupInput): Promise<SignupResult> {
  const body: Record<string, unknown> = {
    name: input.name,
    username: input.username,
    email: input.email,
  };
  if (input.password) body.password = input.password;

  const res = await post("/auth/signup", body);
  if (res.status === 409) {
    // The two collisions need different words -- "email taken" sends
    // someone to the login page, "username taken" sends them back to
    // the same form to pick another.
    const detail = await res.json().catch(() => null);
    throw new Error(
      detail?.detail === "username already taken"
        ? "That username is taken. Try another."
        : "That email is already registered.",
    );
  }
  if (res.status === 422) {
    throw new Error(
      "Check your details: username can use letters, numbers, - and _ (3+ characters), and a password must be 8+ characters.",
    );
  }
  if (!res.ok) throw new Error("Couldn't create your account. Try again.");
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

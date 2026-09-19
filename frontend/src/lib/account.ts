import { apiFetch } from "./apiFetch";
import type { CurrentUser } from "./auth";

/** One signed-in browser/device. Only ever the caller's own. */
export interface LoginSession {
  id: string;
  user_agent: string | null;
  ip: string | null;
  created_at: string;
  last_seen_at: string;
  current: boolean;
}

async function messageOf(res: Response, fallback: string): Promise<string> {
  try {
    const detail = (await res.json()).detail;
    return typeof detail === "string" && detail ? detail : fallback; // validation errors carry a list
  } catch {
    return fallback;
  }
}

async function json(path: string, method: string, body?: unknown): Promise<Response> {
  return apiFetch(path, {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

export async function fetchLoginSessions(): Promise<LoginSession[]> {
  const res = await apiFetch("/auth/sessions");
  if (!res.ok) throw new Error(await messageOf(res, `Couldn't load your sessions (HTTP ${res.status}).`));
  return (await res.json()).sessions;
}

export async function revokeLoginSession(id: string): Promise<void> {
  const res = await json(`/auth/sessions/${id}`, "DELETE");
  if (!res.ok) throw new Error(await messageOf(res, "Couldn't sign that device out."));
}

export async function revokeOtherLoginSessions(): Promise<number> {
  const res = await json("/auth/sessions/revoke-others", "POST");
  if (!res.ok) throw new Error(await messageOf(res, "Couldn't sign the other devices out."));
  return (await res.json()).revoked;
}

export async function updateProfile(patch: { name?: string; username?: string }): Promise<CurrentUser> {
  const res = await json("/auth/me", "PATCH", patch);
  if (res.status === 409) throw new Error("That username is taken. Try another.");
  if (res.status === 422) {
    throw new Error("Check the values: usernames use letters, numbers, - and _ (3 to 32 characters).");
  }
  if (!res.ok) throw new Error(await messageOf(res, "Couldn't save your profile."));
  return res.json();
}

export async function changePassword(current: string | null, next: string): Promise<void> {
  const res = await json("/auth/password", "PUT", { current_password: current || undefined, new_password: next });
  if (res.status === 401) throw new Error("Your current password is incorrect.");
  if (res.status === 422) throw new Error("The new password must be at least 8 characters.");
  if (!res.ok) throw new Error(await messageOf(res, "Couldn't change your password."));
}

export async function requestEmailVerification(): Promise<{ sent: boolean; already_verified?: boolean }> {
  const res = await json("/auth/verify-email/request", "POST");
  if (!res.ok) throw new Error(await messageOf(res, "Couldn't send the verification email."));
  return res.json();
}

/** Save everything the app holds about you (profile and chats) as a JSON file. */
export async function downloadExport(): Promise<void> {
  const res = await apiFetch("/auth/export");
  if (!res.ok) throw new Error(await messageOf(res, "Couldn't export your data."));
  const url = URL.createObjectURL(await res.blob());
  const link = document.createElement("a");
  link.href = url;
  link.download = "pos-export.json";
  link.click();
  URL.revokeObjectURL(url);
}

export async function deleteAccount(confirmEmail: string, password: string): Promise<void> {
  const res = await json("/auth/account/delete", "POST", {
    confirm_email: confirmEmail,
    password: password || undefined,
  });
  if (res.status === 400) throw new Error("That email doesn't match this account.");
  if (res.status === 401) throw new Error("That password is incorrect.");
  if (!res.ok) throw new Error(await messageOf(res, "Couldn't delete the account."));
}

import { API_URL } from "./config";

// Access tokens live 15 minutes; the refresh token lives 30 days. Without this,
// every call would start failing 15 minutes after login even though the user
// is still, for all practical purposes, signed in.
let inFlightRefresh: Promise<boolean> | null = null;

/** Exchange the refresh cookie for a new access token. Refresh tokens rotate
 * and are single-use, and the server treats a replayed one as theft and revokes
 * the whole family, so several requests that 401 together MUST share one call. */
function refreshSession(): Promise<boolean> {
  inFlightRefresh ??= fetch(`${API_URL}/auth/refresh`, { method: "POST", credentials: "include" })
    .then((res) => res.ok)
    .catch(() => false)
    .finally(() => {
      inFlightRefresh = null;
    });
  return inFlightRefresh;
}

/** fetch() against the API with cookies, that survives an expired access
 * token: on a 401 it refreshes once and repeats the request. If the refresh is
 * refused the session is really over, so it sends the user to the sign-in page.
 * Use it for authenticated calls -- not for login/signup, where a 401 is an
 * answer, not an expiry. */
export async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const send = () => fetch(`${API_URL}${path}`, { ...init, credentials: "include" });
  const res = await send();
  if (res.status !== 401) return res;
  if (await refreshSession()) return send();
  if (typeof window !== "undefined" && !window.location.pathname.startsWith("/login")) {
    // Hard navigation on purpose (outside React): it drops all client state along with the dead session.
    // eslint-disable-next-line @next/next/no-location-assign-relative-destination
    window.location.href = "/login";
  }
  return res;
}

// Backend base URLs. In Docker Compose the frontend and backend run in
// separate containers/origins, so these must be configurable at build
// time via env vars rather than assuming same-origin (unlike the old
// static index.html, which was served BY the backend and used
// location.host directly).
export const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

export const WS_URL =
  process.env.NEXT_PUBLIC_WS_URL?.replace(/\/$/, "") || "ws://localhost:8000";

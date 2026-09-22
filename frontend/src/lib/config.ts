// Where the API lives.
//
// Packaged, the backend serves this page too, so the API is same-origin and
// the base URL is empty -- requests go to /chat/stream on whatever host and
// port the app was opened at. In development the UI is served by `next dev`
// on :3000 while the API is on :8000, so it needs the full URL.
//
// NEXT_PUBLIC_API_URL overrides both, for pointing a local UI at a backend
// somewhere else.
const configured = process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "");
const fallback = process.env.NODE_ENV === "production" ? "" : "http://localhost:8000";

export const API_URL = configured ?? fallback;

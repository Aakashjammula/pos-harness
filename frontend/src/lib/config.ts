// Backend base URL. Configurable at build time; defaults to the local dev
// server this project runs against during development. Must match
// next.config.ts's own default -- that's what the CSP's connect-src is
// built from, so a mismatch here gets silently blocked by the browser.
export const API_URL = process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

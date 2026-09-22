import type { NextConfig } from "next";

const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const isProd = process.env.NODE_ENV === "production";

const originOf = (url: string): string => {
  try {
    return new URL(url).origin;
  } catch {
    return "";
  }
};

// What the page may load and talk to: itself plus the backend API (fetch).
// Next.js needs 'unsafe-inline' for its hydration scripts and Tailwind's styles;
// 'unsafe-eval' is added only for `next dev`. frame-ancestors also blocks clickjacking.
const csp = [
  "default-src 'self'",
  `script-src 'self' 'unsafe-inline'${isProd ? "" : " 'unsafe-eval'"}`,
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob:",
  "font-src 'self' data:",
  "media-src 'self' blob:",
  `connect-src 'self' ${originOf(apiUrl)}`.trim(),
  "worker-src 'self' blob:",
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'self'",
  "frame-ancestors 'none'",
].join("; ");

const securityHeaders = [
  { key: "Content-Security-Policy", value: csp },
  { key: "X-Frame-Options", value: "DENY" },
  { key: "X-Content-Type-Options", value: "nosniff" },
  // The magic-link token sits in the URL; nothing here needs a Referer, so send none.
  { key: "Referrer-Policy", value: "no-referrer" },
  // Text-only for now (see the UI/backend replan) -- no microphone/camera/etc. needed.
  { key: "Permissions-Policy", value: "microphone=(), camera=(), geolocation=(), payment=()" },
  { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
  // Browsers ignore this over plain http, so it is harmless in development.
  { key: "Strict-Transport-Security", value: "max-age=31536000; includeSubDomains" },
];

const nextConfig: NextConfig = {
  async headers() {
    return [{ source: "/:path*", headers: securityHeaders }];
  },
};

export default nextConfig;

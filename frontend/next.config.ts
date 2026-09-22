import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Build to plain files the backend can serve, so the packaged app is one
  // process on one port. `headers()` is not supported in this mode -- the
  // CSP and the rest now come from pos/http_headers.py, which is what
  // serves these files.
  output: "export",
};

export default nextConfig;

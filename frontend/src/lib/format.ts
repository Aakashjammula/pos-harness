import type { Latency, Usage } from "./types";

/** 1048576 -> "1M", 200000 -> "200k", 4200 -> "4.2k". */
export function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${+(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${+(n / 1_000).toFixed(1)}k`;
  return String(n);
}

export function formatUsageLine(usage?: Usage, latency?: Latency): string {
  const parts: string[] = [];
  if (latency) parts.push(`${latency.ttft.toFixed(2)}s ttft / ${latency.total.toFixed(2)}s total`);
  if (usage) {
    if (usage.input_tokens != null && usage.output_tokens != null) {
      // No context window here -- the ring beside the composer already
      // tracks how full the window is, and it's a per-chat number, not a
      // per-turn one.
      parts.push(`${usage.input_tokens} in / ${usage.output_tokens} out (${usage.total_tokens} total)`);
    }
    if (usage.cost_usd != null) parts.push(`$${usage.cost_usd.toFixed(4)}`);
  }
  return parts.join(" · ");
}

export const TRACE_RESULT_MAX_CHARS = 200;

export function truncateToolResult(result: unknown): string {
  const str = result != null ? String(result) : "";
  return str.length > TRACE_RESULT_MAX_CHARS ? str.slice(0, TRACE_RESULT_MAX_CHARS) + "…" : str;
}

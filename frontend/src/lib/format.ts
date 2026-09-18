import type { Latency, Usage } from "./types";

export function formatUsageLine(usage?: Usage, latency?: Latency): string {
  const parts: string[] = [];
  if (latency) parts.push(`${latency.ttft.toFixed(2)}s ttft / ${latency.total.toFixed(2)}s total`);
  if (usage) {
    if (usage.input_tokens != null && usage.output_tokens != null) {
      const totalPart =
        usage.context_window != null
          ? `${usage.total_tokens} total / ${usage.context_window} ctx`
          : `${usage.total_tokens} total`;
      parts.push(`${usage.input_tokens} in / ${usage.output_tokens} out (${totalPart})`);
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

export function formatSessionTimestamp(iso: string): string {
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

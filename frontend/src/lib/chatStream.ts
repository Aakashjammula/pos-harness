import { apiFetch } from "./apiFetch";

export interface ChatHandlers {
  onSession: (id: string) => void;
  onToken: (text: string) => void;
  onDone: (d: { text: string; usage?: unknown; latency?: unknown }) => void;
  onTitle: (title: string) => void;
  onError: (message: string) => void;
}

/** Parse one SSE block ("event: x\ndata: {...}") into its name and JSON data.
 * Returns null for comments and blocks with no event/data. */
export function parseSseBlock(block: string): { event: string; data: unknown } | null {
  const event = /^event: (.+)$/m.exec(block)?.[1];
  const data = /^data: (.*)$/m.exec(block)?.[1];
  if (!event || data === undefined) return null;
  try {
    return { event, data: JSON.parse(data) };
  } catch {
    return null; // a malformed block shouldn't kill the whole stream
  }
}

function dispatch(event: string, d: Record<string, unknown>, h: ChatHandlers): void {
  if (event === "session") h.onSession(d.id as string);
  else if (event === "token") h.onToken(d.text as string);
  else if (event === "done") h.onDone(d as { text: string; usage?: unknown; latency?: unknown });
  else if (event === "title") h.onTitle(d.title as string);
  else if (event === "error") h.onError((d.message as string) || "Something went wrong.");
  // "ping" is a keepalive and carries nothing
}

/**
 * POST a message and stream the reply. `EventSource` only supports GET, so this
 * reads the response body directly, like Anthropic's own SDKs do.
 *
 * Problems found before streaming starts arrive as an ordinary HTTP error and
 * are reported through `onError`; failures after that arrive as `error` events.
 * Aborting `signal` stops the request, which makes the server stop generating.
 */
export async function streamChat(
  body: { message: string; session_id?: string | null; provider?: string; llm_model?: string },
  handlers: ChatHandlers,
  signal: AbortSignal
): Promise<void> {
  const res = await apiFetch("/chat/stream", {
    method: "POST",
    signal,
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
  });
  if (!res.ok || !res.body) {
    let detail = `Request failed (HTTP ${res.status}).`;
    try {
      detail = (await res.json()).detail || detail;
    } catch {
      // not JSON: keep the generic message
    }
    handlers.onError(detail);
    return;
  }

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let sep = buffer.indexOf("\n\n");
    while (sep >= 0) {
      const parsed = parseSseBlock(buffer.slice(0, sep));
      buffer = buffer.slice(sep + 2);
      if (parsed) dispatch(parsed.event, parsed.data as Record<string, unknown>, handlers);
      sep = buffer.indexOf("\n\n");
    }
  }
}

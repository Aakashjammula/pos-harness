import { API_URL } from "./config";
import type { ToolCall, Usage } from "./types";

export interface ChatDone {
  text: string;
  model: string | null;
  finish_reason: string | null;
  reasoning_effort: string;
  usage: Usage;
  tool_calls: ToolCall[];
}

export interface ChatHandlers {
  onSession: (id: string) => void;
  onToken: (text: string) => void;
  onActivity: (tool: string, args: Record<string, unknown>) => void;
  onDone: (d: ChatDone) => void;
  onTitle: (title: string) => void;
  onError: (message: string) => void;
}

export interface ChatRequest {
  message: string;
  thread_id: string | null;
  folder: string | null;
  reasoning_effort: string;
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
  else if (event === "activity") h.onActivity(d.tool as string, (d.args as Record<string, unknown>) ?? {});
  else if (event === "done") h.onDone(d as unknown as ChatDone);
  else if (event === "title") h.onTitle(d.title as string);
  else if (event === "error") h.onError((d.message as string) || "Something went wrong.");
}

/**
 * POST a message and stream the reply from POST /chat/stream.
 *
 * Problems found before streaming starts arrive as an ordinary HTTP error and
 * are reported through `onError`; failures after that arrive as `error` events.
 * Aborting `signal` stops the request.
 */
export async function streamChat(body: ChatRequest, handlers: ChatHandlers, signal: AbortSignal): Promise<void> {
  const res = await fetch(`${API_URL}/chat/stream`, {
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

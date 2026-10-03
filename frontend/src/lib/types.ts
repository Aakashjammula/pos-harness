export interface ToolCall {
  name: string;
  args?: Record<string, unknown>;
  result?: unknown;
  round?: number;
  // What this call added to the conversation: the growth in the next round's
  // input. Exact when its round made one call; a share of that growth when it
  // made several, which `cost_estimated` marks.
  cost_tokens?: number | null;
  cost_estimated?: boolean;
  /** Set while a call is in flight, cleared when its result arrives. */
  pending?: boolean;
  /** How much of its arguments has streamed so far -- a whole file, for
   * write_file -- while the model is still writing them. */
  progress_chars?: number;
}

/** One model call within a turn. A turn with tool calls has several. */
export interface Round {
  index: number;
  input_tokens: number;
  output_tokens: number;
  reasoning_tokens: number;
  cache_read: number;
  cache_creation: number;
  delta: number | null; // growth over the previous round; null on the first
  tool_calls: number;
  /** What the model said in this round, before its tool calls ran. */
  text: string;
  /** What it thought before that, for models that report it. Absent on
   * turns recorded before thinking was kept. */
  reasoning?: string;
}

export interface Usage {
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
  context_window?: number;
  cost_usd?: number;
  cache_read?: number;
  cache_creation?: number;
  // A breakdown of output_tokens, not an addition to it: hidden thinking,
  // billed at the output rate.
  reasoning_tokens?: number;
  // The last round's input: how big the conversation actually is. Distinct
  // from input_tokens, which sums every round and so overstates it.
  context_tokens?: number;
  model?: string | null;
  finish_reason?: string | null;
  reasoning_effort?: string;
  tool_calls?: ToolCall[];
  rounds?: Round[];
}

export interface Latency {
  ttft: number;
  total: number;
}

export interface TranscriptLine {
  id: string;
  who: "you" | "bot" | "system" | "error";
  text: string;
  /** Tool calls seen while the reply is still streaming. Replaced by
   * `usage.tool_calls` once the turn finishes. */
  liveCalls?: ToolCall[];
  /** The model's thinking so far, while the reply is still streaming.
   * Replaced by each round's `reasoning` once the turn finishes. */
  liveThinking?: string;
  usage?: Usage;
  latency?: Latency;
}

import type { OptionsResponse } from "./types";

/** True once something can answer: the operator set a server-level provider,
 * or this user saved credentials for any provider except Tavily (web search,
 * which is a tool, not an LLM). */
export function hasLlm(options: OptionsResponse | null, configured: string[]): boolean {
  if (!options) return false;
  return options.llm_configured || configured.some((p) => p !== "tavily");
}

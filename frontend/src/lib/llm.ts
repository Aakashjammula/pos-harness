import type { OptionsResponse } from "./types";

/** True once something can answer: the operator set a server-level provider,
 * or this user saved credentials for any provider except Tavily (web search,
 * which is a tool, not an LLM). */
export function hasLlm(options: OptionsResponse | null, configured: string[]): boolean {
  if (!options) return false;
  return options.llm_configured || configured.some((p) => p !== "tavily");
}

/** The model a session will ask for. With a provider selected the choice must
 * come from that provider's own list: the user's pick if it is in it,
 * otherwise the first listed, and never a stale name left over from another
 * provider. Where nothing can be listed (Azure, or a list that failed) send
 * none and let the provider's own default apply. */
export function effectiveModel(
  provider: string,
  chosen: string,
  listed: { id: string }[],
  listable: boolean
): string {
  if (!provider) return chosen; // no provider selected: the server-level list
  if (!listable || listed.length === 0) return "";
  return listed.some((m) => m.id === chosen) ? chosen : listed[0].id;
}

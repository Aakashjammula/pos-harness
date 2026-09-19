import { useEffect, useState } from "react";
import { fetchTools, type Tool } from "@/lib/tools";

interface Result {
  version: number;
  tools: Tool[];
  error: string | null;
}

/** This user's tools. `version` changes whenever a key or switch changes, which
 * re-fetches. State is only written from the async callbacks; loading and
 * staleness are derived by comparing versions. */
export function useTools(version: number) {
  const [result, setResult] = useState<Result | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchTools()
      .then((tools) => !cancelled && setResult({ version, tools, error: null }))
      .catch((e: Error) => !cancelled && setResult({ version, tools: [], error: e.message }));
    return () => {
      cancelled = true;
    };
  }, [version]);

  const current = result && result.version === version ? result : null;
  return { tools: current?.tools ?? [], loading: current === null, error: current?.error ?? null };
}

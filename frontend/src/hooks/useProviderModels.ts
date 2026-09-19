import { useEffect, useState } from "react";
import { fetchProviderModels, type ProviderModel } from "@/lib/credentials";
import type { KeyProvider } from "@/lib/types";

// Azure deployments are named by the user and can't be listed with a key.
const UNLISTABLE: KeyProvider[] = ["", "azure"];

interface Result {
  key: string;
  models: ProviderModel[];
  error: string | null;
}

/** The models available to the selected provider's saved credential.
 *
 * `version` changes whenever credentials are saved or removed, which
 * re-fetches even when the provider was already configured (a new key can
 * unlock different models). State is only written from the async callbacks,
 * and loading/staleness are derived by comparing request keys. */
export function useProviderModels(provider: KeyProvider, configured: string[], version: number) {
  const active = !UNLISTABLE.includes(provider) && configured.includes(provider);
  const key = active ? `${provider}:${version}` : "";
  const [result, setResult] = useState<Result | null>(null);

  useEffect(() => {
    if (!key) return;
    let cancelled = false;
    fetchProviderModels(provider)
      .then((models) => !cancelled && setResult({ key, models, error: null }))
      .catch((e: Error) => !cancelled && setResult({ key, models: [], error: e.message }));
    return () => {
      cancelled = true;
    };
  }, [key, provider]);

  const current = result && result.key === key ? result : null;
  return {
    listable: !UNLISTABLE.includes(provider),
    models: current?.models ?? [],
    loading: active && current === null,
    error: current?.error ?? null,
  };
}

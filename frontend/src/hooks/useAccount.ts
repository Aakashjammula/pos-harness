import { useEffect, useState } from "react";
import { fetchLoginSessions, type LoginSession } from "@/lib/account";
import { type CurrentUser, fetchMe } from "@/lib/auth";

interface Result<T> {
  version: number;
  data: T;
  error: string | null;
}

/** Load something whenever `version` changes. State is written only from the async
 * callbacks; loading and staleness are derived by comparing versions. */
function useVersioned<T>(load: () => Promise<T>, version: number, empty: T) {
  const [result, setResult] = useState<Result<T> | null>(null);

  useEffect(() => {
    let cancelled = false;
    load()
      .then((data) => !cancelled && setResult({ version, data, error: null }))
      .catch((e: Error) => !cancelled && setResult({ version, data: empty, error: e.message }));
    return () => {
      cancelled = true;
    };
    // `load` and `empty` are module-level constants at the call sites; `version` is the trigger.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [version]);

  const current = result && result.version === version ? result : null;
  return { data: current?.data ?? empty, loading: current === null, error: current?.error ?? null };
}

const NO_SESSIONS: LoginSession[] = [];
const loadMe = () => fetchMe();

/** The signed-in user's full profile. */
export function useMe(version: number) {
  const { data, loading, error } = useVersioned<CurrentUser | null>(loadMe, version, null);
  return { me: data, loading, error };
}

/** The signed-in user's own login sessions (devices). */
export function useLoginSessions(version: number) {
  const { data, loading, error } = useVersioned<LoginSession[]>(fetchLoginSessions, version, NO_SESSIONS);
  return { sessions: data, loading, error };
}

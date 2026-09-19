"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { type CurrentUser, fetchMe } from "@/lib/auth";

export function AuthGuard({ children }: { children: (user: CurrentUser) => React.ReactNode }) {
  const router = useRouter();
  const [user, setUser] = useState<CurrentUser | null>(null);
  const [checked, setChecked] = useState(false);
  const mounted = useRef(true);

  const check = useCallback(() => {
    fetchMe()
      .then((me) => {
        if (!mounted.current) return;
        if (me === null) router.replace("/login");
        else setUser(me);
      })
      .catch(() => mounted.current && router.replace("/login"))
      .finally(() => mounted.current && setChecked(true));
  }, [router]);

  useEffect(() => {
    mounted.current = true;
    check();
    // Back/forward can restore this page from the browser's cache without re-running
    // anything, so someone who signed out could see the app again. Re-check when that happens.
    const onPageShow = (e: PageTransitionEvent) => {
      if (e.persisted) check();
    };
    window.addEventListener("pageshow", onPageShow);
    return () => {
      mounted.current = false;
      window.removeEventListener("pageshow", onPageShow);
    };
  }, [check]);

  if (!checked || !user) {
    return <div className="flex min-h-screen items-center justify-center bg-bg text-[13px] text-text-faint">Loading…</div>;
  }
  return <>{children(user)}</>;
}

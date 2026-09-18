"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { type CurrentUser, fetchMe } from "@/lib/auth";

export function AuthGuard({ children }: { children: (user: CurrentUser) => React.ReactNode }) {
  const router = useRouter();
  const [user, setUser] = useState<CurrentUser | null>(null);
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    let active = true;
    fetchMe()
      .then((me) => {
        if (!active) return;
        if (me === null) router.replace("/login");
        else setUser(me);
      })
      .catch(() => router.replace("/login"))
      .finally(() => active && setChecked(true));
    return () => {
      active = false;
    };
  }, [router]);

  if (!checked || !user) {
    return <div className="flex min-h-screen items-center justify-center bg-bg text-[13px] text-text-faint">Loading…</div>;
  }
  return <>{children(user)}</>;
}

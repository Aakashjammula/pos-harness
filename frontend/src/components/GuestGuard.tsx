"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { fetchMe } from "@/lib/auth";

/** For pages only signed-out people should see (sign in, sign up). Someone who is
 * already signed in is sent to the app instead of being shown a login form. */
export function GuestGuard({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const [isGuest, setIsGuest] = useState(false);

  useEffect(() => {
    let active = true;
    fetchMe()
      .then((me) => {
        if (!active) return;
        if (me) router.replace("/");
        else setIsGuest(true);
      })
      .catch(() => active && setIsGuest(true)); // can't tell (e.g. offline): show the form
    return () => {
      active = false;
    };
  }, [router]);

  if (!isGuest) {
    return <div className="flex min-h-screen items-center justify-center bg-bg text-[13px] text-text-faint">Loading…</div>;
  }
  return <>{children}</>;
}

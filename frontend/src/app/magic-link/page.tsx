"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import Link from "next/link";
import { verifyMagicLink } from "@/lib/auth";
import { AuthBackdrop, authDarkVars, BrandWordmark } from "@/components/AuthTheme";

function MagicLinkVerifier() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const token = searchParams.get("token");
  const [error, setError] = useState<string | null>(
    token ? null : "This link is missing its token."
  );

  useEffect(() => {
    if (!token) return;
    verifyMagicLink(token)
      .then(() => router.replace("/"))
      .catch((err) => setError(err instanceof Error ? err.message : "Something went wrong."));
  }, [token, router]);

  return (
    <main
      style={authDarkVars}
      className="relative flex min-h-screen flex-col items-center justify-center gap-9 bg-bg px-4"
    >
      <AuthBackdrop />
      <div className="relative z-10">
        <BrandWordmark />
      </div>
      <div className="relative z-10 w-full max-w-[440px] rounded-2xl border border-border bg-bg p-9 pt-8 text-center shadow-[var(--shadow)]">
        {error ? (
          <>
            <h1 className="mb-2 font-mono text-[24px] font-semibold tracking-tight text-text">
              Link didn&apos;t work
            </h1>
            <p className="mb-7 text-[13.5px] leading-relaxed text-text-muted">{error}</p>
            <Link
              href="/login"
              className="inline-block w-full rounded-lg bg-accent px-3 py-2.5 text-[13.5px] font-semibold text-white hover:bg-accent-hover"
            >
              Back to sign in
            </Link>
          </>
        ) : (
          <p className="text-[13.5px] text-text-muted">Signing you in…</p>
        )}
      </div>
    </main>
  );
}

export default function MagicLinkPage() {
  return (
    <Suspense>
      <MagicLinkVerifier />
    </Suspense>
  );
}

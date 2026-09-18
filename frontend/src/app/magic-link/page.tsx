"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import Link from "next/link";
import { verifyMagicLink } from "@/lib/auth";
import { BrandMark } from "@/components/icons";

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
    <main className="flex min-h-screen flex-col items-center justify-center gap-7 bg-bg px-4">
      <div className="flex items-center gap-2.5">
        <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-accent">
          <BrandMark className="h-4 w-4" />
        </span>
        <span className="text-[15px] font-semibold tracking-tight text-text">POS</span>
      </div>
      <div className="w-full max-w-[380px] rounded-2xl border border-border bg-bg p-8 pt-7 text-center shadow-[var(--shadow)]">
        {error ? (
          <>
            <h1 className="mb-1.5 text-[20px] font-semibold tracking-tight text-text">
              Link didn&apos;t work
            </h1>
            <p className="mb-6 text-[13px] leading-relaxed text-text-muted">{error}</p>
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

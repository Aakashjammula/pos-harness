"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import Link from "next/link";
import { verifyMagicLink } from "@/lib/auth";
import { AuthHeading, AuthShell } from "@/components/AuthShell";

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
    <AuthShell>
      {error ? (
        <>
          <AuthHeading title="Link didn't work" subtitle={error} />
          <Link
            href="/login"
            className="block w-full rounded-[0.6em] bg-accent px-[1em] py-[0.85em] text-center text-[1em] font-semibold text-white transition-colors hover:bg-accent-hover"
          >
            Back to sign in
          </Link>
        </>
      ) : (
        <AuthHeading title="Signing you in" subtitle="Hold on, this takes a moment." />
      )}
    </AuthShell>
  );
}

export default function MagicLinkPage() {
  return (
    <Suspense>
      <MagicLinkVerifier />
    </Suspense>
  );
}

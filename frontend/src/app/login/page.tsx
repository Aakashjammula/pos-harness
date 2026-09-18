"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import Link from "next/link";
import { login, requestMagicLink } from "@/lib/auth";
import { authDarkVars, BrandWordmark } from "@/components/AuthTheme";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [linkSentTo, setLinkSentTo] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (password) {
        await login(email, password);
        router.replace("/");
      } else {
        await requestMagicLink(email);
        setLinkSentTo(email);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main
      style={authDarkVars}
      className="flex min-h-screen flex-col items-center justify-center gap-8 bg-bg px-4 py-10"
    >
      <BrandWordmark />

      <div className="w-full max-w-[380px] rounded-2xl border border-border bg-bg p-8 pt-7 shadow-[var(--shadow)]">
        {linkSentTo ? (
          <>
            <h1 className="mb-1.5 text-[19px] font-semibold tracking-tight text-text text-balance">
              Check your email
            </h1>
            <p className="mb-6 text-[13px] leading-relaxed text-text-muted">
              Sent a sign-in link to <span className="font-medium text-text">{linkSentTo}</span>. It
              works once and expires in 15 minutes.
            </p>
            <button
              type="button"
              onClick={() => setLinkSentTo(null)}
              className="w-full rounded-lg bg-surface-sunken px-3 py-2.5 text-[13px] font-medium text-text hover:bg-border"
            >
              Use a different email
            </button>
          </>
        ) : (
          <>
            <h1 className="mb-1.5 text-[19px] font-semibold tracking-tight text-text text-balance">
              Sign in
            </h1>
            <p className="mb-6 text-[13px] leading-relaxed text-text-muted">
              Enter your email — leave the password blank to sign in with a link instead.
            </p>

            <form onSubmit={onSubmit} className="grid gap-4">
              <div className="grid gap-1.5">
                <label htmlFor="email" className="text-[12.5px] font-medium text-text-muted">
                  Email
                </label>
                <input
                  id="email"
                  type="email"
                  required
                  autoComplete="email"
                  autoFocus
                  placeholder="you@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full rounded-lg border border-transparent bg-surface-sunken px-3 py-2.5 text-[13.5px] text-text outline-none transition-[border-color,box-shadow] focus:border-accent focus:shadow-[0_0_0_3px_var(--accent-tint)]"
                />
              </div>
              <div className="grid gap-1.5">
                <label htmlFor="password" className="text-[12.5px] font-medium text-text-muted">
                  Password <span className="text-text-faint">(optional)</span>
                </label>
                <input
                  id="password"
                  type="password"
                  autoComplete="current-password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full rounded-lg border border-transparent bg-surface-sunken px-3 py-2.5 text-[13.5px] text-text outline-none transition-[border-color,box-shadow] focus:border-accent focus:shadow-[0_0_0_3px_var(--accent-tint)]"
                />
              </div>

              {error && (
                <div className="rounded-lg bg-danger-tint px-3 py-2.5 text-[12.5px] leading-relaxed text-danger">
                  {error}
                </div>
              )}

              <button
                type="submit"
                disabled={busy}
                className="w-full rounded-lg bg-accent px-3 py-2.5 text-[13.5px] font-semibold text-white hover:bg-accent-hover disabled:opacity-60"
              >
                {busy ? "One sec…" : password ? "Sign in" : "Send sign-in link"}
              </button>
            </form>

            <p className="mt-5 text-center text-[12.5px] text-text-faint">
              No account? <Link href="/signup" className="text-accent hover:underline">Create one</Link>
            </p>
          </>
        )}
      </div>
    </main>
  );
}

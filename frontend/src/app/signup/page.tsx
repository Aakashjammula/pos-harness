"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import Link from "next/link";
import { requestMagicLink, signup } from "@/lib/auth";
import { AuthBackdrop, authDarkVars, BrandWordmark } from "@/components/AuthTheme";

const EXAMPLE_SESSIONS = [
  { title: "Refactoring the export pipeline", meta: "text · 14 turns" },
  { title: "Voice memo → follow-ups", meta: "voice · 6 turns" },
];

function ProductPreview() {
  return (
    <div className="flex h-full flex-col justify-between p-10">
      <BrandWordmark />

      <div className="grid gap-5">
        <div className="grid gap-1 rounded-xl border border-border bg-surface-sunken/60 p-2">
          {EXAMPLE_SESSIONS.map((s) => (
            <div key={s.title} className="rounded-lg px-2.5 py-2">
              <div className="truncate text-[12.5px] text-text">{s.title}</div>
              <div className="text-[11px] text-text-faint">{s.meta}</div>
            </div>
          ))}
        </div>

        <div className="grid gap-2 rounded-xl border border-border bg-surface-sunken/60 p-3.5">
          <div className="ml-auto max-w-[85%] rounded-lg bg-accent px-3 py-2 text-[12.5px] text-white">
            Switch this thread to claude-opus-5, keep everything else
          </div>
          <div className="max-w-[85%] rounded-lg bg-user-bubble px-3 py-2 text-[12.5px] text-text">
            Switched to claude-opus-5 (cloud). Your last 14 turns carried over.
          </div>
        </div>
        <p className="text-[11.5px] text-text-faint">An example POS session, shortened.</p>
      </div>

      <p className="max-w-[280px] text-[13px] leading-relaxed text-text-muted">
        Voice or text. Local or cloud models. One agent that remembers the whole thread either way.
      </p>
    </div>
  );
}

export default function SignupPage() {
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
        await signup(email, password);
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
    <main style={authDarkVars} className="flex min-h-screen bg-bg">
      <div className="hidden w-[440px] shrink-0 border-r border-border bg-sidebar-bg lg:flex">
        <ProductPreview />
      </div>

      <div className="relative flex flex-1 flex-col items-center justify-center gap-9 px-4 py-12">
        <AuthBackdrop />
        <div className="relative z-10 lg:hidden">
          <BrandWordmark />
        </div>

        <div className="relative z-10 w-full max-w-[420px]">
          {linkSentTo ? (
            <>
              <h1 className="mb-2 font-mono text-[24px] font-semibold tracking-tight text-text text-balance">
                Check your email
              </h1>
              <p className="mb-7 text-[13.5px] leading-relaxed text-text-muted">
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
              <h1 className="mb-2 font-mono text-[24px] font-semibold tracking-tight text-text text-balance">
                Create your account
              </h1>
              <p className="mb-7 text-[13.5px] leading-relaxed text-text-muted">
                Set a password, or leave it blank and we&apos;ll send you a sign-in link instead.
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
                    Password <span className="text-text-faint">(optional, 8+ characters)</span>
                  </label>
                  <input
                    id="password"
                    type="password"
                    autoComplete="new-password"
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
                  {busy ? "One sec…" : password ? "Create account" : "Send sign-in link"}
                </button>
              </form>

              <p className="mt-5 text-center text-[12.5px] text-text-faint">
                Already have an account? <Link href="/login" className="text-accent hover:underline">Sign in</Link>
              </p>
            </>
          )}
        </div>
      </div>
    </main>
  );
}

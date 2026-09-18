"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { login, requestMagicLink } from "@/lib/auth";
import { BrandMark } from "@/components/icons";

type ViewState = "email" | "sent" | "password";

export default function LoginPage() {
  const router = useRouter();
  const [view, setView] = useState<ViewState>("email");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function onSendLink(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await requestMagicLink(email);
      setView("sent");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }

  async function onPasswordSignIn(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await login(email, password);
      router.replace("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-7 bg-bg px-4 py-10">
      <div className="flex items-center gap-2.5">
        <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-accent">
          <BrandMark className="h-4 w-4" />
        </span>
        <span className="text-[15px] font-semibold tracking-tight text-text">POS</span>
      </div>

      <div className="w-full max-w-[380px] rounded-2xl border border-border bg-bg p-8 pt-7 shadow-[var(--shadow)]">
        {view === "sent" ? (
          <>
            <h1 className="mb-1.5 text-[20px] font-semibold tracking-tight text-text text-balance">
              Check your email
            </h1>
            <p className="mb-6 text-[13px] leading-relaxed text-text-muted">
              We sent a sign-in link to <span className="font-medium text-text">{email}</span>. Click it
              to continue — it works once and expires in 15 minutes.
            </p>
            <button
              type="button"
              onClick={() => {
                setView("email");
                setError(null);
              }}
              className="w-full rounded-lg bg-surface-sunken px-3 py-2.5 text-[13px] font-medium text-text hover:bg-border"
            >
              Use a different email
            </button>
          </>
        ) : (
          <>
            <h1 className="mb-1.5 text-[20px] font-semibold tracking-tight text-text text-balance">
              Sign in to POS
            </h1>
            <p className="mb-6 text-[13px] leading-relaxed text-text-muted">
              Voice, text, and local or cloud models in one agent. Enter your email and we&apos;ll send
              you a link — no password needed.
            </p>

            <form onSubmit={view === "password" ? onPasswordSignIn : onSendLink} className="grid gap-4">
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

              {view === "password" && (
                <div className="grid gap-1.5">
                  <label htmlFor="password" className="text-[12.5px] font-medium text-text-muted">
                    Password
                  </label>
                  <input
                    id="password"
                    type="password"
                    required
                    autoComplete="current-password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="w-full rounded-lg border border-transparent bg-surface-sunken px-3 py-2.5 text-[13.5px] text-text outline-none transition-[border-color,box-shadow] focus:border-accent focus:shadow-[0_0_0_3px_var(--accent-tint)]"
                  />
                </div>
              )}

              {error && (
                <div className="flex items-start gap-2 rounded-lg bg-danger-tint px-3 py-2.5 text-[12.5px] leading-relaxed text-danger">
                  {error}
                </div>
              )}

              <button
                type="submit"
                disabled={busy}
                className="w-full rounded-lg bg-accent px-3 py-2.5 text-[13.5px] font-semibold text-white hover:bg-accent-hover disabled:opacity-60"
              >
                {busy
                  ? view === "password" ? "Signing in…" : "Sending link…"
                  : view === "password" ? "Sign in" : "Continue with email"}
              </button>
            </form>

            <button
              type="button"
              onClick={() => {
                setView(view === "password" ? "email" : "password");
                setError(null);
              }}
              className="mt-4 w-full text-center text-[12.5px] font-medium text-text-faint hover:text-text-muted"
            >
              {view === "password" ? "Use an email link instead" : "Use a password instead"}
            </button>
          </>
        )}
      </div>
    </main>
  );
}

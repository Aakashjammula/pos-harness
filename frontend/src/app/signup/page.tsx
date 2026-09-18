"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import Link from "next/link";
import { signup } from "@/lib/auth";

export default function SignupPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await signup(email, password);
      router.replace("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center bg-bg px-4">
      <form onSubmit={onSubmit} className="w-full max-w-sm">
        <h1 className="mb-6 text-xl font-semibold text-text">Create account</h1>
        <label className="mb-1 block text-[12.5px] text-text-muted" htmlFor="email">Email</label>
        <input
          id="email" type="email" required autoComplete="email"
          value={email} onChange={(e) => setEmail(e.target.value)}
          className="mb-4 w-full rounded-lg border border-transparent bg-surface-sunken px-2.5 py-2 text-[13px] text-text outline-none focus:border-accent"
        />
        <label className="mb-1 block text-[12.5px] text-text-muted" htmlFor="password">Password</label>
        <input
          id="password" type="password" required autoComplete="new-password"
          value={password} onChange={(e) => setPassword(e.target.value)}
          className="mb-4 w-full rounded-lg border border-transparent bg-surface-sunken px-2.5 py-2 text-[13px] text-text outline-none focus:border-accent"
        />
        {error && <p className="mb-3 text-[12.5px] text-danger">{error}</p>}
        <button
          type="submit" disabled={busy}
          className="w-full rounded-lg bg-accent px-3 py-2 text-[13px] font-medium text-white hover:bg-accent-hover disabled:opacity-50"
        >
          {busy ? "Creating…" : "Create account"}
        </button>
        <p className="mt-4 text-[12.5px] text-text-faint">
          Already have an account? <Link href="/login" className="text-accent">Sign in</Link>
        </p>
      </form>
    </main>
  );
}

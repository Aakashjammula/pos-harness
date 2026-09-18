"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { login, requestMagicLink, signup } from "@/lib/auth";
import {
  AuthHeading,
  AuthShell,
  ErrorNote,
  Field,
  PasswordField,
  PrimaryButton,
  SecondaryButton,
} from "./AuthShell";

type Mode = "login" | "signup";

const COPY = {
  login: {
    title: "Sign in",
    subtitle: "Welcome back. Use your password, or leave it blank for an email link.",
    submit: "Sign in",
    submitLink: "Send magic link",
    switchPrompt: "No account?",
    switchLabel: "Create one",
    switchHref: "/signup",
    autocomplete: "current-password",
  },
  signup: {
    title: "Create your account",
    subtitle: "Choose a password, or skip it and use an email link every time.",
    submit: "Create account",
    submitLink: "Create account & send link",
    switchPrompt: "Already have an account?",
    switchLabel: "Sign in",
    switchHref: "/login",
    autocomplete: "new-password",
  },
} as const;

export function AuthForm({ mode }: { mode: Mode }) {
  const copy = COPY[mode];
  const router = useRouter();
  const [name, setName] = useState("");
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [linkSentTo, setLinkSentTo] = useState<string | null>(null);

  // The password field IS the method switch: typing in it means "use a
  // password", leaving it empty means "email me a link". One less
  // control to understand than the tabs this replaced, and the button
  // label below always says which one is about to happen.
  const usingPassword = password.length > 0;

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (mode === "signup") {
        const result = await signup({
          name,
          username,
          email,
          password: usingPassword ? password : undefined,
        });
        // A passwordless signup creates the account but no session --
        // the link in their inbox is what signs them in.
        if (result.magic_link_sent) setLinkSentTo(email);
        else router.replace("/");
      } else if (usingPassword) {
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

  if (linkSentTo) {
    return (
      <AuthShell>
        <AuthHeading
          title="Check your email"
          subtitle={`We sent a sign-in link to ${linkSentTo}. It works once and expires in 15 minutes.`}
        />
        <SecondaryButton type="button" onClick={() => setLinkSentTo(null)}>
          Use a different email
        </SecondaryButton>
      </AuthShell>
    );
  }

  return (
    <AuthShell>
      <AuthHeading title={copy.title} subtitle={copy.subtitle} />

      <form onSubmit={onSubmit} className="grid gap-[1.1em]">
        {mode === "signup" && (
          <>
            <Field
              id="name"
              label="Name"
              required
              autoComplete="name"
              autoFocus
              placeholder="Ada Lovelace"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
            <Field
              id="username"
              label="Username"
              required
              minLength={3}
              maxLength={32}
              pattern="[A-Za-z0-9_\-]+"
              autoComplete="username"
              placeholder="ada"
              title="Letters, numbers, hyphens and underscores. At least 3 characters."
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />
          </>
        )}

        <Field
          id="email"
          label="Email"
          type="email"
          required
          autoComplete="email"
          autoFocus={mode === "login"}
          placeholder="you@example.com"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />

        <PasswordField
          id="password"
          label="Password (optional)"
          minLength={8}
          autoComplete={copy.autocomplete}
          shown={showPassword}
          onToggle={() => setShowPassword((s) => !s)}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        {error && <ErrorNote>{error}</ErrorNote>}

        <PrimaryButton type="submit" disabled={busy}>
          {busy ? "One sec…" : usingPassword ? copy.submit : copy.submitLink}
        </PrimaryButton>

        <p className="text-center text-[0.85em] leading-relaxed text-text-faint">
          {usingPassword
            ? "Leave the password blank to get a sign-in link by email instead."
            : "Type a password to set one, or leave it blank and we'll email you a link."}
        </p>
      </form>

      <p className="mt-[1.4em] text-center text-[0.9em] text-text-muted">
        {copy.switchPrompt}{" "}
        <Link href={copy.switchHref} className="font-medium text-accent hover:underline">
          {copy.switchLabel}
        </Link>
      </p>
    </AuthShell>
  );
}

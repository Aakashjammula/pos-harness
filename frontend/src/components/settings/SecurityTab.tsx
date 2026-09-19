"use client";

import { useState } from "react";
import {
  changePassword,
  type LoginSession,
  revokeLoginSession,
  revokeOtherLoginSessions,
} from "@/lib/account";
import { describeDevice, timeAgo } from "@/lib/device";
import { useLoginSessions, useMe } from "@/hooks/useAccount";
import { Notice, PrimaryButton, QuietButton, Section, TextField } from "./ui";

type Feedback = { kind: "error" | "ok"; text: string } | null;

function PasswordSection({ hasPassword, onChanged }: { hasPassword: boolean; onChanged: () => void }) {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState<Feedback>(null);
  const mismatch = confirm.length > 0 && next !== confirm;
  const ready = next.length >= 8 && next === confirm && (!hasPassword || current.length > 0);

  async function save() {
    setBusy(true);
    setFeedback(null);
    try {
      await changePassword(hasPassword ? current : null, next);
      setCurrent("");
      setNext("");
      setConfirm("");
      onChanged();
      setFeedback({ kind: "ok", text: "Password saved. Your other devices were signed out." });
    } catch (e) {
      setFeedback({ kind: "error", text: (e as Error).message });
    } finally {
      setBusy(false);
    }
  }

  return (
    <Section
      title={hasPassword ? "Change password" : "Set a password"}
      description={
        hasPassword
          ? "Changing it signs out every other device."
          : "You sign in with an emailed link. Set a password if you would rather type one."
      }
    >
      {hasPassword && (
        <TextField
          label="Current password"
          id="pw-current"
          type="password"
          autoComplete="current-password"
          value={current}
          onChange={(e) => setCurrent(e.target.value)}
        />
      )}
      <TextField
        label="New password"
        id="pw-new"
        type="password"
        autoComplete="new-password"
        hint="At least 8 characters."
        value={next}
        onChange={(e) => setNext(e.target.value)}
      />
      <TextField
        label="Confirm new password"
        id="pw-confirm"
        type="password"
        autoComplete="new-password"
        value={confirm}
        onChange={(e) => setConfirm(e.target.value)}
      />
      {mismatch && <Notice kind="error">The two passwords don&apos;t match.</Notice>}
      <div>
        <PrimaryButton disabled={busy || !ready} onClick={save}>
          {hasPassword ? "Change password" : "Set password"}
        </PrimaryButton>
      </div>
      {feedback && <Notice kind={feedback.kind}>{feedback.text}</Notice>}
    </Section>
  );
}

function SessionRow({
  session,
  busy,
  onRevoke,
}: {
  session: LoginSession;
  busy: boolean;
  onRevoke: (id: string) => void;
}) {
  return (
    <li className="flex items-center justify-between gap-3 rounded-lg bg-surface-sunken px-3 py-2.5">
      <div className="grid min-w-0 gap-0.5">
        <div className="flex items-center gap-2 text-[13px] text-text">
          <span className="truncate font-medium">{describeDevice(session.user_agent)}</span>
          {session.current && <span className="text-[11.5px] font-medium text-accent">This device</span>}
        </div>
        <div className="truncate text-[11.5px] text-text-faint">
          {session.ip ? `${session.ip} · ` : ""}active {timeAgo(session.last_seen_at)} · signed in{" "}
          {new Date(session.created_at).toLocaleDateString()}
        </div>
      </div>
      {!session.current && (
        <QuietButton disabled={busy} onClick={() => onRevoke(session.id)}>
          Sign out
        </QuietButton>
      )}
    </li>
  );
}

function SessionsSection({ version, onChanged }: { version: number; onChanged: () => void }) {
  const { sessions, loading, error } = useLoginSessions(version);
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState<Feedback>(null);
  const others = sessions.filter((s) => !s.current).length;

  async function act(action: () => Promise<string>) {
    setBusy(true);
    setFeedback(null);
    try {
      setFeedback({ kind: "ok", text: await action() });
      onChanged();
    } catch (e) {
      setFeedback({ kind: "error", text: (e as Error).message });
    } finally {
      setBusy(false);
    }
  }

  return (
    <Section
      title="Where you're signed in"
      description="Every browser or device signed in to your account. Only you can see this list. Sign out any you don't recognise."
    >
      {loading && <div className="text-[12.5px] text-text-faint">Loading…</div>}
      {error && <Notice kind="error">Couldn&apos;t load your sessions: {error}</Notice>}
      <ul className="m-0 grid list-none gap-2 p-0">
        {sessions.map((s) => (
          <SessionRow
            key={s.id}
            session={s}
            busy={busy}
            onRevoke={(id) =>
              act(async () => {
                await revokeLoginSession(id);
                return "Signed that device out.";
              })
            }
          />
        ))}
      </ul>
      <div>
        <QuietButton
          disabled={busy || others === 0}
          onClick={() =>
            act(async () => {
              const n = await revokeOtherLoginSessions();
              return `Signed out ${n} other device${n === 1 ? "" : "s"}.`;
            })
          }
        >
          Sign out all other devices
        </QuietButton>
      </div>
      {feedback && <Notice kind={feedback.kind}>{feedback.text}</Notice>}
    </Section>
  );
}

export function SecurityTab() {
  const [version, setVersion] = useState(0);
  const { me, loading } = useMe(version);
  const refresh = () => setVersion((v) => v + 1); // one counter: a password change also ends other sessions
  if (loading || !me) return <div className="text-[12.5px] text-text-faint">Loading…</div>;
  return (
    <div className="grid gap-3.5">
      <PasswordSection hasPassword={!!me.has_password} onChanged={refresh} />
      <SessionsSection version={version} onChanged={refresh} />
    </div>
  );
}

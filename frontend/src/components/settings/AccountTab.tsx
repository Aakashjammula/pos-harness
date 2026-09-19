"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { deleteAccount, downloadExport, requestEmailVerification, updateProfile } from "@/lib/account";
import { useMe } from "@/hooks/useAccount";
import { DangerButton, Notice, PrimaryButton, QuietButton, Section, TextField } from "./ui";

type Feedback = { kind: "error" | "ok"; text: string } | null;

/** Run one action, keep its outcome as feedback for its own section. */
function useAction() {
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState<Feedback>(null);
  async function run(action: () => Promise<string | void>) {
    setBusy(true);
    setFeedback(null);
    try {
      const done = await action();
      if (done) setFeedback({ kind: "ok", text: done });
    } catch (e) {
      setFeedback({ kind: "error", text: (e as Error).message });
    } finally {
      setBusy(false);
    }
  }
  return { busy, feedback, run };
}

function ProfileSection({ name, username, onSaved }: { name: string; username: string; onSaved: () => void }) {
  // Edits are kept separately from what the server has; null means "untouched".
  const [draftName, setDraftName] = useState<string | null>(null);
  const [draftUsername, setDraftUsername] = useState<string | null>(null);
  const { busy, feedback, run } = useAction();
  const shownName = draftName ?? name;
  const shownUsername = draftUsername ?? username;
  const changed = shownName.trim() !== name || shownUsername.trim() !== username;

  return (
    <Section title="Profile" description="How you appear in the app.">
      <TextField
        label="Name"
        id="profile-name"
        value={shownName}
        maxLength={80}
        onChange={(e) => setDraftName(e.target.value)}
      />
      <TextField
        label="Username"
        id="profile-username"
        value={shownUsername}
        maxLength={32}
        hint="Letters, numbers, - and _ (3 to 32 characters)."
        onChange={(e) => setDraftUsername(e.target.value)}
      />
      <div className="flex items-center gap-2">
        <PrimaryButton
          disabled={busy || !changed}
          onClick={() =>
            run(async () => {
              await updateProfile({
                name: shownName.trim() !== name ? shownName.trim() : undefined,
                username: shownUsername.trim() !== username ? shownUsername.trim() : undefined,
              });
              setDraftName(null);
              setDraftUsername(null);
              onSaved();
              return "Saved.";
            })
          }
        >
          Save changes
        </PrimaryButton>
      </div>
      {feedback && <Notice kind={feedback.kind}>{feedback.text}</Notice>}
    </Section>
  );
}

function EmailSection({ email, verified }: { email: string; verified: boolean }) {
  const { busy, feedback, run } = useAction();
  return (
    <Section
      title="Email"
      description="Your sign-in address. Verifying it proves the address is yours; until then anyone who knew it could have set up an account with it."
    >
      <div className="flex items-center gap-2 text-[13px] text-text">
        <span>{email}</span>
        <span className={`text-[11.5px] font-medium ${verified ? "text-accent" : "text-text-faint"}`}>
          {verified ? "Verified" : "Not verified"}
        </span>
      </div>
      {!verified && (
        <div className="grid gap-2">
          <div>
            <PrimaryButton
              disabled={busy}
              onClick={() =>
                run(async () => {
                  const r = await requestEmailVerification();
                  return r.sent
                    ? `We sent a link to ${email}. Open it in this browser to verify (opening it somewhere else resets the account).`
                    : "Couldn't send the email. Try again in a minute.";
                })
              }
            >
              Send verification email
            </PrimaryButton>
          </div>
          {feedback && <Notice kind={feedback.kind}>{feedback.text}</Notice>}
        </div>
      )}
    </Section>
  );
}

function DataSection() {
  const { busy, feedback, run } = useAction();
  return (
    <Section
      title="Your data"
      description="A copy of your profile and every chat, as a JSON file. It never includes your password or saved API keys."
    >
      <div>
        <QuietButton disabled={busy} onClick={() => run(async () => void (await downloadExport()))}>
          Download my data
        </QuietButton>
      </div>
      {feedback && <Notice kind={feedback.kind}>{feedback.text}</Notice>}
    </Section>
  );
}

function DangerSection({ email, hasPassword }: { email: string; hasPassword: boolean }) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [typedEmail, setTypedEmail] = useState("");
  const [password, setPassword] = useState("");
  const { busy, feedback, run } = useAction();
  const ready = typedEmail.trim().toLowerCase() === email.toLowerCase() && (!hasPassword || password.length > 0);

  return (
    <Section
      danger
      title="Delete account"
      description="Permanently removes your account, chats, saved keys and tool settings. This cannot be undone."
    >
      {!open ? (
        <div>
          <QuietButton onClick={() => setOpen(true)}>Delete my account…</QuietButton>
        </div>
      ) : (
        <div className="grid gap-2.5">
          <TextField
            label={`Type ${email} to confirm`}
            id="delete-email"
            autoComplete="off"
            value={typedEmail}
            onChange={(e) => setTypedEmail(e.target.value)}
          />
          {hasPassword && (
            <TextField
              label="Your password"
              id="delete-password"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          )}
          <div className="flex items-center gap-2">
            <DangerButton
              disabled={busy || !ready}
              onClick={() =>
                run(async () => {
                  await deleteAccount(typedEmail.trim(), password);
                  router.replace("/login");
                })
              }
            >
              Delete my account permanently
            </DangerButton>
            <QuietButton
              disabled={busy}
              onClick={() => {
                setOpen(false);
                setTypedEmail("");
                setPassword("");
              }}
            >
              Cancel
            </QuietButton>
          </div>
        </div>
      )}
      {feedback && <Notice kind={feedback.kind}>{feedback.text}</Notice>}
    </Section>
  );
}

export function AccountTab() {
  const [version, setVersion] = useState(0);
  const { me, loading, error } = useMe(version);

  if (loading) return <div className="text-[12.5px] text-text-faint">Loading…</div>;
  if (error || !me) return <Notice kind="error">Couldn&apos;t load your account{error ? `: ${error}` : "."}</Notice>;

  return (
    <div className="grid gap-3.5">
      <ProfileSection
        name={me.name ?? ""}
        username={me.username ?? ""}
        onSaved={() => setVersion((v) => v + 1)}
      />
      <EmailSection email={me.email} verified={!!me.email_verified} />
      <DataSection />
      <DangerSection email={me.email} hasPassword={!!me.has_password} />
    </div>
  );
}

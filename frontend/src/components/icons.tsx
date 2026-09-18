import type { ConnState } from "@/lib/types";

export function MicIcon({ className, dimmed }: { className?: string; dimmed?: boolean }) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" className={className}>
      <path
        opacity={dimmed ? 0.5 : 1}
        d="M12 15a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3Zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-2.08A7 7 0 0 0 19 12h-2Z"
      />
    </svg>
  );
}

export function SpeakingIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" className={className}>
      <rect x="4" y="10" width="2.5" height="4" rx="1" />
      <rect x="9" y="6" width="2.5" height="12" rx="1" />
      <rect x="14" y="3" width="2.5" height="18" rx="1" />
      <rect x="19" y="8" width="2.5" height="8" rx="1" />
    </svg>
  );
}

export function MutedIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" className={className}>
      <path d="M16.5 12a4.5 4.5 0 0 0-.2-1.3l-1.5 1.5v-.2H12.8L16.5 15.7A4.48 4.48 0 0 0 16.5 12ZM19 12a7 7 0 0 1-.68 3l1.45 1.45A8.96 8.96 0 0 0 21 12h-2ZM4.27 3 3 4.27l6 6V12a3 3 0 0 0 4.55 2.57l1.2 1.2A4.98 4.98 0 0 1 7 12H5a7 7 0 0 0 6 6.92V21h2v-2.08a6.96 6.96 0 0 0 2.61-.88L19.73 22 21 20.73 4.27 3ZM15 6a3 3 0 0 0-6 0v1.79l6 6V6Z" />
    </svg>
  );
}

export function ErrorIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" className={className}>
      <path d="M12 2 1 21h22L12 2Zm1 15h-2v2h2v-2Zm0-7h-2v5h2v-5Z" />
    </svg>
  );
}

export function StateIcon({ state, className }: { state: ConnState; className?: string }) {
  if (state === "listening") return <MicIcon className={className} />;
  if (state === "speaking") return <SpeakingIcon className={className} />;
  if (state === "muted") return <MutedIcon className={className} />;
  if (state === "error") return <ErrorIcon className={className} />;
  return <MicIcon className={className} dimmed />;
}

export function BrandMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="#fff" className={className}>
      <path d="M12 15a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3Zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-2.08A7 7 0 0 0 19 12h-2Z" />
    </svg>
  );
}

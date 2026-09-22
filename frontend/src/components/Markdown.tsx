"use client";

import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * The assistant writes markdown -- bullets, tables, fenced code. Rendering
 * it as plain text showed the raw characters, so it goes through
 * react-markdown here.
 *
 * Only the elements the model actually emits are styled; everything else
 * falls back to the browser default. No raw HTML is enabled: react-markdown
 * escapes it by default, and we don't add `rehype-raw`, so a reply that
 * contains HTML is shown as text rather than run.
 */
const COMPONENTS: Components = {
  p: ({ children }) => <p className="my-2 first:mt-0 last:mb-0">{children}</p>,
  ul: ({ children }) => <ul className="my-2 list-disc pl-5 [&>li]:my-0.5">{children}</ul>,
  ol: ({ children }) => <ol className="my-2 list-decimal pl-5 [&>li]:my-0.5">{children}</ol>,
  h1: ({ children }) => <h1 className="mt-4 mb-2 text-[17px] font-semibold first:mt-0">{children}</h1>,
  h2: ({ children }) => <h2 className="mt-4 mb-2 text-[15.5px] font-semibold first:mt-0">{children}</h2>,
  h3: ({ children }) => <h3 className="mt-3 mb-1.5 text-[14.5px] font-semibold first:mt-0">{children}</h3>,
  a: ({ children, href }) => (
    <a href={href} target="_blank" rel="noreferrer noopener" className="text-accent underline underline-offset-2">
      {children}
    </a>
  ),
  blockquote: ({ children }) => (
    <blockquote className="my-2 border-l-2 border-border pl-3 text-text-muted">{children}</blockquote>
  ),
  hr: () => <hr className="my-3 border-border" />,

  // Tables need the wrapper: a wide one should scroll inside the message
  // rather than stretch the transcript.
  table: ({ children }) => (
    <div className="my-2.5 overflow-x-auto">
      <table className="w-full border-collapse text-[13px]">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="border-b border-border">{children}</thead>,
  tr: ({ children }) => <tr className="border-b border-border last:border-0">{children}</tr>,
  th: ({ children }) => <th className="px-2.5 py-1.5 text-left font-semibold whitespace-nowrap">{children}</th>,
  td: ({ children }) => <td className="px-2.5 py-1.5 align-top">{children}</td>,

  // react-markdown gives inline code a bare <code>, and fenced blocks a
  // <pre> wrapping a <code> -- so <pre> carries the block styling and
  // <code> only styles itself when it isn't inside one.
  pre: ({ children }) => (
    <pre className="my-2.5 overflow-x-auto rounded-lg border border-border bg-surface-sunken p-3 font-mono text-[12.5px] leading-[1.55]">
      {children}
    </pre>
  ),
  code: ({ children, className }) => {
    const fenced = typeof className === "string" && className.startsWith("language-");
    if (fenced) return <code className={className}>{children}</code>;
    return (
      <code className="rounded border border-border bg-surface-sunken px-1 py-0.5 font-mono text-[0.88em]">
        {children}
      </code>
    );
  },
};

export function Markdown({ children }: { children: string }) {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={COMPONENTS}>
      {children}
    </ReactMarkdown>
  );
}

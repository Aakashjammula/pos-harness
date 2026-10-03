"use client";

import { useState } from "react";
import { projectPath } from "@/lib/files";
import type { Round, ToolCall } from "@/lib/types";

interface ActivityTrailProps {
  rounds?: Round[];
  toolCalls?: ToolCall[];
  /** Opens a project file in the file panel. */
  onOpenFile?: (path: string) => void;
}

/** Tool calls that did the same sort of thing, shown as one row. */
interface Group {
  kind: string;
  verb: string; // "Read", "Edited", "Ran", ...
  calls: ToolCall[];
}

const KINDS: Record<string, { kind: string; verb: string }> = {
  read_file: { kind: "read", verb: "Read" },
  write_file: { kind: "edit", verb: "Edited" },
  edit_file: { kind: "edit", verb: "Edited" },
  delete: { kind: "edit", verb: "Deleted" },
  execute: { kind: "run", verb: "Ran" },
  ls: { kind: "look", verb: "Looked through" },
  glob: { kind: "look", verb: "Searched" },
  grep: { kind: "look", verb: "Searched" },
  tavily_search: { kind: "web", verb: "Searched the web" },
};

function describe(call: ToolCall): { kind: string; verb: string } {
  return KINDS[call.name] ?? { kind: call.name, verb: call.name };
}

/** The bit of a call worth showing: a path, or a command. */
function subject(call: ToolCall): string {
  const args = (call.args ?? {}) as Record<string, unknown>;
  const value = args.file_path ?? args.command ?? args.path ?? args.query ?? args.pattern;
  return typeof value === "string" ? value : JSON.stringify(args);
}

/**
 * Line counts for an edit, where they are real.
 *
 * `edit_file` carries the old and new strings, so the delta is computable.
 * `write_file` does not say what was there before, so it gets nothing
 * rather than a number that looks measured but isn't.
 */
function diffOf(call: ToolCall): string {
  const args = (call.args ?? {}) as Record<string, unknown>;
  const before = args.old_string;
  const after = args.new_string;
  if (typeof before !== "string" || typeof after !== "string") return "";
  const removed = before.split("\n").length;
  const added = after.split("\n").length;
  return `+${added} −${removed}`;
}

/** The project file a call read or wrote, if it can be opened: not a
 * deleted one, and not the app's own /skills/ or /memory/. */
function openablePath(call: ToolCall): string | null {
  if (call.name !== "read_file" && call.name !== "write_file" && call.name !== "edit_file") return null;
  const path = ((call.args ?? {}) as Record<string, unknown>).file_path;
  return typeof path === "string" ? projectPath(path) : null;
}

/** Consecutive calls of the same kind collapse into one row. */
function group(calls: ToolCall[]): Group[] {
  const groups: Group[] = [];
  for (const call of calls) {
    const { kind, verb } = describe(call);
    const last = groups[groups.length - 1];
    if (last && last.kind === kind) last.calls.push(call);
    else groups.push({ kind, verb, calls: [call] });
  }
  return groups;
}

function summarise(g: Group): string {
  // One call names what it touched; several are counted, since a row of
  // paths would not fit and the expansion lists them anyway.
  if (g.calls.length === 1) return `${g.verb} ${subject(g.calls[0])}`;
  const noun = g.kind === "run" ? "commands" : g.kind === "web" ? "searches" : "files";
  return `${g.verb} ${g.calls.length} ${noun}`;
}

function GroupRow({ group: g, onOpenFile }: { group: Group; onOpenFile?: (path: string) => void }) {
  const [open, setOpen] = useState(false);
  const diffs = g.calls.map(diffOf).filter(Boolean);
  // One file touched: offer to open it right on the row.
  const single = g.calls.length === 1 && onOpenFile ? openablePath(g.calls[0]) : null;

  return (
    <div className="min-w-0">
      <div className="flex w-full min-w-0 items-baseline gap-1.5 text-[12.5px] text-text-faint">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          className="flex min-w-0 flex-1 items-baseline gap-1.5 rounded py-0.5 text-left hover:text-text-muted"
        >
          <span className="shrink-0 text-[9px] leading-none">{open ? "▾" : "▸"}</span>
          <span className="min-w-0 flex-1 truncate">{summarise(g)}</span>
        </button>
        {g.calls.some((c) => c.pending) && (
          <span className="shrink-0 animate-pulse text-[11px] text-text-faint">running…</span>
        )}
        {diffs.length > 0 && (
          <span className="shrink-0 font-mono text-[11px] text-text-faint">{diffs.join(" ")}</span>
        )}
        {single && (
          <button
            type="button"
            onClick={() => onOpenFile?.(single)}
            className="shrink-0 text-[11.5px] text-text-faint underline-offset-2 hover:text-accent hover:underline"
          >
            open
          </button>
        )}
      </div>

      {open && (
        <div className="grid min-w-0 gap-1 border-l border-border pt-1 pb-0.5 pl-3">
          {g.calls.map((call, i) => (
            <div key={i} className="grid min-w-0 gap-0.5">
              <div className="flex min-w-0 items-baseline justify-between gap-2">
                {onOpenFile && openablePath(call) ? (
                  <button
                    type="button"
                    onClick={() => onOpenFile(openablePath(call)!)}
                    title={`Open ${subject(call)}`}
                    className="min-w-0 truncate text-left font-mono text-[11.5px] text-text-muted hover:text-accent hover:underline"
                  >
                    {subject(call)}
                  </button>
                ) : (
                  <span className="min-w-0 truncate font-mono text-[11.5px] text-text-muted" title={subject(call)}>
                    {subject(call)}
                  </span>
                )}
                {diffOf(call) && (
                  <span className="shrink-0 font-mono text-[11px] text-text-faint">{diffOf(call)}</span>
                )}
              </div>
              {call.cost_tokens != null && (
                <span className="font-mono text-[10.5px] text-text-faint">
                  {call.cost_estimated ? "~" : "+"}
                  {call.cost_tokens.toLocaleString()} tokens of context
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * The model's thinking, collapsed to one line until asked for. `live` while
 * it is still arriving -- then it reads "Thinking…" rather than "Thought".
 */
export function ThinkingRow({ text, live = false }: { text: string; live?: boolean }) {
  const [open, setOpen] = useState(false);
  if (!text.trim()) return null;
  return (
    <div className="min-w-0">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full min-w-0 items-baseline gap-1.5 rounded py-0.5 text-left text-[12.5px] text-text-faint hover:text-text-muted"
      >
        <span className="shrink-0 text-[9px] leading-none">{open ? "▾" : "▸"}</span>
        <span className={`min-w-0 flex-1 truncate ${live ? "animate-pulse" : ""}`}>
          {live ? "Thinking…" : "Thought"}
        </span>
      </button>
      {open && (
        <div className="max-h-72 overflow-y-auto whitespace-pre-wrap border-l border-border py-1 pl-3 text-[12.5px] leading-relaxed text-text-muted">
          {text}
        </div>
      )}
    </div>
  );
}

/**
 * What the assistant did during one turn, in order: what it said, what it
 * ran, what it said next.
 *
 * Each round's own text is stored (see trace.py), which is what lets the
 * commentary sit between the tool batches instead of all of it landing
 * after them.
 */
export function ActivityTrail({ rounds, toolCalls, onOpenFile }: ActivityTrailProps) {
  const calls = toolCalls ?? [];
  const thought = (rounds ?? []).some((r) => r.reasoning?.trim());
  if (calls.length === 0 && !thought) return null;

  // Turns recorded before per-round text existed have no rounds to walk, so
  // fall back to every tool group followed by the whole reply.
  if (!rounds || rounds.length === 0) {
    return (
      <div className="grid min-w-0 gap-0.5">
        {group(calls).map((g, i) => (
          <GroupRow key={i} group={g} onOpenFile={onOpenFile} />
        ))}
      </div>
    );
  }

  return (
    <div className="grid min-w-0 gap-2.5">
      {rounds.map((round) => {
        const mine = calls.filter((c) => (c.round ?? 0) === round.index);
        const isLast = round.index === rounds[rounds.length - 1].index;
        return (
          <div key={round.index} className="grid min-w-0 gap-1">
            <ThinkingRow text={round.reasoning ?? ""} />
            {/* The last round's text is the reply itself, rendered by the
                transcript -- showing it here too would duplicate it. */}
            {round.text && !isLast && (
              <p className="m-0 text-[15px] leading-relaxed text-text">{round.text}</p>
            )}
            {group(mine).map((g, i) => (
              <GroupRow key={i} group={g} onOpenFile={onOpenFile} />
            ))}
          </div>
        );
      })}
    </div>
  );
}

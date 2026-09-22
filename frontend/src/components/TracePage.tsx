"use client";

import { useMemo, useState } from "react";
import { Background, Controls, ReactFlow, type Edge, type Node } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { TranscriptLine } from "@/lib/types";

interface TracePageProps {
  lines: TranscriptLine[]; // the whole transcript, in order
  onBack: () => void;
}

/** A row in the summary's per-turn / per-round table. */
interface UsageRow {
  label: string;
  indented: boolean; // a round, shown under its turn
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
  hit: string; // "96%", or "miss" when nothing was served from cache
  rate: string; // output tokens per second, blank where we have no timing
  total?: boolean;
}

/** What the side panel shows for the node you clicked. */
interface NodeDetail {
  title: string;
  rows: [string, string][];
  table?: UsageRow[]; // the summary's breakdown; nothing else uses it
  block?: string; // long text (a prompt, a tool result), shown as-is
}

const CARD = "rounded-lg border px-3 py-2 text-[12px] leading-[1.45] w-[230px]";
const STEP = 110; // vertical gap between nodes on the spine

/** Cache hit rate: read tokens over input tokens, the standard definition.
 * Zero reads is called a miss rather than 0% -- it reads faster. */
function hitRate(read: number, input: number): string {
  if (!input) return "—";
  if (read === 0) return "miss";
  return `${Math.round((read / input) * 100)}%`;
}

function fmt(n: number | undefined): string {
  return n == null ? "—" : n.toLocaleString();
}

/** Tool results run to thousands of characters; the node shows a taste and
 * the side panel shows the rest. */
function preview(value: unknown, max = 90): string {
  const text = typeof value === "string" ? value : JSON.stringify(value ?? null);
  if (!text) return "";
  return text.length > max ? `${text.slice(0, max)}…` : text;
}

function full(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

/**
 * The whole chat as one graph, top to bottom: for each turn, your message,
 * the model, each tool call with its result, then the answer -- with one
 * turn feeding into the next. Every value here is already in the stored
 * trace; this is a second view of it, not new data.
 */
function buildGraph(lines: TranscriptLine[]): {
  nodes: Node[];
  edges: Edge[];
  details: Record<string, NodeDetail>;
  turns: number;
} {
  const nodes: Node[] = [];
  const edges: Edge[] = [];
  const details: Record<string, NodeDetail> = {};

  const place = (id: string, x: number, y: number, label: React.ReactNode) => {
    nodes.push({
      id,
      position: { x, y },
      data: { label },
      draggable: false,
      style: { background: "transparent", border: "none", padding: 0, width: "auto" },
    });
  };

  const link = (source: string, target: string, animated = false) => {
    edges.push({ id: `e-${source}-${target}`, source, target, animated, style: { stroke: "var(--border)" } });
  };

  let y = 0;
  let previousTurnEnd: string | null = "summary";
  let turns = 0;

  const traced = lines.filter((l) => l.usage);
  const sum = (pick: (u: NonNullable<TranscriptLine["usage"]>) => number | undefined) =>
    traced.reduce((n, l) => n + (pick(l.usage!) ?? 0), 0);
  const totals = {
    input: sum((u) => u.input_tokens),
    output: sum((u) => u.output_tokens),
    total: sum((u) => u.total_tokens),
    reasoning: sum((u) => u.reasoning_tokens),
    cacheRead: sum((u) => u.cache_read),
    cacheWrite: sum((u) => u.cache_creation),
    cost: traced.reduce((n, l) => n + (l.usage?.cost_usd ?? 0), 0),
    calls: traced.reduce((n, l) => n + (l.usage?.tool_calls?.length ?? 0), 0),
  };
  // The thread as it now stands: the newest turn's last-round input.
  const contextNow = traced.length > 0 ? traced[traced.length - 1].usage?.context_tokens : undefined;
  place(
    "summary",
    0,
    y,
    <div className={`${CARD} border-accent bg-bg text-text`}>
      <div className="mb-0.5 text-[10.5px] uppercase tracking-wide text-text-faint">Summary</div>
      <div className="font-medium">
        {traced.length} turn{traced.length === 1 ? "" : "s"} · {totals.calls} tool call
        {totals.calls === 1 ? "" : "s"}
      </div>
      <div className="mt-1 font-mono text-[10.5px] text-text-faint">
        {fmt(totals.total)} tokens · ${totals.cost.toFixed(4)}
      </div>
    </div>
  );
  // Per turn, then per round under it. Everything here is provider-reported
  // except `tok/s`, which needs timing we only have per turn.
  const usageRows: UsageRow[] = [];
  traced.forEach((line, i) => {
    const u = line.usage!;
    const rounds = u.rounds ?? [];
    // Output tokens per second, measured over the generating part only:
    // total time minus the wait for the first token.
    const generating = line.latency ? line.latency.total - line.latency.ttft : 0;
    usageRows.push({
      label: `turn ${i + 1}`,
      indented: false,
      input: u.input_tokens ?? 0,
      output: u.output_tokens ?? 0,
      cacheRead: u.cache_read ?? 0,
      cacheWrite: u.cache_creation ?? 0,
      hit: hitRate(u.cache_read ?? 0, u.input_tokens ?? 0),
      rate: generating > 0 && u.output_tokens ? (u.output_tokens / generating).toFixed(1) : "",
    });
    rounds.forEach((r, n) => {
      usageRows.push({
        label: `round ${n + 1}`,
        indented: true,
        input: r.input_tokens,
        output: r.output_tokens,
        cacheRead: r.cache_read,
        cacheWrite: r.cache_creation,
        hit: hitRate(r.cache_read, r.input_tokens),
        // Timing is recorded per turn, not per round -- leave it blank
        // rather than inventing a number.
        rate: "",
      });
    });
  });
  usageRows.push({
    label: "total",
    indented: false,
    input: totals.input,
    output: totals.output,
    cacheRead: totals.cacheRead,
    cacheWrite: totals.cacheWrite,
    hit: hitRate(totals.cacheRead, totals.input),
    rate: "",
    total: true,
  });

  details.summary = {
    title: "Whole conversation",
    rows: [
      ["context now", fmt(contextNow)],
      ["total cost", `$${totals.cost.toFixed(4)}`],
    ],
    table: usageRows,
  };
  y += STEP;

  // A turn is an assistant line with a trace on it, plus the most recent
  // user message before it.
  lines.forEach((line, lineIndex) => {
    if (!line.usage) return;
    const t = turns;
    turns += 1;
    const prompt =
      [...lines.slice(0, lineIndex)].reverse().find((l) => l.who === "you")?.text ?? "";
    buildTurn({ line, prompt, t, y, place, link, details });
    // Each turn is its own little column of nodes; advance past it.
    const roundCount = line.usage.rounds?.length ?? 1;
    y += STEP * (2 + roundCount + (line.usage.tool_calls?.length ?? 0) * 0.55 + 1);
    if (previousTurnEnd) link(previousTurnEnd, `prompt-${t}`);
    previousTurnEnd = `answer-${t}`;
  });

  return { nodes, edges, details, turns };
}

interface TurnArgs {
  line: TranscriptLine;
  prompt: string;
  t: number; // which turn this is, used to keep node ids unique
  y: number;
  place: (id: string, x: number, y: number, label: React.ReactNode) => void;
  link: (source: string, target: string, animated?: boolean) => void;
  details: Record<string, NodeDetail>;
}

function buildTurn({ line, prompt, t, y, place, link, details }: TurnArgs): void {
  const usage = line.usage;
  const calls = usage?.tool_calls ?? [];
  const rounds = usage?.rounds ?? [];
  const promptId = `prompt-${t}`;
  const answerId = `answer-${t}`;

  place(
    promptId,
    0,
    y,
    <div className={`${CARD} border-border bg-user-bubble text-text`}>
      <div className="mb-0.5 text-[10.5px] uppercase tracking-wide text-text-faint">You · turn {t + 1}</div>
      <div className="break-words">{preview(prompt, 110)}</div>
    </div>
  );
  details[promptId] = { title: `Turn ${t + 1} — your message`, rows: [], block: prompt };

  // Turns recorded before per-round tracking existed have no `rounds`, so
  // fall back to a single node carrying the turn's summed numbers.
  const spine: { index: number; input: number; output: number; delta: number | null }[] =
    rounds.length > 0
      ? rounds.map((r) => ({ index: r.index, input: r.input_tokens, output: r.output_tokens, delta: r.delta }))
      : [{ index: 0, input: usage?.input_tokens ?? 0, output: usage?.output_tokens ?? 0, delta: null }];

  let previous = promptId;
  spine.forEach((round, r) => {
    y += STEP;
    const roundId = `model-${t}-${r}`;
    const source = rounds[r];
    const only = spine.length === 1;

    place(
      roundId,
      0,
      y,
      <div className={`${CARD} border-accent-tint bg-accent-tint text-accent`}>
        <div className="mb-0.5 text-[10.5px] uppercase tracking-wide opacity-70">
          {only ? "Model" : `Round ${r + 1}`}
        </div>
        <div className="break-words font-medium">{usage?.model ?? "unknown"}</div>
        <div className="mt-1 font-mono text-[10.5px] opacity-80">
          {fmt(round.input)} in · {fmt(round.output)} out
          {round.delta != null && round.delta > 0 ? `  (+${fmt(round.delta)})` : ""}
        </div>
      </div>
    );
    details[roundId] = {
      title: only ? `Turn ${t + 1} — model call` : `Turn ${t + 1} — round ${r + 1}`,
      rows: [
        ["model", usage?.model ?? "—"],
        ["reasoning_effort", usage?.reasoning_effort ?? "—"],
        ["input tokens", fmt(round.input)],
        ["grew by", round.delta == null ? "— (first round)" : fmt(round.delta)],
        ["output tokens", fmt(round.output)],
        ["of which reasoning", fmt(source?.reasoning_tokens ?? usage?.reasoning_tokens)],
        ["  served from cache", fmt(source?.cache_read ?? usage?.cache_read)],
        ["  processed fresh", fmt(round.input - (source?.cache_read ?? usage?.cache_read ?? 0))],
        ["  of that, cached for later", fmt(source?.cache_creation ?? usage?.cache_creation)],
        ...(only
          ? ([
              ["finish_reason", usage?.finish_reason ?? "—"],
              ["billed input (all rounds)", fmt(usage?.input_tokens)],
              ["context after this turn", fmt(usage?.context_tokens)],
              ["context window", fmt(usage?.context_window)],
              ["cost", usage?.cost_usd != null ? `$${usage.cost_usd.toFixed(4)}` : "—"],
              ["time to first token", line.latency ? `${Math.round(line.latency.ttft)} ms` : "—"],
              ["total time", line.latency ? `${Math.round(line.latency.total)} ms` : "—"],
            ] as [string, string][])
          : []),
      ],
    };
    link(previous, roundId);
    previous = roundId;

    // The calls this round made, each with what it cost -- the growth it
    // caused in the round after it.
    calls
      .map((call, i) => ({ call, i }))
      .filter(({ call }) => (call.round ?? 0) === round.index)
      .forEach(({ call, i }, n) => {
        const callId = `call-${t}-${i}`;
        const resultId = `result-${t}-${i}`;
        const rowY = y + STEP * (n + 1) * 0.55;
        const cost =
          call.cost_tokens == null
            ? null
            : `${call.cost_estimated ? "~" : "+"}${call.cost_tokens.toLocaleString()} tok`;

        place(
          callId,
          280,
          rowY,
          <div className={`${CARD} border-border bg-surface-sunken text-text`}>
            <div className="mb-0.5 flex items-baseline justify-between gap-2 text-[10.5px] uppercase tracking-wide text-text-faint">
              <span>Tool</span>
              {cost && <span className="font-mono normal-case text-accent">{cost}</span>}
            </div>
            <div className="break-words font-mono font-medium">{call.name}</div>
            <div className="mt-1 break-words font-mono text-[10.5px] text-text-faint">{preview(call.args, 60)}</div>
          </div>
        );
        details[callId] = {
          title: `${call.name} — arguments`,
          rows: [
            [
              "cost",
              call.cost_tokens == null
                ? "—"
                : `${call.cost_tokens.toLocaleString()} tokens${call.cost_estimated ? " (share of a batched round)" : ""}`,
            ],
            ["round", String((call.round ?? 0) + 1)],
          ],
          block: full(call.args ?? {}),
        };

        place(
          resultId,
          560,
          rowY,
          <div className={`${CARD} border-border bg-bg text-text-muted`}>
            <div className="mb-0.5 text-[10.5px] uppercase tracking-wide text-text-faint">Result</div>
            <div className="break-words font-mono text-[10.5px]">{preview(call.result) || "(empty)"}</div>
          </div>
        );
        details[resultId] = { title: `${call.name} — result`, rows: [], block: full(call.result) };

        link(roundId, callId, true);
        link(callId, resultId);
      });

    // Leave room for the rows that hang off this round.
    const hanging = calls.filter((c) => (c.round ?? 0) === round.index).length;
    y += STEP * Math.max(0, hanging) * 0.55;
  });

  y += STEP;
  place(
    answerId,
    0,
    y,
    <div className={`${CARD} border-accent bg-bg text-text`}>
      <div className="mb-0.5 text-[10.5px] uppercase tracking-wide text-text-faint">Answer</div>
      <div className="break-words">{preview(line.text, 110)}</div>
    </div>
  );
  details[answerId] = {
    title: `Turn ${t + 1} — answer`,
    rows: [
      ["finish_reason", usage?.finish_reason ?? "—"],
      ["billed input (all rounds)", fmt(usage?.input_tokens)],
      ["context after this turn", fmt(usage?.context_tokens)],
      ["cost", usage?.cost_usd != null ? `$${usage.cost_usd.toFixed(4)}` : "—"],
    ],
    block: line.text,
  };
  link(previous, answerId);
}

export function TracePage({ lines, onBack }: TracePageProps) {
  const { nodes, edges, details, turns } = useMemo(() => buildGraph(lines), [lines]);
  const [selected, setSelected] = useState("summary");
  const detail = details[selected];
  const toolCalls = lines.reduce((n, l) => n + (l.usage?.tool_calls?.length ?? 0), 0);
  const cost = lines.reduce((sum, l) => sum + (l.usage?.cost_usd ?? 0), 0);

  return (
    <div className="flex h-screen flex-1 flex-col">
      <div className="flex shrink-0 items-center gap-3.5 border-b border-border px-6 py-3.5">
        <button
          type="button"
          onClick={onBack}
          className="rounded-lg px-2.5 py-1.5 text-[13px] font-medium text-text-muted hover:bg-surface-sunken hover:text-text"
        >
          ← Back
        </button>
        <div className="text-[14.5px] font-semibold">Trace</div>
        <div className="text-[12.5px] text-text-faint">
          {turns} turn{turns === 1 ? "" : "s"} · {toolCalls} tool call{toolCalls === 1 ? "" : "s"} · $
          {cost.toFixed(4)}
        </div>
      </div>

      <div className="flex min-h-0 flex-1 max-[900px]:flex-col">
        <div className="min-w-0 flex-1">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            fitView
            fitViewOptions={{ padding: 0.25 }}
            onNodeClick={(_, node) => setSelected(node.id)}
            nodesConnectable={false}
            nodesDraggable={false}
          >
            <Background gap={16} color="var(--border)" />
            <Controls showInteractive={false} />
          </ReactFlow>
        </div>

        {/* Nodes only have room for a preview; the full arguments and
            results live here. */}
        <aside className="w-[360px] shrink-0 overflow-y-auto border-l border-border p-4 max-[900px]:w-auto max-[900px]:border-l-0 max-[900px]:border-t">
          {detail ? (
            <>
              <h2 className="m-0 text-[13.5px] font-semibold text-text">{detail.title}</h2>
              {detail.rows.length > 0 && (
                <div className="mt-2.5 grid gap-1">
                  {detail.rows.map(([key, value]) => (
                    <div key={key} className="flex items-baseline justify-between gap-3 text-[12px]">
                      <span className="text-text-faint">{key}</span>
                      <span className="break-all text-right font-mono text-[11.5px] text-text-muted">{value}</span>
                    </div>
                  ))}
                </div>
              )}
              {detail.table && (
                <div className="mt-3 overflow-x-auto">
                  <table className="w-full border-collapse font-mono text-[11px]">
                    <thead className="text-text-faint">
                      <tr className="border-b border-border">
                        <th className="py-1 pr-2 text-left font-normal">&nbsp;</th>
                        <th className="py-1 px-1 text-right font-normal">in</th>
                        <th className="py-1 px-1 text-right font-normal">out</th>
                        <th className="py-1 px-1 text-right font-normal">rd</th>
                        <th className="py-1 px-1 text-right font-normal">wr</th>
                        <th className="py-1 px-1 text-right font-normal">hit</th>
                        <th className="py-1 pl-1 text-right font-normal">tok/s</th>
                      </tr>
                    </thead>
                    <tbody>
                      {detail.table.map((r, i) => (
                        <tr
                          // Position, not label: "round 1" repeats under every
                          // turn, and duplicate keys make React drop rows.
                          key={i}
                          className={r.total ? "border-t border-border font-semibold text-text" : "text-text-muted"}
                        >
                          <td className={`py-0.5 pr-2 whitespace-nowrap ${r.indented ? "pl-3 text-text-faint" : ""}`}>
                            {r.label}
                          </td>
                          <td className="py-0.5 px-1 text-right">{r.input.toLocaleString()}</td>
                          <td className="py-0.5 px-1 text-right">{r.output.toLocaleString()}</td>
                          <td className="py-0.5 px-1 text-right">{r.cacheRead.toLocaleString()}</td>
                          <td className="py-0.5 px-1 text-right">{r.cacheWrite.toLocaleString()}</td>
                          <td className={`py-0.5 px-1 text-right ${r.hit === "miss" ? "text-danger" : ""}`}>
                            {r.hit}
                          </td>
                          <td className="py-0.5 pl-1 text-right">{r.rate}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {detail.block && (
                <pre className="mt-3 mb-0 max-h-[60vh] overflow-auto whitespace-pre-wrap break-words rounded-lg border border-border bg-surface-sunken p-2.5 font-mono text-[11px] leading-[1.55] text-text-muted">
                  {detail.block}
                </pre>
              )}
            </>
          ) : (
            <div className="text-[12.5px] text-text-faint">Click a node to inspect it.</div>
          )}
        </aside>
      </div>
    </div>
  );
}

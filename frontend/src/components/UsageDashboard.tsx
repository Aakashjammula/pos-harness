"use client";

import { useEffect, useState } from "react";
import { formatTokens } from "@/lib/format";
import { fetchUsage, type UsageSummary } from "@/lib/usage";

// Two series (input vs output tokens). Both colors are already in this app's
// design system, and the pair was checked with the palette validator: it
// clears the lightness band, chroma floor, CVD separation and contrast in
// both light and dark. Identity is never colour-alone -- there's a legend
// and the per-model rows state the numbers directly.
const INPUT_COLOR = "#297AFF";
const OUTPUT_COLOR = "#10a37f";

const RANGES = [
  { key: "all", label: "All" },
  { key: "30d", label: "30d" },
  { key: "7d", label: "7d" },
];

function Tile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border bg-surface-sunken px-3 py-2.5">
      <div className="text-[11.5px] text-text-faint">{label}</div>
      <div className="mt-0.5 truncate text-[15px] font-semibold text-text">{value}</div>
    </div>
  );
}

function formatHour(hour: number | null): string {
  if (hour === null) return "—";
  const suffix = hour < 12 ? "AM" : "PM";
  const h = hour % 12 === 0 ? 12 : hour % 12;
  return `${h} ${suffix}`;
}

/** Day squares, shaded by that day's token volume -- one hue, light to dark,
 * which is the rule for magnitude. */
function Heatmap({ data }: { data: UsageSummary["by_day"] }) {
  if (data.length === 0) return null;
  const max = Math.max(...data.map((d) => d.input + d.output), 1);

  return (
    <div className="flex flex-wrap gap-[3px]">
      {data.map((d) => {
        const total = d.input + d.output;
        // Four steps rather than a continuous ramp: easier to read, and
        // every step stays distinguishable against the surface.
        const step = total === 0 ? 0 : Math.ceil((total / max) * 4);
        const opacity = [0.08, 0.3, 0.5, 0.75, 1][step];
        return (
          <div
            key={d.day}
            title={`${d.day}: ${formatTokens(total)} tokens · ${d.messages} messages · $${d.cost.toFixed(4)}`}
            className="h-[13px] w-[13px] rounded-[3px]"
            style={{ backgroundColor: OUTPUT_COLOR, opacity }}
          />
        );
      })}
    </div>
  );
}

/** Stacked bars per day: input below, output above, with a 2px surface gap
 * between the two segments so they never read as one block. */
function DayBars({ data }: { data: UsageSummary["by_day"] }) {
  if (data.length === 0) return <div className="text-[12.5px] text-text-faint">No usage yet.</div>;
  const max = Math.max(...data.map((d) => d.input + d.output), 1);

  return (
    <div>
      <div className="flex h-[150px] items-end gap-[3px]">
        {data.map((d) => {
          const total = d.input + d.output;
          return (
            <div
              key={d.day}
              title={`${d.day}: ${formatTokens(d.input)} in · ${formatTokens(d.output)} out · $${d.cost.toFixed(4)}`}
              className="flex min-w-[6px] flex-1 flex-col justify-end"
              style={{ height: "100%" }}
            >
              <div
                className="w-full rounded-t-[4px]"
                style={{ height: `${(d.output / max) * 100}%`, backgroundColor: OUTPUT_COLOR }}
              />
              <div className="h-[2px] w-full" />
              <div
                className="w-full"
                style={{ height: `${(d.input / max) * 100}%`, backgroundColor: INPUT_COLOR }}
              />
              <div className="sr-only">{`${d.day}: ${total} tokens`}</div>
            </div>
          );
        })}
      </div>
      <div className="mt-2 flex items-center justify-between text-[11px] text-text-faint">
        <span>{data[0]?.day}</span>
        <span>{data[data.length - 1]?.day}</span>
      </div>
    </div>
  );
}

function Legend() {
  return (
    <div className="flex items-center gap-3 text-[11.5px] text-text-muted">
      <span className="flex items-center gap-1.5">
        <span className="h-2 w-2 rounded-[2px]" style={{ backgroundColor: INPUT_COLOR }} />
        Input
      </span>
      <span className="flex items-center gap-1.5">
        <span className="h-2 w-2 rounded-[2px]" style={{ backgroundColor: OUTPUT_COLOR }} />
        Output
      </span>
    </div>
  );
}

export function UsageDashboard() {
  const [tab, setTab] = useState<"overview" | "models">("overview");
  const [range, setRange] = useState("all");
  const [data, setData] = useState<UsageSummary | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- refetches when the range changes
    setError(false);
    fetchUsage(range)
      .then((d) => !cancelled && setData(d))
      .catch(() => !cancelled && setError(true));
    return () => {
      cancelled = true;
    };
  }, [range]);

  if (error) return <div className="text-[12.5px] text-text-faint">Couldn&apos;t reach the backend.</div>;
  if (!data) return <div className="text-[12.5px] text-text-faint">Loading…</div>;

  return (
    <div className="grid gap-3">
      <div className="flex items-center justify-between">
        <div className="flex gap-0.5 rounded-lg bg-surface-sunken p-[3px]">
          {(["overview", "models"] as const).map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setTab(t)}
              className={`rounded-md px-2.5 py-1 text-[12px] font-medium capitalize transition-colors ${
                tab === t ? "bg-bg text-text shadow-[var(--shadow)]" : "text-text-muted hover:text-text"
              }`}
            >
              {t}
            </button>
          ))}
        </div>
        <div className="flex gap-0.5 rounded-lg bg-surface-sunken p-[3px]">
          {RANGES.map((r) => (
            <button
              key={r.key}
              type="button"
              onClick={() => setRange(r.key)}
              className={`rounded-md px-2.5 py-1 text-[12px] font-medium transition-colors ${
                range === r.key ? "bg-bg text-text shadow-[var(--shadow)]" : "text-text-muted hover:text-text"
              }`}
            >
              {r.label}
            </button>
          ))}
        </div>
      </div>

      {tab === "overview" ? (
        <>
          <div className="grid grid-cols-3 gap-2 max-[560px]:grid-cols-2">
            <Tile label="Sessions" value={String(data.sessions)} />
            <Tile label="Messages" value={String(data.messages)} />
            <Tile label="Total tokens" value={formatTokens(data.totals.total)} />
            <Tile label="Active days" value={String(data.active_days)} />
            <Tile label="Peak hour" value={formatHour(data.peak_hour)} />
            <Tile label="Total cost" value={`$${data.totals.cost.toFixed(4)}`} />
          </div>
          <Heatmap data={data.by_day} />
          <div className="text-[11.5px] text-text-faint">
            {formatTokens(data.totals.cache_read)} tokens served from cache
            {data.favorite_model ? ` · mostly ${data.favorite_model}` : ""}
          </div>
        </>
      ) : (
        <>
          <Legend />
          <DayBars data={data.by_day} />
          <div className="grid gap-1">
            {data.by_model.map((m) => (
              <div key={m.model} className="flex items-baseline justify-between gap-3 text-[12.5px]">
                <span className="truncate text-text">{m.model}</span>
                <span className="shrink-0 font-mono text-[11.5px] text-text-faint">
                  {formatTokens(m.input)} in · {formatTokens(m.output)} out
                </span>
                <span className="w-[46px] shrink-0 text-right font-medium text-text">
                  {(m.share * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

"use client";

import { useEffect, useMemo, useState } from "react";
import { Background, ReactFlow, type Edge, type Node } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { fetchTools, type SkillInfo, type ToolInfo } from "@/lib/tools";

const NODE_BASE =
  "rounded-lg border px-3 py-1.5 text-[12px] font-medium whitespace-nowrap";

/** Builds the model-in-the-middle graph: one node per tool, each wired
 * back to the model. Inactive tools (no API key) are dimmed rather than
 * hidden, so it's clear what's available but switched off. */
function toGraph(tools: ToolInfo[], model: string): { nodes: Node[]; edges: Edge[] } {
  const modelId = "model";
  const nodes: Node[] = [
    {
      id: modelId,
      position: { x: 0, y: Math.max(0, (tools.length - 1) * 21) },
      data: {
        label: (
          <div className={`${NODE_BASE} border-accent-tint bg-accent-tint text-accent`}>{model}</div>
        ),
      },
      type: "input",
      draggable: false,
      style: { background: "transparent", border: "none", padding: 0, width: "auto" },
    },
  ];
  const edges: Edge[] = [];

  tools.forEach((tool, i) => {
    nodes.push({
      id: tool.name,
      position: { x: 240, y: i * 42 },
      data: {
        label: (
          <div
            title={tool.description}
            className={`${NODE_BASE} ${
              tool.active
                ? "border-border bg-surface-sunken text-text"
                : "border-border bg-surface-sunken text-text-faint opacity-60"
            }`}
          >
            {tool.name}
            {!tool.active && " (off)"}
          </div>
        ),
      },
      type: "output",
      draggable: false,
      style: { background: "transparent", border: "none", padding: 0, width: "auto" },
    });
    edges.push({
      id: `${modelId}-${tool.name}`,
      source: modelId,
      target: tool.name,
      animated: tool.active,
      style: { stroke: "var(--border)" },
    });
  });

  return { nodes, edges };
}

/** What the agent can currently do, as a graph. */
export function ToolGraph() {
  const [data, setData] = useState<{ tools: ToolInfo[]; skills: SkillInfo[]; model: string } | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    fetchTools()
      .then(setData)
      .catch(() => setError(true));
  }, []);

  const graph = useMemo(() => (data ? toGraph(data.tools ?? [], data.model) : { nodes: [], edges: [] }), [data]);
  const skills = data?.skills ?? [];

  if (error) return <div className="text-[12.5px] text-text-faint">Couldn&apos;t reach the backend.</div>;
  if (!data) return <div className="text-[12.5px] text-text-faint">Loading…</div>;

  return (
    <div className="grid gap-3">
      <div className="h-[320px] w-full overflow-hidden rounded-xl border border-border">
        <ReactFlow
          nodes={graph.nodes}
          edges={graph.edges}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          nodesConnectable={false}
          nodesDraggable={false}
          zoomOnScroll={false}
          panOnDrag={false}
        >
          <Background gap={16} color="var(--border)" />
        </ReactFlow>
      </div>

      {/* Skills as chips rather than graph nodes -- 20+ boxes would make the
          graph unreadable, and skills are loaded on demand, not wired in.
          Defaulted, since an older backend's /tools has no `skills` at all. */}
      {skills.length > 0 && (
        <div>
          <div className="mb-1.5 text-[12px] text-text-muted">
            Skills <span className="text-text-faint">({skills.length}, loaded on demand)</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {skills.map((s) => (
              <span
                key={s.name}
                title={s.description}
                className="rounded-full border border-border bg-surface-sunken px-2.5 py-1 text-[11.5px] text-text-muted"
              >
                {s.name}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

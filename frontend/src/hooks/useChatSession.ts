"use client";

import { useCallback, useRef, useState } from "react";
import { streamChat } from "@/lib/chatStream";
import type { StoredTurn } from "@/lib/sessions";
import type { TranscriptLine, Usage } from "@/lib/types";

let lineIdSeq = 0;
const nextLineId = () => `line-${++lineIdSeq}`;

interface SendOptions {
  folder?: string | null;
  reasoningEffort?: string;
  model?: string | null;
}

/**
 * Chat state backed by the real backend (POST /chat/stream, server-sent
 * events). `thread_id` is the session id -- reusing it across messages is
 * what gives a chat real memory (see the backend design discussion); a new
 * chat just stops reusing the old one. `historyVersion` bumps whenever a
 * session is created or titled, so the sidebar knows to refetch its list.
 */
export function useChatSession() {
  const [lines, setLines] = useState<TranscriptLine[]>([]);
  const [lineCountLabel, setLineCountLabel] = useState("");
  const [replying, setReplying] = useState(false);
  const [activity, setActivity] = useState<string | null>(null); // what it's doing right now
  // Call ids in the order they were announced, so a result can be matched
  // back to its row. Reset per turn.
  const liveIds = useRef<string[]>([]);
  const [historyVersion, setHistoryVersion] = useState(0);
  const [threadId, setThreadIdState] = useState<string | null>(null);
  const threadIdRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const setThreadId = useCallback((id: string | null) => {
    threadIdRef.current = id;
    setThreadIdState(id);
  }, []);

  const addLine = useCallback((who: TranscriptLine["who"], text: string, usage?: Usage) => {
    setLines((prev) => {
      const next = [...prev, { id: nextLineId(), who, text, usage }];
      const turns = next.filter((l) => l.who === "you" || l.who === "bot").length;
      setLineCountLabel(turns ? `${turns}${turns === 1 ? " message" : " messages"}` : "");
      return next;
    });
  }, []);

  // Replace one transcript line in place (the streaming bot reply).
  const patchLine = useCallback((id: string, patch: (line: TranscriptLine) => TranscriptLine) => {
    setLines((prev) => prev.map((l) => (l.id === id ? patch(l) : l)));
  }, []);

  const sendText = useCallback(
    (text: string, options: SendOptions = {}) => {
      const value = text.trim();
      if (!value || abortRef.current) return;

      addLine("you", value);
      const botId = nextLineId();
      setLines((prev) => [...prev, { id: botId, who: "bot", text: "" }]);

      liveIds.current = [];
      const controller = new AbortController();
      abortRef.current = controller;
      setReplying(true);

      streamChat(
        {
          message: value,
          thread_id: threadIdRef.current,
          folder: options.folder ?? null,
          reasoning_effort: options.reasoningEffort ?? "medium",
          model: options.model ?? null,
        },
        {
          onSession: (id) => {
            if (threadIdRef.current !== id) {
              setThreadId(id);
              setHistoryVersion((v) => v + 1); // a new session now exists
            }
          },
          onToken: (piece) => {
            setActivity(null); // text arriving means the tool work is done
            patchLine(botId, (l) => ({ ...l, text: l.text + piece }));
          },
          onActivity: (id, tool, args) => {
            // Show the most telling argument (a path, a query) rather than
            // the whole blob -- this is a one-line status, not a trace.
            const detail = args.file_path ?? args.path ?? args.query ?? args.command ?? "";
            setActivity(detail ? `${tool} ${String(detail)}` : tool);
            // And add it to the reply's own list, so the activity trail
            // builds up as the turn runs rather than appearing at the end.
            patchLine(botId, (l) => ({
              ...l,
              liveCalls: [...(l.liveCalls ?? []), { name: tool, args, pending: true }],
            }));
            liveIds.current.push(id);
          },
          onActivityResult: (id, result) => {
            const index = liveIds.current.indexOf(id);
            if (index < 0) return;
            patchLine(botId, (l) => ({
              ...l,
              liveCalls: (l.liveCalls ?? []).map((c, i) =>
                i === index ? { ...c, result, pending: false } : c
              ),
            }));
          },
          onDone: (d) => {
            const usage: Usage = {
              ...d.usage,
              model: d.model,
              finish_reason: d.finish_reason,
              reasoning_effort: d.reasoning_effort,
              tool_calls: d.tool_calls,
            };
            patchLine(botId, (l) => ({ ...l, text: d.text, usage }));
          },
          onTitle: (_title, titleUsage) => {
            setHistoryVersion((v) => v + 1); // the sidebar's row label just changed
            // Titling is its own billed request, made after `done`. The
            // backend re-states the turn's totals with it folded in, so the
            // figure on screen matches what was stored.
            if (titleUsage) {
              patchLine(botId, (l) => ({ ...l, usage: { ...l.usage, ...titleUsage } }));
            }
          },
          onError: (message) => {
            // Drop the empty bubble if nothing had arrived; keep partial text if it had.
            setLines((prev) => prev.filter((l) => l.id !== botId || l.text !== ""));
            addLine("error", message);
          },
        },
        controller.signal
      )
        .catch((e: unknown) => {
          if ((e as Error).name === "AbortError") return; // a new chat started mid-reply
          setLines((prev) => prev.filter((l) => l.id !== botId || l.text !== ""));
          addLine("error", `Connection problem: ${(e as Error).message}`);
        })
        .finally(() => {
          if (abortRef.current === controller) abortRef.current = null;
          setReplying(false);
          setActivity(null);
        });
    },
    [addLine, patchLine, setThreadId]
  );

  const newChat = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setThreadId(null);
    setLines([]);
    setLineCountLabel("");
    setReplying(false);
    setActivity(null);
  }, [setThreadId]);

  /** Loads a past session's stored turns into the transcript, and makes
   * new messages continue that same thread. */
  const loadHistory = useCallback(
    (id: string, turns: StoredTurn[]) => {
      abortRef.current?.abort();
      abortRef.current = null;
      setReplying(false);
      setThreadId(id);
      setLines(
        turns.map((turn) => ({
          id: nextLineId(),
          who: turn.role === "user" ? "you" : "bot",
          text: turn.text,
          usage: turn.usage
            ? {
                ...turn.usage,
                model: turn.model,
                finish_reason: turn.finish_reason,
                reasoning_effort: turn.reasoning_effort ?? undefined,
                // The API returns these beside `usage`, not inside it; the
                // trace view reads them off usage, so fold them in here.
                tool_calls: turn.tool_calls ?? undefined,
              }
            : undefined,
        }))
      );
      const count = turns.filter((t) => t.role === "user" || t.role === "assistant").length;
      setLineCountLabel(count ? `${count}${count === 1 ? " message" : " messages"}` : "");
    },
    [setThreadId]
  );

  return { lines, lineCountLabel, replying, activity, threadId, historyVersion, sendText, newChat, loadHistory };
}

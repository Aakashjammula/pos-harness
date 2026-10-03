"use client";

import { useEffect, useMemo, useState } from "react";
import CodeMirror, { EditorState, EditorView, type Extension } from "@uiw/react-codemirror";
import { vscodeDark, vscodeLight } from "@uiw/codemirror-theme-vscode";
import { css } from "@codemirror/lang-css";
import { html } from "@codemirror/lang-html";
import { javascript } from "@codemirror/lang-javascript";
import { json } from "@codemirror/lang-json";
import { markdown } from "@codemirror/lang-markdown";
import { python } from "@codemirror/lang-python";

/** Syntax highlighting for a file, by its extension. Plain text otherwise. */
function languageFor(path: string): Extension[] {
  const ext = path.toLowerCase().split(".").pop() ?? "";
  switch (ext) {
    case "js":
    case "mjs":
    case "cjs":
      return [javascript()];
    case "jsx":
      return [javascript({ jsx: true })];
    case "ts":
    case "mts":
    case "cts":
      return [javascript({ typescript: true })];
    case "tsx":
      return [javascript({ jsx: true, typescript: true })];
    case "py":
    case "pyi":
      return [python()];
    case "html":
    case "htm":
    case "svg":
    case "xml":
      return [html()];
    case "css":
      return [css()];
    case "json":
    case "jsonc":
      return [json()];
    case "md":
    case "mdx":
      return [markdown()];
    default:
      return [];
  }
}

/** Follows the OS theme, the same signal globals.css switches on. */
function usePrefersDark(): boolean {
  const [dark, setDark] = useState(false);
  useEffect(() => {
    const query = window.matchMedia("(prefers-color-scheme: dark)");
    // eslint-disable-next-line react-hooks/set-state-in-effect -- syncs with the media query once mounted
    setDark(query.matches);
    const onChange = (e: MediaQueryListEvent) => setDark(e.matches);
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);
  return dark;
}

/**
 * A read-only code viewer -- VS Code's look, CodeMirror's weight.
 *
 * Read-only for now: the agent is the one writing these files, and an
 * editor that saves would race it. Making it editable is `editable` plus a
 * save endpoint, not a different component.
 */
export default function CodeView({ path, content }: { path: string; content: string }) {
  const dark = usePrefersDark();
  const extensions = useMemo(
    () => [...languageFor(path), EditorState.readOnly.of(true), EditorView.editable.of(false)],
    [path]
  );
  return (
    <CodeMirror
      value={content}
      height="100%"
      className="h-full text-[12.5px]"
      theme={dark ? vscodeDark : vscodeLight}
      extensions={extensions}
      basicSetup={{ foldGutter: true, highlightActiveLine: false, searchKeymap: true }}
    />
  );
}

import { API_URL } from "./config";

export interface FileEntry {
  name: string;
  /** Relative to the open folder, forward slashes, no leading slash. */
  path: string;
  type: "dir" | "file";
  size: number | null;
  /** A usually-generated folder (node_modules, .git…): shown, greyed. */
  heavy: boolean;
  /** Listed but not openable: a .env, or a link pointing outside. */
  locked: boolean;
}

export interface DirListing {
  path: string;
  entries: FileEntry[];
  truncated: boolean;
  preview_key: string;
}

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      detail = (await res.json()).detail || detail;
    } catch {
      // not JSON: keep the status
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

/** One folder's entries. The tree asks for each level as it is expanded. */
export function fetchDir(folder: string, path = ""): Promise<DirListing> {
  const q = new URLSearchParams({ folder, path });
  return getJson(`${API_URL}/fs/tree?${q}`);
}

/** One file's text. Fails with the backend's reason for binaries, big
 * files, .env files and anything outside the folder. */
export async function fetchFile(folder: string, path: string): Promise<string> {
  const q = new URLSearchParams({ folder, path });
  return (await getJson<{ content: string }>(`${API_URL}/fs/file?${q}`)).content;
}

/** Where a file runs as itself, sandboxed. The folder travels in the path
 * (as `key`) so the page's own relative links resolve too. */
export function previewUrl(key: string, path: string): string {
  return `${API_URL}/fs/preview/${key}/${path.split("/").map(encodeURIComponent).join("/")}`;
}

/**
 * A path as the agent writes it ("/src/app.py") in the panel's terms
 * ("src/app.py"), or null for one that isn't in the project at all --
 * /skills/ and /memory/ are the app's own folders, mounted for the agent.
 */
export function projectPath(agentPath: string): string | null {
  if (/^\/(skills|memory)\//.test(agentPath)) return null;
  const path = agentPath.replace(/^\/+/, "");
  return path || null;
}

/** Tools that can change what is on disk -- the panel reloads after them. */
export const WRITING_TOOLS = new Set(["write_file", "edit_file", "delete", "execute"]);

/** Files and folders matching what was typed after "@", best first. */
export async function searchFiles(folder: string, q: string, signal?: AbortSignal): Promise<FileEntry[]> {
  const params = new URLSearchParams({ folder, q });
  const res = await fetch(`${API_URL}/fs/search?${params}`, { signal });
  if (!res.ok) return [];
  return ((await res.json()) as { entries: FileEntry[] }).entries;
}

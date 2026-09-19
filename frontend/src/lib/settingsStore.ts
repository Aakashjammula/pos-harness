import { DEFAULT_SETTINGS, type Settings } from "./types";

// Which provider and model you last used, your voice and input preferences. Kept in this
// browser, per account. Never API keys: those live encrypted on the server, and the fields
// that hold them are not part of what is saved here.
const PERSISTED = [
  "provider",
  "llmModel",
  "ttsEngine",
  "ttsVoice",
  "micDeviceId",
  "voiceInputMode",
  "triggerWord",
  "vadThreshold",
  "vadMinSilenceMs",
  "vadSpeechPadMs",
] as const;

const storageKey = (userId: string) => `pos.settings.v1.${userId}`;

/** Saved preferences, or the defaults. Unknown or wrongly-typed entries are ignored, so an old
 * or hand-edited value can never break the page. Safe to call only in the browser. */
export function loadSettings(userId: string): Settings {
  try {
    const raw = JSON.parse(localStorage.getItem(storageKey(userId)) ?? "{}") as Record<string, unknown>;
    const picked: Record<string, string> = {};
    for (const key of PERSISTED) {
      if (typeof raw[key] === "string") picked[key] = raw[key] as string;
    }
    return { ...DEFAULT_SETTINGS, ...picked };
  } catch {
    return DEFAULT_SETTINGS; // storage blocked or unreadable: just start fresh
  }
}

export function saveSettings(userId: string, settings: Settings): void {
  try {
    const picked = Object.fromEntries(PERSISTED.map((key) => [key, settings[key]]));
    localStorage.setItem(storageKey(userId), JSON.stringify(picked));
  } catch {
    // private mode / quota: preferences just won't persist
  }
}

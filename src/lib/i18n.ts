// b141.142: UI language — Russian (default) or English, switched in the top bar.
// Strings live in place as pairs: L("Сохранить", "Save"). L reads the language
// signal, so JSX that calls it re-renders on a switch; a string built at call
// time (toast, history label, dialog) takes the language of that moment.
// The choice is an app preference (localStorage), not a project setting.
import { createSignal } from "solid-js";
import { invoke } from "@tauri-apps/api/core";

export type Lang = "ru" | "en";

const KEY = "phaseforge.lang";

function initial(): Lang {
  try { return localStorage.getItem(KEY) === "en" ? "en" : "ru"; } catch { return "ru"; }
}

const [lang, setLangSignal] = createSignal<Lang>(initial());
export { lang };

/** Tells the backend, whose analysis/export messages are worded per language. */
function syncBackend(l: Lang): void {
  try { Promise.resolve(invoke("set_ui_language", { lang: l })).catch(() => {}); } catch { /* no backend (tests) */ }
}
syncBackend(lang());
try { document.documentElement.lang = lang(); } catch { /* no DOM */ }

export function setLang(l: Lang): void {
  if (l === lang()) return;
  try { localStorage.setItem(KEY, l); } catch { /* private storage — session only */ }
  setLangSignal(l);
  document.documentElement.lang = l;
  syncBackend(l);
}

export const isEn = (): boolean => lang() === "en";

/** The string of the current language. */
export function L(ru: string, en: string): string {
  return lang() === "en" ? en : ru;
}

/** Plural: Russian 1 / 2–4 / 5+ forms, English one / other. */
export function plural(n: number, ru: [string, string, string], en: [string, string]): string {
  if (lang() === "en") return Math.abs(n) === 1 ? en[0] : en[1];
  const a = Math.abs(n) % 100;
  const b = a % 10;
  if (a >= 11 && a <= 14) return ru[2];
  if (b === 1) return ru[0];
  if (b >= 2 && b <= 4) return ru[1];
  return ru[2];
}

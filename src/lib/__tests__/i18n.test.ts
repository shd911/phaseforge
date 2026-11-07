// b141.142: the RU | EN switch — L follows the signal, plurals per language,
// the choice survives a restart.
import { describe, it, expect, afterEach, vi } from "vitest";
import { createRoot, createMemo } from "solid-js";
import { L, plural, lang, setLang } from "../i18n";

// The test environment's localStorage is not a usable Storage — stub one.
const store = new Map<string, string>();
vi.stubGlobal("localStorage", {
  getItem: (k: string) => store.get(k) ?? null,
  setItem: (k: string, v: string) => void store.set(k, v),
});

afterEach(() => setLang("ru"));

describe("i18n", () => {
  it("defaults to Russian", () => {
    expect(lang()).toBe("ru");
    expect(L("Сохранить", "Save")).toBe("Сохранить");
  });

  it("a memo calling L re-runs on a switch and the choice is stored", () => {
    createRoot((dispose) => {
      const label = createMemo(() => L("Настройки", "Settings"));
      expect(label()).toBe("Настройки");
      setLang("en");
      expect(label()).toBe("Settings");
      expect(localStorage.getItem("phaseforge.lang")).toBe("en");
      expect(document.documentElement.lang).toBe("en");
      dispose();
    });
  });

  it("plural forms per language", () => {
    const f = (n: number) => plural(n, ["полоса", "полосы", "полос"], ["band", "bands"]);
    expect([1, 2, 5, 11, 21, 22].map(f)).toEqual(["полоса", "полосы", "полос", "полос", "полоса", "полосы"]);
    setLang("en");
    expect([0, 1, 2, 21].map(f)).toEqual(["bands", "band", "bands", "bands"]);
  });
});

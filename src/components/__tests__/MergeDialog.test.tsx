// b141.105 (audit 2026-10-23 §5.23): «Авто» is computed with the same baffle
// correction (and splice) the merge applies.
import { describe, it, expect, vi, afterEach } from "vitest";
import { render } from "solid-js/web";

const invoke = vi.hoisted(() => vi.fn(async (_cmd: string, _args: any) => ({ auto_level_offset_db: 1.5, measurement: {} })));
vi.mock("@tauri-apps/api/core", () => ({ invoke }));
const picks = vi.hoisted(() => ["/m/nf.txt", "/m/ff.txt"]);
vi.mock("@tauri-apps/plugin-dialog", () => ({ open: vi.fn(async () => picks.shift()) }));
vi.mock("uplot", () => ({ default: class { destroy() {} } }));

import MergeDialog from "../MergeDialog";

let dispose: (() => void) | undefined;
afterEach(() => { dispose?.(); dispose = undefined; document.body.innerHTML = ""; });

const flush = () => new Promise((r) => setTimeout(r, 0));
const btn = (text: string) =>
  Array.from(document.querySelectorAll("button")).find((b) => b.textContent?.includes(text)) as HTMLButtonElement;

describe("MergeDialog auto offset", () => {
  it("is recomputed with the baffle once it is enabled", async () => {
    const host = document.createElement("div");
    document.body.appendChild(host);
    dispose = render(() => <MergeDialog onClose={() => {}} onMerge={() => {}} />, host);

    btn("Выбрать файл NF").click(); await flush();
    btn("Выбрать файл FF").click(); await flush(); await flush();
    const autoCalls = () => invoke.mock.calls.filter((c) => c[0] === "merge_measurements");
    expect(autoCalls().length).toBe(1);
    expect(autoCalls()[0][1].config.baffle).toBeNull();

    const baffleBox = Array.from(document.querySelectorAll("label.merge-check-label"))
      .find((l) => l.textContent?.includes("baffle"))!.querySelector("input") as HTMLInputElement;
    baffleBox.click(); await flush();
    const last = autoCalls()[autoCalls().length - 1];
    expect(autoCalls().length).toBe(2);
    expect(last[1].config.baffle).not.toBeNull();
    expect(last[1].config.level_offset_db).toBeNull();
  });

  // b141.129: the field reads 0.0 dB with no auto estimate; «Слить» sent auto.
  it("a manual field with no auto estimate sends the 0 dB it shows", async () => {
    invoke.mockImplementation(async (_c: string, a: any) => {
      if (a.config.level_offset_db === null) throw new Error("no overlap");
      return { auto_level_offset_db: 0, measurement: {} };
    });
    picks.push("/m/nf.txt", "/m/ff.txt");
    invoke.mockClear();
    const host = document.createElement("div");
    document.body.appendChild(host);
    dispose = render(() => <MergeDialog onClose={() => {}} onMerge={() => {}} />, host);
    btn("Выбрать файл NF").click(); await flush();
    btn("Выбрать файл FF").click(); await flush(); await flush();
    const manual = Array.from(document.querySelectorAll("label.merge-check-label"))
      .find((l) => l.textContent?.includes("Задать вручную"))!.querySelector("input") as HTMLInputElement;
    manual.click(); await flush();
    invoke.mockClear();
    Array.from(document.querySelectorAll("button")).find((b) => b.textContent?.trim().startsWith("Слить"))!.click();
    await flush();
    expect(invoke.mock.calls[0][1].config.level_offset_db).toBe(0);
  });
});

/** b141.103 (audit 2026-10-23 3.12, 3.14, 3.21, 3.22): project folder I/O. */
import { describe, it, expect, vi, beforeEach } from "vitest";

const calls: { cmd: string; args: any }[] = [];
let failSave = false;
vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string, args: any) => {
    calls.push({ cmd, args });
    if (cmd === "copy_into_project_unique") return "wfr_2.txt";
    if (cmd === "create_project_folder") return `${args.parentDir}/${args.projectName}`;
    if (cmd === "save_project" && failSave) throw new Error("Write error: disk full");
    return null;
  }),
}));
vi.mock("@tauri-apps/plugin-dialog", () => ({ save: vi.fn(), open: vi.fn(async () => "/parent") }));
vi.mock("../../components/MeasurementAnalysisDialog", () => ({ openMeasurementAnalysis: vi.fn() }));

import {
  deriveProjectInfo, pathBaseName, copyMeasurementToProject, saveProjectAs, resolvePrompt,
  projectDir, setProjectDir, projectName, setProjectName, setCurrentProjectPath, currentProjectPath,
  newProject,
} from "../project-io";
import { appState, setBandMeasurement, isDirty, setIsDirty } from "../../stores/bands";
import { toasts } from "../toast";

beforeEach(() => { calls.length = 0; failSave = false; });

describe("Windows paths (3.14)", () => {
  it("project dir and name from a backslash path", () => {
    expect(deriveProjectInfo("C:\\Users\\me\\Proj\\Proj.pfproj")).toEqual({ dir: "C:\\Users\\me\\Proj", name: "Proj" });
    expect(deriveProjectInfo("/Users/me/Proj/Proj.pfproj")).toEqual({ dir: "/Users/me/Proj", name: "Proj" });
    expect(pathBaseName("inbox\\wfr.txt")).toBe("wfr.txt");
  });
});

describe("same-named measurements (3.12)", () => {
  it("take the name the unique copy chose", async () => {
    setProjectDir("/p");
    expect(await copyMeasurementToProject("/R/wfr.txt", "b")).toBe("inbox/wfr_2.txt");
    const c = calls.find((x) => x.cmd === "copy_into_project_unique")!;
    expect(c.args).toEqual({ sourcePath: "/R/wfr.txt", destDir: "/p/inbox" });
    expect(calls.some((x) => x.cmd === "copy_file_to_project")).toBe(false);
  });
});

describe("Save As write failure (3.21)", () => {
  it("rolls the project folder back and keeps the measurement pending", async () => {
    setProjectDir("/old");
    setProjectName("Old");
    setCurrentProjectPath("/old/Old.pfproj");
    const id = appState.bands[0].id;
    setBandMeasurement(id, {
      name: "m", source_path: "/src/m.txt", sample_rate: 48000, freq: [20, 20000],
      magnitude: [80, 80], phase: [0, 0], metadata: { date: null, mic: null, notes: null, smoothing: null },
    } as any);
    failSave = true;
    const p = saveProjectAs();
    resolvePrompt({ name: "New", bandCount: 1 });
    await p;
    expect(projectDir()).toBe("/old");
    expect(projectName()).toBe("Old");
    expect(currentProjectPath()).toBe("/old/Old.pfproj");
    expect(appState.bands[0].measurementFile).toBeNull();
  });
});

describe("new project auto-save failure (3.22)", () => {
  it("is reported and leaves the project unsaved", async () => {
    setIsDirty(false);
    failSave = true;
    const before = toasts().length;
    const p = newProject();
    await Promise.resolve();
    resolvePrompt({ name: "Fresh", bandCount: 2 });
    await p;
    expect(isDirty()).toBe(true);
    expect(toasts().length).toBeGreaterThan(before);
    expect(toasts()[toasts().length - 1].text).toMatch(/сохранить/);
  });
});

import { invoke } from "@tauri-apps/api/core";
import { save } from "@tauri-apps/plugin-dialog";
import type { BandState } from "../stores/bands";
import {
  exportSampleRate, exportTaps, exportWindow,
  firMaxBoost, firNoiseFloor, firIterations, firFreqWeighting,
  firNarrowbandLimit, firNbSmoothingOct, firNbMaxExcess,
} from "../stores/bands";
import { projectDir, sanitize } from "./project-io";
import { evaluateBandFull } from "./band-evaluator";
import { hasActiveSubsonicProtect } from "./types";
import type { FilterConfig } from "./types";
import { pickFirRoute } from "./fir-routing";

export function driverName(b: BandState): string {
  let name = b.measurement?.name ?? b.name;
  const dot = name.indexOf("·");
  if (dot >= 0) name = name.substring(dot + 1).trim();
  return name;
}

async function generateBandImpulse(b: BandState): Promise<number[]> {
  // b139.3: route through canonical BandEvaluator. The b138.4 isLin
  // demotion (Gaussian linear + subsonic → MinimumPhase) lives inside the
  // evaluator, so this call site no longer carries duplicate phase logic.
  const result = await evaluateBandFull({
    band: b,
    fir: {
      taps: exportTaps(),
      sampleRate: exportSampleRate(),
      window: exportWindow(),
      maxBoostDb: firMaxBoost(),
      noiseFloorDb: firNoiseFloor(),
      iterations: firIterations(),
      freqWeighting: firFreqWeighting(),
      narrowbandLimit: firNarrowbandLimit(),
      nbSmoothingOct: firNbSmoothingOct(),
      nbMaxExcessDb: firNbMaxExcess(),
    },
  });
  if (!result.fir) {
    throw new Error("FIR generation failed");
  }
  return result.fir.impulse;
}

/** b141.8 (audit): WAV peak-position convention of a band's exported FIR.
 *  "centered" — peak at ~N/2 (linear-phase mains are symmetric; the IIR
 *  path zero-pads deliberately for REW import). "zero" — peak at sample 0
 *  (cepstral min-phase: Gaussian/Bessel/subsonic-protect/custom). */
export async function bandWavConvention(b: BandState): Promise<"centered" | "zero"> {
  const isUserLin = (f: FilterConfig | null | undefined) => !f || f.linear_phase === true;
  const hp = b.target.high_pass;
  const lp = b.target.low_pass;
  const linearMain = isUserLin(hp) && isUserLin(lp);
  if (linearMain) return "centered";
  const subsonicCutoff = hasActiveSubsonicProtect(hp) ? hp!.freq_hz / 8 : null;
  const route = await pickFirRoute(hp, lp, linearMain, subsonicCutoff);
  return route === "iir" ? "centered" : "zero";
}

/** When the project mixes both conventions across enabled bands, the WAVs
 *  carry an N/2-sample relative latency: loaded as-is into a convolver the
 *  crossover desynchronizes. Returns a user-facing warning, or null. */
export async function mixedWavConventionWarning(bands: BandState[]): Promise<string | null> {
  const active = bands.filter((b) => b.targetEnabled);
  if (active.length < 2) return null;
  const conv = await Promise.all(active.map(bandWavConvention));
  if (new Set(conv).size <= 1) return null;
  const centered = active.filter((_, i) => conv[i] === "centered").map(driverName).join(", ");
  const zero = active.filter((_, i) => conv[i] === "zero").map(driverName).join(", ");
  return `Внимание: FIR-файлы полос имеют разную задержку пика. ` +
    `Пик в центре (N/2): ${centered}. Пик в начале: ${zero}. ` +
    `При загрузке в конвольвер выровняйте задержки между полосами, ` +
    `иначе кроссовер рассинхронизируется на N/2 отсчётов.`;
}

/** Export active band to WAV. Returns true on success, false on cancel, throws on error.
 *  Stale PEQ is gated by a confirm dialog at higher-level call sites — keep this
 *  function focused on the export pipeline. */
export async function exportBandWav(b: BandState): Promise<boolean> {
  const impulse = await generateBandImpulse(b);
  const sr = exportSampleRate();
  const fileName = `${sanitize(driverName(b))}_${sr}_${exportTaps()}_${exportWindow()}.wav`;
  const dir = projectDir();
  if (dir) await invoke("ensure_dir", { path: `${dir}/export` }).catch(() => {});
  const defPath = dir ? `${dir}/export/${fileName}` : fileName;
  const path = await save({
    defaultPath: defPath,
    filters: [{ name: "WAV", extensions: ["wav"] }],
  });
  if (!path) return false;
  await invoke("export_fir_wav", { impulse, sampleRate: sr, path });
  return true;
}

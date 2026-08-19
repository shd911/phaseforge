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
import { buildFirGrid } from "./band-evaluator/grid";
import { showToast } from "./toast";

export function driverName(b: BandState): string {
  let name = b.measurement?.name ?? b.name;
  const dot = name.indexOf("·");
  if (dot >= 0) name = name.substring(dot + 1).trim();
  return name;
}

async function generateBandImpulse(b: BandState): Promise<{ impulse: number[]; delaySamples: number }> {
  // b139.3: route through canonical BandEvaluator. The b138.4 isLin
  // demotion (Gaussian linear + subsonic → MinimumPhase) lives inside the
  // evaluator, so this call site no longer carries duplicate phase logic.
  const result = await evaluateBandFull({
    band: b,
    // b141.19 (audit): the same grid the Export tab previews on. Without it
    // the evaluator fell back to the measurement's grid and shipped a
    // different impulse than the one on screen.
    freq: buildFirGrid(),
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
  return { impulse: result.fir.impulse, delaySamples: result.fir.wavDelaySamples };
}

// b141.14: every route pads the impulse with N/2 leading zeros, so bands
// share a latency and stay aligned in a convolver. The b141.8
// mixed-convention warning (`bandWavConvention` / `mixedWavConventionWarning`)
// is gone.

/** b141.16 (audit): residual desync check, b141.19: measured on the applied
 *  DELAY rather than the peak index. The N/2 shift is adaptive — when the
 *  impulse tail still carries content above -100 dB at N/2 (few taps + an
 *  LF/high-Q correction), the shift shrinks to avoid dropping it and the band
 *  ends up with less latency than its neighbours. The peak index cannot stand
 *  in for this: a min-phase band's peak sits past its delay by the filter's
 *  own rise time, which is a property of the filter, not a desync — reading it
 *  as one both under-reports the offset and misses it entirely when the rise
 *  time happens to fill the shortfall.
 *  Returns a user-facing warning, or null when the band carries the full N/2. */
export function offCenterWavWarning(
  delaySamples: number, taps: number, bandName: string,
): string | null {
  if (taps < 2) return null;
  const half = Math.floor(taps / 2);
  // 64 samples ≈ 1.3 ms @ 48k — below that the desync is inaudible.
  if (delaySamples >= half - 64) return null;
  const offsetSamples = half - delaySamples;
  return `Внимание: полоса «${bandName}» экспортирована с задержкой ` +
    `${delaySamples} отсчётов вместо ${half} — хвост фильтра не уместился в ` +
    `половину файла. В конвольвере она заиграет на ${offsetSamples} отсчётов ` +
    `раньше остальных: увеличьте число тапов или скомпенсируйте разницу ` +
    `задержкой в конвольвере (в WAV задержка полосы не запекается).`;
}

/** Export active band to WAV. Returns true on success, false on cancel, throws on error.
 *  Stale PEQ is gated by a confirm dialog at higher-level call sites — keep this
 *  function focused on the export pipeline. */
export async function exportBandWav(b: BandState): Promise<boolean> {
  const { impulse, delaySamples } = await generateBandImpulse(b);
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
  const warn = offCenterWavWarning(delaySamples, impulse.length, driverName(b));
  if (warn) showToast(warn, "warn", 12000);
  return true;
}

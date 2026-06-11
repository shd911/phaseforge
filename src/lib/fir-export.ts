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
import { showToast } from "./toast";

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

// b141.14: the WAV peak convention is unified — every route (linear-phase,
// IIR-analytical, cepstral min-phase) ships the impulse peak at ~N/2, so
// bands stay time-aligned in a convolver. The b141.8 mixed-convention
// warning (`bandWavConvention` / `mixedWavConventionWarning`) is gone.

/** b141.16 (audit): residual desync check. The N/2 centering shift is
 *  adaptive — when the impulse tail still carries content above -100 dB of
 *  peak at N/2 (small taps + LF/high-Q correction), the shift shrinks to
 *  avoid dropping it and the peak lands short of center. Such a WAV is
 *  correct alone but lags the other (centered) bands in a convolver.
 *  Returns a user-facing warning, or null when the peak is centered. */
export function offCenterWavWarning(impulse: number[], bandName: string): string | null {
  if (impulse.length < 2) return null;
  let peakIdx = 0, peakVal = 0;
  for (let i = 0; i < impulse.length; i++) {
    const a = Math.abs(impulse[i]);
    if (a > peakVal) { peakVal = a; peakIdx = i; }
  }
  const half = Math.floor(impulse.length / 2);
  // Raw cascade rise puts the peak a hair past N/2 — only a SHORTFALL
  // (tail-limited shift) signals desync. 64 samples ≈ 1.3 ms @ 48k.
  if (peakIdx >= half - 64) return null;
  const offsetSamples = half - peakIdx;
  return `Внимание: у полосы «${bandName}» пик импульса смещён от центра на ` +
    `${offsetSamples} отсчётов (хвост фильтра не уместился в половину файла). ` +
    `В конвольвере эта полоса заиграет раньше остальных — добавьте ей задержку ` +
    `${offsetSamples} отсчётов или увеличьте число тапов.`;
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
  const warn = offCenterWavWarning(impulse, driverName(b));
  if (warn) showToast(warn, "warn", 12000);
  return true;
}

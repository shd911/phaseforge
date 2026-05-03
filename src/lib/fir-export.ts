import { invoke } from "@tauri-apps/api/core";
import { save } from "@tauri-apps/plugin-dialog";
import type { FilterConfig, PeqBand } from "./types";
import type { BandState } from "../stores/bands";
import {
  exportSampleRate, exportTaps, exportWindow,
  firMaxBoost, firNoiseFloor, firIterations, firFreqWeighting,
  firNarrowbandLimit, firNbSmoothingOct, firNbMaxExcess,
} from "../stores/bands";
import { projectDir, sanitize } from "./project-io";
import { hasActiveSubsonicProtect } from "./band-evaluation";

export function driverName(b: BandState): string {
  let name = b.measurement?.name ?? b.name;
  const dot = name.indexOf("·");
  if (dot >= 0) name = name.substring(dot + 1).trim();
  return name;
}

async function generateBandImpulse(b: BandState): Promise<number[]> {
  const sr = exportSampleRate();
  const taps = exportTaps();
  const win = exportWindow();
  const peqBands = b.peqBands?.filter((p: PeqBand) => p.enabled) ?? [];

  const [freq, response] = await invoke<[number[], { magnitude: number[]; phase: number[] }]>(
    "evaluate_target_standalone",
    { target: { ...b.target }, nPoints: 512, fMin: 5, fMax: 40000 }
  );
  const targetMag = response.magnitude;

  let peqMagArr: number[] = [];
  if (peqBands.length > 0) {
    const [peqMag] = await invoke<[number[], number[]]>("compute_peq_complex", {
      freq, bands: peqBands, sampleRate: sr,
    });
    peqMagArr = peqMag;
  }

  // b138.4: subsonic_protect on a Gaussian HP is always min-phase, so a
  // "linear" Gaussian HP with subsonic active is no longer fully linear —
  // the FIR must run MinimumPhase mode so Rust's Hilbert sees the subsonic
  // contribution baked into target magnitude.
  const isLin = (f: FilterConfig | null | undefined) =>
    !f || (f.linear_phase && !hasActiveSubsonicProtect(f));
  const fir = await invoke<{ impulse: number[] }>("generate_model_fir", {
    freq,
    targetMag,
    peqMag: peqMagArr,
    modelPhase: new Array(freq.length).fill(0),
    config: {
      taps, sample_rate: sr,
      max_boost_db: firMaxBoost(),
      noise_floor_db: firNoiseFloor(),
      window: win,
      phase_mode: (isLin(b.target.high_pass) && isLin(b.target.low_pass)) ? "LinearPhase" : "MinimumPhase",
      iterations: firIterations(),
      freq_weighting: firFreqWeighting(),
      narrowband_limit: firNarrowbandLimit(),
      nb_smoothing_oct: firNbSmoothingOct(),
      nb_max_excess_db: firNbMaxExcess(),
    },
  });
  return fir.impulse;
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

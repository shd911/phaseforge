// Plot helpers — shared utilities for unified plot
import type { SmoothingMode } from "../stores/bands";
import type { FilterConfig } from "./types";

/**
 * Unwrap a phase array (in degrees) so adjacent samples differ by < 180°.
 * Needed when adding wrapped PEQ/XO phase to unwrapped measurement phase.
 */
export function unwrapDegrees(ph: number[]): number[] {
  if (ph.length === 0) return ph;
  const out = [ph[0]];
  let offset = 0;
  for (let i = 1; i < ph.length; i++) {
    let d = ph[i] - ph[i - 1];
    while (d > 180) { offset -= 360; d -= 360; }
    while (d < -180) { offset += 360; d += 360; }
    out.push(ph[i] + offset);
  }
  return out;
}

// --- Gaussian min-phase helpers ---

/** Check if a filter is Gaussian with min-phase (not linear-phase) */
export function isGaussianMinPhase(f: FilterConfig | null | undefined): boolean {
  return !!f && f.filter_type === "Gaussian" && !f.linear_phase;
}

/** Compute Gaussian LP magnitude in dB for a frequency array.
 *  H_LP(f) = exp(-ln(2) * (f/fc)^(2M)), converted to dB. */
function gaussianLpMagDb(freq: number[], fc: number, m: number): number[] {
  const ln2 = Math.LN2;
  return freq.map(f => {
    if (f <= 0) return 0;
    const ratio = f / fc;
    const lin = Math.exp(-ln2 * Math.pow(ratio, 2 * m));
    return lin > 1e-20 ? 20 * Math.log10(lin) : -400;
  });
}

/** Compute individual Gaussian filter magnitude in dB (HP or LP). */
export function gaussianFilterMagDb(freq: number[], f: FilterConfig, isLowPass: boolean): number[] {
  const fc = f.freq_hz;
  const m = f.shape ?? 1.0;
  return isLowPass ? gaussianLpMagDb(freq, fc, m) : gaussianHpMagDb(freq, fc, m);
}

/** Compute Gaussian HP magnitude in dB: HP = 1 - LP. */
function gaussianHpMagDb(freq: number[], fc: number, m: number): number[] {
  const ln2 = Math.LN2;
  return freq.map(f => {
    if (f <= 0) return -400;
    const ratio = f / fc;
    const lpLin = Math.exp(-ln2 * Math.pow(ratio, 2 * m));
    const hpLin = 1 - lpLin;
    return hpLin > 1e-20 ? 20 * Math.log10(hpLin) : -400;
  });
}

// --- Color constants ---
export const SUM_TARGET_COLOR = "#FFD700";
export const SUM_TARGET_PHASE_COLOR = "#B8960A";
export const SUM_CORRECTED_COLOR = "#4ADE80";
export const SUM_MEAS_COLOR = "#94A3B8";
export const FREQ_SNAP_COLORS = ["#808080", "#A855F7", "#EC4899", "#14B8A6"];

// --- Curve type colors ---
export const PEQ_COLOR = "#FF9F43";
export const PEQ_HOVER_COLOR = "#FFA726";
export const PEQ_DRAG_COLOR = "#FFC107";
export const PEQ_BASE_COLOR = "#FF9800";
export const FIR_COLOR = "#38BDF8";
export const CORRECTED_COLOR = "#22C55E";
export const MEAS_DEFAULT_COLOR = "#4A9EFF";

// --- Status thresholds use CSS vars, but fallback for canvas ---
export const STATUS_GOOD = "#22C55E";
export const STATUS_WARN = "#FFD700";
export const STATUS_BAD = "#EF4444";

// --- Default IR/GD colors (single-band mode) ---
export const DEFAULT_IR_COLORS = {
  measIr: "#4A9EFF", measStep: "#4A9EFF80",
  targetIr: "#FFD700", targetStep: "#F59E0B",
  corrIr: "#22C55E", corrStep: "#22C55E80"
};
export const DEFAULT_GD_COLORS = {
  meas: "#F59E0B", target: "#FFD700", corr: "#22C55E"
};
export const DEFAULT_EXPORT_COLORS = {
  model: "#FF9F43", fir: "#38BDF8",
  modelPhase: "#9b8060", firPhase: "#1a6e8a"
};

// --- HSL color utilities ---
export function hexToHsl(hex: string): [number, number, number] {
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  let h = 0, s = 0;
  const l = (max + min) / 2;
  if (max !== min) {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
    else if (max === g) h = ((b - r) / d + 2) / 6;
    else h = ((r - g) / d + 4) / 6;
  }
  return [h * 360, s * 100, l * 100];
}

export function hslToHex(h: number, s: number, l: number): string {
  h /= 360; s /= 100; l /= 100;
  const hue2rgb = (p: number, q: number, t: number) => {
    if (t < 0) t += 1; if (t > 1) t -= 1;
    if (t < 1/6) return p + (q - p) * 6 * t;
    if (t < 1/2) return q;
    if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
    return p;
  };
  let r: number, g: number, b: number;
  if (s === 0) { r = g = b = l; } else {
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1/3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1/3);
  }
  const toHex = (v: number) => Math.round(v * 255).toString(16).padStart(2, "0");
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

/** Derive measurement (muted), target (pastel), corrected (vivid) from base color */
export function bandColorFamily(baseHex: string) {
  const [h] = hexToHsl(baseHex);
  return {
    meas:           hslToHex(h, 30, 45) + "A0",
    measPhase:      hslToHex(h, 20, 35) + "70",
    target:         hslToHex(h, 25, 78),
    targetPhase:    hslToHex(h, 20, 65),
    corrected:      hslToHex(h, 100, 60),
    correctedPhase: hslToHex(h, 80, 42),
  };
}

// --- Smoothing config ---
export function smoothingConfig(mode: SmoothingMode): { variable: boolean; fixed_fraction: number | null } {
  if (mode === "var") return { variable: true, fixed_fraction: null };
  const fractions: Record<string, number> = { "1/3": 1/3, "1/6": 1/6, "1/12": 1/12, "1/24": 1/24 };
  return { variable: false, fixed_fraction: fractions[mode] ?? 1/6 };
}

// --- Phase wrapping (returns new array, does NOT mutate input) ---
export function wrapPhase(phase: number[]): number[] {
  return phase.map(p => {
    let w = p % 360;
    if (w > 180) w -= 360;
    else if (w < -180) w += 360;
    return w;
  });
}

// --- Frequency formatting ---
export function fmtFreq(v: number): string {
  if (v >= 1000) return (v / 1000).toFixed(2) + " kHz";
  return v.toFixed(1) + " Hz";
}

/** Compute group delay from frequency and phase arrays.
 *  τ(f) = -(1/360) · dφ/df  (seconds → milliseconds) */
export function computeGroupDelay(freq: number[], phaseDeg: number[]): { freqOut: number[]; gdMs: number[] } {
  const n = freq.length;
  if (n < 2) return { freqOut: [], gdMs: [] };
  const gd: number[] = new Array(n);
  gd[0] = -(phaseDeg[1] - phaseDeg[0]) / (360 * (freq[1] - freq[0])) * 1000;
  for (let i = 1; i < n - 1; i++) {
    const df = freq[i + 1] - freq[i - 1];
    gd[i] = df > 0 ? -(phaseDeg[i + 1] - phaseDeg[i - 1]) / (360 * df) * 1000 : 0;
  }
  gd[n - 1] = -(phaseDeg[n - 1] - phaseDeg[n - 2]) / (360 * (freq[n - 1] - freq[n - 2])) * 1000;
  return { freqOut: freq, gdMs: gd };
}

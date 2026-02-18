export interface MeasurementMetadata {
  date: string | null;
  mic: string | null;
  notes: string | null;
  smoothing: number | null;
}

export interface Measurement {
  name: string;
  source_path: string | null;
  sample_rate: number | null;
  freq: number[];
  magnitude: number[];
  phase: number[] | null;
  metadata: MeasurementMetadata;
}

export interface SmoothingConfig {
  variable: boolean;
  fixed_fraction: number | null;
}

// --- Target Curve ---

export type FilterType = "Butterworth" | "Bessel" | "LinkwitzRiley" | "Gaussian";

export interface FilterConfig {
  filter_type: FilterType;
  order: number;     // 1..8
  freq_hz: number;
  shape: number | null; // M coefficient (Gaussian only)
  linear_phase: boolean;
}

export interface TargetResponse {
  magnitude: number[];
  phase: number[];
}

export interface ShelfConfig {
  freq_hz: number;
  gain_db: number;
  q: number;
}

export interface TargetCurve {
  reference_level_db: number;
  tilt_db_per_octave: number;
  tilt_ref_freq: number;
  high_pass: FilterConfig | null;
  low_pass: FilterConfig | null;
  low_shelf: ShelfConfig | null;
  high_shelf: ShelfConfig | null;
}

// --- Impulse Response ---

export interface ImpulseResult {
  time: number[];
  impulse: number[];
  step: number[];
}

// --- Baffle Step ---

export interface BaffleConfig {
  baffle_width_m: number;
  baffle_height_m: number;
  driver_offset_x_m: number;
  driver_offset_y_m: number;
}

export interface BaffleStepPreview {
  freq: number[];
  correction_db: number[];
  f3_hz: number;
  edge_frequencies: [number, number, number, number];
}

// --- Merge NF+FF ---

export interface MergeConfig {
  splice_freq: number;
  blend_octaves: number;
  level_offset_db: number | null;
  match_range: [number, number] | null;
  baffle: BaffleConfig | null;
}

export interface MergeResult {
  measurement: Measurement;
  auto_level_offset_db: number;
  delay_diff_seconds: number;
}

// --- Auto Align: PEQ ---

export interface PeqBand {
  freq_hz: number;
  gain_db: number;
  q: number;
  enabled: boolean;
}

export interface PeqConfig {
  max_bands: number;
  tolerance_db: number;
  peak_bias: number;
  max_boost_db: number;
  max_cut_db: number;
  freq_range: [number, number];
}

export interface PeqResult {
  bands: PeqBand[];
  max_error_db: number;
  iterations: number;
}

// --- Auto Align: FIR ---

export type PhaseMode = "MinimumPhase" | "LinearPhase" | "MixedPhase";
export type WindowType = "Blackman" | "Kaiser" | "Tukey" | "Hann";

export interface FirConfig {
  taps: number;
  sample_rate: number;
  max_boost_db: number;
  noise_floor_db: number;
  window: WindowType;
  phase_mode: PhaseMode;
}

export interface FirResult {
  impulse: number[];
  time_ms: number[];
  taps: number;
  sample_rate: number;
  norm_db: number;
}

export interface FirModelResult {
  impulse: number[];
  time_ms: number[];
  realized_mag: number[];
  realized_phase: number[];
  taps: number;
  sample_rate: number;
  norm_db: number;
}

// Colors for multi-measurement overlay
export const MEASUREMENT_COLORS = [
  "#4A9EFF", // blue
  "#FF8C42", // orange
  "#22C55E", // green
  "#A855F7", // purple
  "#EF4444", // red
  "#06B6D4", // cyan
  "#F59E0B", // amber
  "#EC4899", // pink
];

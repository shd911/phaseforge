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

export type FilterType = "Butterworth" | "Bessel" | "LinkwitzRiley" | "Gaussian" | "Custom";

export interface FilterConfig {
  filter_type: FilterType;
  order: number;     // 1..8
  freq_hz: number;
  shape: number | null; // M coefficient (Gaussian only)
  linear_phase: boolean;
  q: number | null;  // Q factor (Custom only, default 0.707)
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

export type PeqFilterType = "Peaking" | "LowShelf" | "HighShelf";

export interface PeqBand {
  freq_hz: number;
  gain_db: number;
  q: number;
  enabled: boolean;
  filter_type: PeqFilterType;
}

export interface ExclusionZone {
  startHz: number;
  endHz: number;
}

// Measurement analysis (b135). Mirrors Rust src-tauri/src/analysis/mod.rs.
export type AnalysisSeverity = "Info" | "Warning" | "Error";

export type AnalysisAction =
  | { type: "SetOptLowerBound"; value: number }
  | { type: "SetOptUpperBound"; value: number }
  | { type: "AddExclusionZone"; value: { low_hz: number; high_hz: number } }
  | { type: "ApplySmoothing"; value: string };

export interface AnalysisRecommendation {
  action: AnalysisAction;
  label: string;
}

export interface AnalysisFinding {
  id: string;
  severity: AnalysisSeverity;
  title: string;
  description: string;
  freq_range: [number, number] | null;
  recommendations: AnalysisRecommendation[];
}

export interface AnalysisResult {
  timestamp: string;
  app_version: string;
  findings: AnalysisFinding[];
}

export interface PeqConfig {
  max_bands: number;
  tolerance_db: number;
  peak_bias: number;
  max_boost_db: number;
  max_cut_db: number;
  freq_range: [number, number];
  hybrid?: boolean;
  gain_regularization?: number;
}

export interface PeqResult {
  bands: PeqBand[];
  max_error_db: number;
  iterations: number;
}

// --- Auto Align: FIR ---

export type PhaseMode = "MinimumPhase" | "LinearPhase" | "MixedPhase" | "HybridPhase";
export type WindowType =
  // Basic / classical
  | "Rectangular" | "Bartlett" | "Hann" | "Hamming" | "Blackman"
  // Blackman-Harris family
  | "ExactBlackman" | "BlackmanHarris" | "Nuttall3" | "Nuttall4" | "FlatTop"
  // Parametric
  | "Kaiser" | "DolphChebyshev" | "Gaussian" | "Tukey"
  // Special
  | "Lanczos" | "Poisson" | "HannPoisson" | "Bohman" | "Cauchy" | "Riesz";

export interface FirConfig {
  taps: number;
  sample_rate: number;
  max_boost_db: number;
  noise_floor_db: number;
  window: WindowType;
  phase_mode: PhaseMode;
  iterations: number;          // iterative WLS passes (0=off, 1-10)
  freq_weighting: boolean;     // frequency-dependent WLS weights
  narrowband_limit: boolean;   // narrowband boost limiting
  nb_smoothing_oct: number;    // smoothing width in octaves
  nb_max_excess_db: number;    // max dB above smoothed curve
}

export interface FirResult {
  impulse: number[];
  time_ms: number[];
  taps: number;
  sample_rate: number;
  norm_db: number;
  causality: number;
}

export interface FirModelResult {
  impulse: number[];
  time_ms: number[];
  realized_mag: number[];
  realized_phase: number[];
  taps: number;
  sample_rate: number;
  norm_db: number;
  causality: number;
}

// Colors for multi-measurement overlay
export const MEASUREMENT_COLORS = [
  "#4A9EFF", // bright blue (like REW default)
  "#FF6B6B", // coral red
  "#A855F7", // vivid purple
  "#22C55E", // green
  "#F59E0B", // amber
  "#EC4899", // pink
  "#06B6D4", // cyan
  "#8B5CF6", // violet
];

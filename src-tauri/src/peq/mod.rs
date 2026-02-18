// Parametric EQ engine: Levenberg-Marquardt optimization
//
// Unified PEQ optimizer using LMA (Levenberg-Marquardt Algorithm) for
// simultaneous optimization of all band parameters [freq, gain, Q].
//
// Pipeline:
//   1. Compute 3-zone target: service curve outside filters, flat at max Target SPL
//      in-band, peaks clamped above LP — with cosine blends at HP/LP boundaries
//   2. Compute ERB-inspired frequency weights with null suppression
//   3. Greedy initialization → starting band placement
//   4. LMA optimization → simultaneous refinement of all parameters
//   5. Band addition for large residuals → re-optimize
//   6. Post-processing: merge + prune + sort

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use tracing::info;

use crate::dsp::{variable_smoothing, SmoothingConfig};
use crate::error::AppError;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeqBand {
    pub freq_hz: f64,
    pub gain_db: f64,
    pub q: f64,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeqConfig {
    /// Maximum number of PEQ bands to generate (default: 20)
    pub max_bands: usize,
    /// Convergence tolerance in dB (default: 1.0, range: 0.5..3.0)
    pub tolerance_db: f64,
    /// Peak bias: weight positive errors more (cuts preferred, default: 1.5)
    pub peak_bias: f64,
    /// Maximum boost allowed in dB (default: 6.0)
    pub max_boost_db: f64,
    /// Maximum cut allowed in dB (default: 18.0)
    pub max_cut_db: f64,
    /// Frequency range for PEQ operation: (f_low, f_high) from HP/LP crossover
    pub freq_range: (f64, f64),
    /// Optional fixed smoothing fraction for error curve (e.g. 1/6 octave).
    /// If None, uses variable smoothing (default PEQ behavior).
    #[serde(default)]
    pub smoothing_fraction: Option<f64>,
    /// Minimum distance between bands in octaves (default: 1/3 octave).
    /// Smaller values allow denser band placement (useful for HF correction).
    #[serde(default)]
    pub min_band_distance_oct: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeqResult {
    pub bands: Vec<PeqBand>,
    pub max_error_db: f64,
    pub iterations: u32,
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SAMPLE_RATE: f64 = 48000.0;
/// Minimum distance between two PEQ bands in octaves
const MIN_BAND_DISTANCE_OCT: f64 = 0.333; // 1/3 octave
/// Q limits
const Q_MIN: f64 = 0.5;
const Q_MAX: f64 = 10.0;

// LMA-specific constants
/// Maximum Q above LP crossover (phase-safe wide filters)
const Q_MAX_ABOVE_LP: f64 = 2.5;
/// Maximum LMA iterations per optimization round
const LMA_MAX_ITER: usize = 50;
/// LMA damping factor upper bound (stop if stuck)
const LMA_LAMBDA_MAX: f64 = 1e6;
/// LMA convergence threshold (relative step size)
const LMA_CONVERGENCE: f64 = 1e-4;
/// Minimum band gain to keep (dB)
const LMA_MIN_GAIN_DB: f64 = 0.2;
/// Band addition residual multiplier for weighted threshold
const LMA_ADD_BAND_FACTOR: f64 = 1.5;

// ===========================================================================
// LMA Core: Cholesky Solver, Weights, Target, Optimizer
// ===========================================================================

// ---------------------------------------------------------------------------
// Cholesky decomposition solver
// ---------------------------------------------------------------------------

/// Solve A·x = b via Cholesky decomposition (A must be symmetric positive definite).
/// Returns None if decomposition fails (matrix not positive definite).
/// Dense solver — max size ~60×60 for 20 bands × 3 params.
fn cholesky_solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    if n == 0 || b.len() != n {
        return None;
    }

    // Cholesky decomposition: A = L·Lᵀ
    let mut l = vec![vec![0.0_f64; n]; n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i][k] * l[j][k];
            }
            if i == j {
                let diag = a[i][i] - sum;
                if diag <= 0.0 {
                    return None; // Not positive definite
                }
                l[i][j] = diag.sqrt();
            } else {
                if l[j][j].abs() < 1e-30 {
                    return None;
                }
                l[i][j] = (a[i][j] - sum) / l[j][j];
            }
        }
    }

    // Forward substitution: L·y = b
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[i][j] * y[j];
        }
        if l[i][i].abs() < 1e-30 {
            return None;
        }
        y[i] = (b[i] - sum) / l[i][i];
    }

    // Back substitution: Lᵀ·x = y
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..n {
            sum += l[j][i] * x[j];
        }
        if l[i][i].abs() < 1e-30 {
            return None;
        }
        x[i] = (y[i] - sum) / l[i][i];
    }

    Some(x)
}

// ---------------------------------------------------------------------------
// ERB-inspired frequency weights
// ---------------------------------------------------------------------------

/// Compute psychoacoustically-weighted frequency weights for LMA optimization.
///
/// Weight bands:
/// - 20–150 Hz: 1.0  (room modes, important but limited correction potential)
/// - 150–800 Hz: 1.5  (baffle step region)
/// - 800–5000 Hz: 2.0 (maximum ear sensitivity)
/// - 5000–20000 Hz: 0.5 (broad strokes, ignore narrow peaks)
///
/// Modifiers:
/// - Above LP: ×0.6 (lower priority, correction above crossover)
/// - Deep nulls (>12 dB below median): weight → 0 (don't try to fill acoustic nulls)
fn compute_weights(freq: &[f64], meas_mag: &[f64], hp_freq: f64, lp_freq: f64) -> Vec<f64> {
    let n = freq.len();

    // Compute median level in HP–LP band for null detection
    let median = compute_median_in_range(freq, meas_mag, hp_freq, lp_freq)
        .unwrap_or(80.0);

    let mut weights = Vec::with_capacity(n);
    for i in 0..n {
        let f = freq[i];

        // Base ERB-inspired weight
        let base = if f < 150.0 {
            1.0
        } else if f < 800.0 {
            1.5
        } else if f < 5000.0 {
            2.0
        } else {
            0.5
        };

        // Above-LP modifier
        let lp_mod = if f > lp_freq { 0.6 } else { 1.0 };

        // Null suppression: deep dips below median → weight → 0
        let null_mod = if meas_mag[i] < median - 12.0 {
            0.0
        } else if meas_mag[i] < median - 8.0 {
            // Gradual fade: linearly from 0 at -12dB to 1 at -8dB
            (meas_mag[i] - (median - 12.0)) / 4.0
        } else {
            1.0
        };

        weights.push(base * lp_mod * null_mod);
    }

    weights
}

// ---------------------------------------------------------------------------
// Unified target curve
// ---------------------------------------------------------------------------

/// Compute a unified target for the entire frequency range.
///
/// - Below LP: 1/1-octave smoothed measurement (service curve — keeps broad shape)
/// - Above LP: flat at median level of HP-LP band
/// - Transition: cosine blend in LP..LP×2 (one octave)
fn compute_unified_target(
    freq: &[f64],
    meas_mag: &[f64],
    hp_freq: f64,
    lp_freq: f64,
) -> Vec<f64> {
    let n = freq.len();

    // 1. Service curve: 1/1-octave smoothed measurement
    let smooth_config = SmoothingConfig {
        variable: false,
        fixed_fraction: Some(1.0),
    };
    let smoothed = variable_smoothing(freq, meas_mag, &smooth_config);

    // 2. Flat level above LP: median of measurement in HP–LP
    let flat_level = compute_median_in_range(freq, meas_mag, hp_freq, lp_freq)
        .unwrap_or(80.0);

    // 3. Cosine blend in LP..LP×2
    let lp2 = lp_freq * 2.0;

    let mut target = Vec::with_capacity(n);
    for i in 0..n {
        let f = freq[i];
        if f <= lp_freq {
            target.push(smoothed[i]);
        } else if f >= lp2 {
            target.push(flat_level);
        } else {
            // Cosine blend: 0 at LP, 1 at LP×2
            let t = (f / lp_freq).log2(); // 0 at LP, 1 at LP×2
            let blend = 0.5 * (1.0 - (t * PI).cos()); // smooth 0→1
            target.push(smoothed[i] * (1.0 - blend) + flat_level * blend);
        }
    }

    target
}

// ---------------------------------------------------------------------------
// LMA Solver
// ---------------------------------------------------------------------------

struct LmaSolver<'a> {
    freq: &'a [f64],
    meas_mag: &'a [f64],
    target_mag: &'a [f64],
    weights: &'a [f64],
    range_indices: Vec<usize>,
    lp_freq: f64,
    config: &'a PeqConfig,
}

impl<'a> LmaSolver<'a> {
    fn new(
        freq: &'a [f64],
        meas_mag: &'a [f64],
        target_mag: &'a [f64],
        weights: &'a [f64],
        lp_freq: f64,
        config: &'a PeqConfig,
    ) -> Self {
        let range_indices: Vec<usize> = (0..freq.len())
            .filter(|&i| freq[i] >= config.freq_range.0 && freq[i] <= config.freq_range.1)
            .collect();

        Self {
            freq,
            meas_mag,
            target_mag,
            weights,
            range_indices,
            lp_freq,
            config,
        }
    }

    /// Pack bands into parameter vector θ = [f₁, G₁, Q₁, f₂, G₂, Q₂, ...]
    fn bands_to_params(bands: &[PeqBand]) -> Vec<f64> {
        let mut params = Vec::with_capacity(bands.len() * 3);
        for b in bands {
            params.push(b.freq_hz);
            params.push(b.gain_db);
            params.push(b.q);
        }
        params
    }

    /// Unpack parameter vector back to bands
    fn params_to_bands(params: &[f64]) -> Vec<PeqBand> {
        let n = params.len() / 3;
        let mut bands = Vec::with_capacity(n);
        for i in 0..n {
            bands.push(PeqBand {
                freq_hz: params[i * 3],
                gain_db: params[i * 3 + 1],
                q: params[i * 3 + 2],
                enabled: true,
            });
        }
        bands
    }

    /// Clamp parameters to valid ranges
    fn clamp_params(&self, params: &mut [f64]) {
        let n_bands = params.len() / 3;
        for i in 0..n_bands {
            params[i * 3] = params[i * 3].clamp(self.config.freq_range.0, self.config.freq_range.1);
            params[i * 3 + 1] = params[i * 3 + 1].clamp(-self.config.max_cut_db, self.config.max_boost_db);
            let q_max = if params[i * 3] > self.lp_freq {
                Q_MAX_ABOVE_LP
            } else {
                Q_MAX
            };
            params[i * 3 + 2] = params[i * 3 + 2].clamp(Q_MIN, q_max);
        }
    }

    /// Compute weighted residual vector: r_i = √W(f_i) × √bias × (meas[i] + correction[i] - target[i])
    fn compute_residuals(&self, params: &[f64]) -> Vec<f64> {
        let bands = Self::params_to_bands(params);
        let correction = apply_peq(self.freq, &bands, SAMPLE_RATE);

        let mut residuals = Vec::with_capacity(self.range_indices.len());
        for &i in &self.range_indices {
            let err = self.meas_mag[i] + correction[i] - self.target_mag[i];
            let bias = if err > 0.0 {
                self.config.peak_bias.sqrt()
            } else {
                1.0
            };
            let w = self.weights[i].sqrt();
            residuals.push(w * bias * err);
        }

        residuals
    }

    /// Compute total cost = Σ r_i² + Q penalties
    fn compute_cost(&self, params: &[f64]) -> f64 {
        let residuals = self.compute_residuals(params);
        let mut cost: f64 = residuals.iter().map(|r| r * r).sum();

        // Q penalty: penalize Q > Q_MAX_ABOVE_LP for above-LP bands
        let n_bands = params.len() / 3;
        let m = self.range_indices.len().max(1) as f64;
        for i in 0..n_bands {
            let freq = params[i * 3];
            let q = params[i * 3 + 2];
            let gain = params[i * 3 + 1];
            // Q penalty above LP: gentle quadratic penalty for Q > 1.5
            if freq > self.lp_freq && q > 1.5 {
                cost += (q - 1.5).powi(2) * gain.abs() * 0.5 * m;
            }
            // General Q penalty for very high Q in-band
            if q > 8.0 {
                cost += (q - 8.0).powi(2) * 0.1 * m;
            }
        }

        cost
    }

    /// Compute Jacobian via numerical finite differences
    fn compute_jacobian(&self, params: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>) {
        let n_params = params.len();
        let r0 = self.compute_residuals(params);
        let m = r0.len();

        let mut jacobian = vec![vec![0.0_f64; n_params]; m];

        for p in 0..n_params {
            // Step size: relative for freq, absolute for gain and Q
            let h = match p % 3 {
                0 => params[p] * 1e-3,    // freq: relative
                1 => 0.01,                 // gain: 0.01 dB
                2 => 0.01,                 // Q: 0.01
                _ => unreachable!(),
            };
            if h.abs() < 1e-12 {
                continue; // Skip zero parameters
            }

            let mut params_h = params.to_vec();
            params_h[p] += h;

            let r_h = self.compute_residuals(&params_h);

            for i in 0..m {
                jacobian[i][p] = (r_h[i] - r0[i]) / h;
            }
        }

        (jacobian, r0)
    }

    /// Run LMA optimization on the given bands.
    /// Returns (optimized bands, iteration count).
    fn optimize(&self, initial_bands: &[PeqBand]) -> (Vec<PeqBand>, u32) {
        if initial_bands.is_empty() {
            return (Vec::new(), 0);
        }

        let mut params = Self::bands_to_params(initial_bands);
        self.clamp_params(&mut params);

        let mut lambda = 1.0_f64; // LMA damping factor
        let mut cost = self.compute_cost(&params);
        let mut total_iters = 0u32;

        for iter in 0..LMA_MAX_ITER {
            total_iters = iter as u32 + 1;
            if lambda > LMA_LAMBDA_MAX {
                info!("LMA: lambda exceeded max at iter {}, stopping", iter);
                break;
            }

            let (jacobian, residuals) = self.compute_jacobian(&params);
            let n_params = params.len();
            let m = residuals.len();

            // H = Jᵀ·J  (approximate Hessian, n_params × n_params)
            let mut h = vec![vec![0.0_f64; n_params]; n_params];
            for i in 0..n_params {
                for j in 0..=i {
                    let mut sum = 0.0;
                    for k in 0..m {
                        sum += jacobian[k][i] * jacobian[k][j];
                    }
                    h[i][j] = sum;
                    h[j][i] = sum;
                }
            }

            // g = Jᵀ·r  (gradient, n_params)
            let mut g = vec![0.0_f64; n_params];
            for i in 0..n_params {
                let mut sum = 0.0;
                for k in 0..m {
                    sum += jacobian[k][i] * residuals[k];
                }
                g[i] = sum;
            }

            // Damped normal equations: (H + λ·diag(H))·δ = −g
            let mut h_damped = h.clone();
            for i in 0..n_params {
                h_damped[i][i] += lambda * h[i][i].max(1e-8);
            }

            let neg_g: Vec<f64> = g.iter().map(|&gi| -gi).collect();

            let delta = match cholesky_solve(&h_damped, &neg_g) {
                Some(d) => d,
                None => {
                    // Cholesky failed — increase damping
                    lambda *= 4.0;
                    continue;
                }
            };

            // Check convergence: ‖δ‖ / ‖θ‖
            let delta_norm: f64 = delta.iter().map(|d| d * d).sum::<f64>().sqrt();
            let params_norm: f64 = params.iter().map(|p| p * p).sum::<f64>().sqrt().max(1e-10);
            if delta_norm / params_norm < LMA_CONVERGENCE {
                info!("LMA: converged at iter {} (rel step {:.2e})", iter, delta_norm / params_norm);
                break;
            }

            // Trial step
            let mut params_new: Vec<f64> = params.iter().zip(&delta).map(|(p, d)| p + d).collect();
            self.clamp_params(&mut params_new);

            let cost_new = self.compute_cost(&params_new);

            if cost_new < cost {
                // Accept step
                params = params_new;
                cost = cost_new;
                lambda *= 0.5;
                lambda = lambda.max(1e-10);
            } else {
                // Reject step — increase damping
                lambda *= 2.0;
            }
        }

        (Self::params_to_bands(&params), total_iters)
    }
}

// ---------------------------------------------------------------------------
// Main unified optimizer
// ---------------------------------------------------------------------------

/// Unified PEQ optimizer using Levenberg-Marquardt Algorithm.
///
/// Handles in-band and above-LP correction in a single pass.
/// When `target_mag` is Some, builds a 3-zone composite target:
///   - Below HP: 1/1-octave smoothed measurement (service curve)
///   - HP..LP:   FLAT at max SPL of Target Curve (straight line, no tilt)
///   - Above LP: smoothed measurement with peaks clamped to that flat level
/// Cosine blends at HP and LP boundaries (1 octave each).
/// The tilt/shelves from Target Curve are handled later by FIR.
/// If `target_mag` is None, uses compute_unified_target() fallback.
pub fn auto_peq_lma(
    meas_mag: &[f64],
    target_mag: Option<&[f64]>,
    freq: &[f64],
    config: &PeqConfig,
    hp_freq: f64,
    lp_freq: f64,
) -> Result<PeqResult, AppError> {
    let n = freq.len();
    if n == 0 || meas_mag.len() != n {
        return Err(AppError::Dsp {
            message: format!(
                "auto_peq_lma: length mismatch: freq={}, meas={}",
                n, meas_mag.len()
            ),
        });
    }
    if config.freq_range.0 >= config.freq_range.1 {
        return Err(AppError::Config {
            message: format!(
                "auto_peq_lma: invalid freq_range: ({}, {})",
                config.freq_range.0, config.freq_range.1
            ),
        });
    }

    info!(
        "auto_peq_lma: {} points, hp={:.0}, lp={:.0}, range=({:.0}, {:.0}), max_bands={}",
        n, hp_freq, lp_freq, config.freq_range.0, config.freq_range.1, config.max_bands
    );

    // 1. Target: 3-zone composite
    //    - Below HP: 1/1-octave smoothed measurement (service curve)
    //    - HP..LP:   FLAT at max SPL of Target Curve in HP..LP ("прямая" АЧХ,
    //                не заваливаем к тильту — минфазовый раздел сделает FIR)
    //    - Above LP: smoothed measurement, peaks clamped to that flat level
    //    Cosine blends at HP and LP boundaries (1 octave each).
    let computed_target;
    let target = match target_mag {
        Some(t) => {
            if t.len() != n {
                return Err(AppError::Dsp {
                    message: format!("auto_peq_lma: target length {} != freq length {}", t.len(), n),
                });
            }

            // Service curve: 1/1-octave smoothed measurement
            let smooth_1oct = SmoothingConfig {
                variable: false,
                fixed_fraction: Some(1.0),
            };
            let service = variable_smoothing(freq, meas_mag, &smooth_1oct);

            // Flat level = max SPL of Target Curve between HP and LP
            // This gives us a "straight line" target — PEQ flattens the response,
            // the actual target slope/tilt is handled by FIR later.
            let flat_level = freq.iter().zip(t.iter())
                .filter(|(&f, _)| f >= hp_freq && f <= lp_freq)
                .map(|(_, &v)| v)
                .fold(f64::NEG_INFINITY, f64::max);
            let flat_level = if flat_level.is_finite() { flat_level } else { 80.0 };

            info!(
                "auto_peq_lma: flat target level = {:.1} dB (max of Target Curve in {:.0}..{:.0} Hz)",
                flat_level, hp_freq, lp_freq
            );

            // Blend zones (1 octave each)
            let hp_lo = hp_freq / 2.0; // blend start below HP
            let lp_hi = lp_freq * 2.0; // blend end above LP

            computed_target = (0..n).map(|i| {
                let f = freq[i];
                if f <= hp_lo {
                    // Well below HP: pure service curve
                    service[i]
                } else if f < hp_freq {
                    // Blend zone HP/2..HP: service → flat
                    let blend_t = (f / hp_lo).log2() / (hp_freq / hp_lo).log2(); // 0→1
                    let blend = 0.5 * (1.0 - (blend_t * PI).cos());
                    service[i] * (1.0 - blend) + flat_level * blend
                } else if f <= lp_freq {
                    // In-band HP..LP: flat at max SPL level
                    flat_level
                } else if f < lp_hi {
                    // Blend zone LP..LP×2: flat → clamped service
                    let above_target = service[i].min(flat_level);
                    let blend_t = (f / lp_freq).log2(); // 0→1 over one octave
                    let blend = 0.5 * (1.0 - (blend_t * PI).cos());
                    flat_level * (1.0 - blend) + above_target * blend
                } else {
                    // Well above LP: smoothed measurement clamped to flat level
                    service[i].min(flat_level)
                }
            }).collect::<Vec<_>>();
            &computed_target
        }
        None => {
            computed_target = compute_unified_target(freq, meas_mag, hp_freq, lp_freq);
            &computed_target
        }
    };

    // 2. Weights
    let weights = compute_weights(freq, meas_mag, hp_freq, lp_freq);

    // 3. Greedy initialization
    let raw_error: Vec<f64> = meas_mag.iter().zip(target.iter()).map(|(m, t)| m - t).collect();
    let smooth_config = if let Some(frac) = config.smoothing_fraction {
        SmoothingConfig {
            variable: false,
            fixed_fraction: Some(frac),
        }
    } else {
        SmoothingConfig {
            variable: true,
            fixed_fraction: None,
        }
    };
    let smoothed_error = variable_smoothing(freq, &raw_error, &smooth_config);

    let mut bands = greedy_fit_adaptive(freq, &smoothed_error, config);

    // Enforce Q constraints: Q ≤ 2.5 for bands above LP
    for b in &mut bands {
        if b.freq_hz > lp_freq && b.q > Q_MAX_ABOVE_LP {
            b.q = Q_MAX_ABOVE_LP;
        }
    }

    info!("auto_peq_lma: greedy init → {} bands", bands.len());

    // 4. Merge nearby bands
    let merge_dist = config.min_band_distance_oct.unwrap_or(MIN_BAND_DISTANCE_OCT);
    merge_nearby_bands(&mut bands, merge_dist);

    // 5. LMA optimization
    let solver = LmaSolver::new(freq, meas_mag, target, &weights, lp_freq, config);
    let mut total_iterations = 0u32;

    if !bands.is_empty() {
        let (opt_bands, iters) = solver.optimize(&bands);
        bands = opt_bands;
        total_iterations += iters;
    }

    info!("auto_peq_lma: after first LMA → {} bands, {} iters", bands.len(), total_iterations);

    // 6. Band addition loop: check for large residuals, add bands, re-optimize
    for add_round in 0..3 {
        if bands.len() >= config.max_bands {
            break;
        }

        // Find largest weighted residual
        let correction = apply_peq(freq, &bands, SAMPLE_RATE);
        let mut worst_idx = 0;
        let mut worst_val = 0.0_f64;
        let mut available = vec![true; n];

        // Mark zones around existing bands as unavailable
        for b in &bands {
            mark_exclusion_zone(freq, b.freq_hz, merge_dist, &mut available);
        }

        for i in 0..n {
            if freq[i] < config.freq_range.0 || freq[i] > config.freq_range.1 || !available[i] {
                continue;
            }
            let err = meas_mag[i] + correction[i] - target[i];
            let w_err = err.abs() * weights[i].sqrt();
            // Bias: peaks (positive error) are more important
            let biased = if err > 0.0 { w_err * config.peak_bias } else { w_err };
            if biased > worst_val {
                worst_val = biased;
                worst_idx = i;
            }
        }

        // Raw error at worst point
        let raw_err = meas_mag[worst_idx] + correction[worst_idx] - target[worst_idx];

        if raw_err.abs() < config.tolerance_db * LMA_ADD_BAND_FACTOR {
            info!("auto_peq_lma: add round {}: residual {:.1} dB < threshold, stopping", add_round, raw_err.abs());
            break;
        }

        // Add a new band at worst residual
        let new_fc = freq[worst_idx];
        let new_gain = (-raw_err).clamp(-config.max_cut_db, config.max_boost_db);
        let new_q = if new_fc > lp_freq {
            1.0_f64 // wide filter above LP
        } else {
            estimate_q_from_peak_width(freq, &{
                let c = apply_peq(freq, &bands, SAMPLE_RATE);
                meas_mag.iter().zip(target.iter()).zip(c.iter()).map(|((m, t), cr)| m + cr - t).collect::<Vec<_>>()
            }, worst_idx).min(Q_MAX)
        };

        bands.push(PeqBand {
            freq_hz: new_fc,
            gain_db: new_gain,
            q: new_q,
            enabled: true,
        });

        info!(
            "auto_peq_lma: add round {}: added band at {:.0} Hz, gain={:.1} dB, Q={:.1}",
            add_round, new_fc, new_gain, new_q
        );

        // Re-optimize all bands together
        let (opt_bands, iters) = solver.optimize(&bands);
        bands = opt_bands;
        total_iterations += iters;
    }

    // 7. Post-processing
    // Merge nearby bands that LMA may have pushed together
    merge_nearby_bands(&mut bands, merge_dist * 0.8);

    // Remove weak bands
    bands.retain(|b| b.gain_db.abs() >= LMA_MIN_GAIN_DB);

    // Sort by frequency
    bands.sort_by(|a, b| a.freq_hz.partial_cmp(&b.freq_hz).unwrap_or(std::cmp::Ordering::Equal));

    // Final Q enforcement
    for b in &mut bands {
        if b.freq_hz > lp_freq {
            b.q = b.q.clamp(Q_MIN, Q_MAX_ABOVE_LP);
        } else {
            b.q = b.q.clamp(Q_MIN, Q_MAX);
        }
    }

    // Compute final max error
    let correction = apply_peq(freq, &bands, SAMPLE_RATE);
    let max_err = compute_max_error_in_range(freq, meas_mag, target, &correction, config.freq_range);

    info!(
        "auto_peq_lma: result: {} bands, max_error={:.1} dB",
        bands.len(), max_err
    );
    for (i, b) in bands.iter().enumerate() {
        info!(
            "  band[{}]: freq={:.0} Hz, gain={:.1} dB, Q={:.2}",
            i, b.freq_hz, b.gain_db, b.q
        );
    }

    Ok(PeqResult {
        bands,
        max_error_db: max_err,
        iterations: total_iterations,
    })
}

// ---------------------------------------------------------------------------
// Public API (backward-compatible wrappers + direct functions)
// ---------------------------------------------------------------------------

/// Auto-fit PEQ bands to minimize error between measurement and target.
/// Backward-compatible wrapper: delegates to unified LMA optimizer.
pub fn auto_peq(
    meas_mag: &[f64],
    target_mag: &[f64],
    freq: &[f64],
    config: &PeqConfig,
) -> Result<PeqResult, AppError> {
    info!("auto_peq: delegating to auto_peq_lma");
    auto_peq_lma(
        meas_mag,
        Some(target_mag),
        freq,
        config,
        config.freq_range.0,
        config.freq_range.1,
    )
}

/// Compute the combined magnitude response (dB) of all PEQ bands at each frequency point.
/// Bands with `enabled == false` are skipped.
pub fn apply_peq(freq: &[f64], bands: &[PeqBand], sample_rate: f64) -> Vec<f64> {
    let n = freq.len();
    let mut total = vec![0.0_f64; n];
    for band in bands {
        if !band.enabled {
            continue;
        }
        let response = peq_band_response(freq, band, sample_rate);
        for i in 0..n {
            total[i] += response[i];
        }
    }
    total
}

/// Compute the magnitude response (dB) of a single PEQ band at each frequency point.
pub fn peq_band_response(freq: &[f64], band: &PeqBand, sample_rate: f64) -> Vec<f64> {
    freq.iter()
        .map(|&f| biquad_peaking_mag_db(f, band.freq_hz, band.gain_db, band.q, sample_rate))
        .collect()
}

/// Compute the combined complex response (magnitude dB + phase degrees) of all PEQ bands.
/// Bands with `enabled == false` are skipped.
pub fn apply_peq_complex(freq: &[f64], bands: &[PeqBand], sample_rate: f64) -> (Vec<f64>, Vec<f64>) {
    let n = freq.len();
    let mut total_mag = vec![0.0_f64; n];
    let mut total_phase = vec![0.0_f64; n];
    for band in bands {
        if !band.enabled {
            continue;
        }
        for (i, &f) in freq.iter().enumerate() {
            let (mag_db, phase_deg) = biquad_peaking_complex(f, band.freq_hz, band.gain_db, band.q, sample_rate);
            total_mag[i] += mag_db;
            total_phase[i] += phase_deg;
        }
    }
    // Wrap phase to [-180°, 180°] (REW convention)
    for p in total_phase.iter_mut() {
        *p = *p % 360.0;
        if *p > 180.0 { *p -= 360.0; }
        else if *p < -180.0 { *p += 360.0; }
    }

    (total_mag, total_phase)
}

/// Compute both magnitude (dB) and phase (degrees) of a single peaking biquad.
fn biquad_peaking_complex(f: f64, fc: f64, gain_db: f64, q: f64, sample_rate: f64) -> (f64, f64) {
    if gain_db.abs() < 1e-10 || q <= 0.0 || fc <= 0.0 || sample_rate <= 0.0 {
        return (0.0, 0.0);
    }

    let w0 = 2.0 * PI * fc / sample_rate;
    let a_lin = 10.0_f64.powf(gain_db / 40.0);
    let alpha = w0.sin() / (2.0 * q);

    let b0 = 1.0 + alpha * a_lin;
    let b1 = -2.0 * w0.cos();
    let b2 = 1.0 - alpha * a_lin;
    let a0 = 1.0 + alpha / a_lin;
    let a1 = -2.0 * w0.cos();
    let a2 = 1.0 - alpha / a_lin;

    let w = 2.0 * PI * f / sample_rate;
    let cos_w = w.cos();
    let cos_2w = (2.0 * w).cos();
    let sin_w = w.sin();
    let sin_2w = (2.0 * w).sin();

    let num_re = b0 + b1 * cos_w + b2 * cos_2w;
    let num_im = -b1 * sin_w - b2 * sin_2w;
    let den_re = a0 + a1 * cos_w + a2 * cos_2w;
    let den_im = -a1 * sin_w - a2 * sin_2w;

    let num_mag_sq = num_re * num_re + num_im * num_im;
    let den_mag_sq = den_re * den_re + den_im * den_im;

    let mag_db = if den_mag_sq < 1e-30 {
        0.0
    } else {
        10.0 * (num_mag_sq / den_mag_sq).log10()
    };

    let num_phase = num_im.atan2(num_re);
    let den_phase = den_im.atan2(den_re);
    let phase_rad = num_phase - den_phase;
    let phase_deg = phase_rad * 180.0 / PI;

    (mag_db, phase_deg)
}

// ---------------------------------------------------------------------------
// Internal: Greedy Fitting with Adaptive Q
// ---------------------------------------------------------------------------

/// Greedy fitting with adaptive Q estimated from error curve peak width,
/// and exclusion zones to prevent duplicate bands.
fn greedy_fit_adaptive(freq: &[f64], smoothed_error: &[f64], config: &PeqConfig) -> Vec<PeqBand> {
    let n = freq.len();
    let mut bands: Vec<PeqBand> = Vec::new();
    let mut current_error = smoothed_error.to_vec();
    // Exclusion mask: true = available, false = excluded
    let mut available = vec![true; n];
    let exclusion_oct = config.min_band_distance_oct.unwrap_or(MIN_BAND_DISTANCE_OCT);

    for _ in 0..config.max_bands {
        // Find the peak error within frequency range, excluding masked points
        let peak = find_peak_error_masked(freq, &current_error, config.peak_bias, config.freq_range, &available);
        let (peak_idx, peak_val) = match peak {
            Some(p) => p,
            None => break,
        };

        // Convergence check
        if peak_val.abs() < config.tolerance_db {
            break;
        }

        let peak_freq = freq[peak_idx];

        // Estimate Q from the width of the error peak (adaptive)
        let q = estimate_q_from_peak_width(freq, &current_error, peak_idx);

        // Gain = negative of error (to correct it), clamped to limits
        let mut gain = -peak_val;
        if gain > 0.0 {
            gain = gain.min(config.max_boost_db);
        } else {
            gain = gain.max(-config.max_cut_db);
        }

        let band = PeqBand {
            freq_hz: peak_freq,
            gain_db: gain,
            q,
            enabled: true,
        };

        // Apply this band's response to the working error
        let response = peq_band_response(freq, &band, SAMPLE_RATE);
        for i in 0..n {
            current_error[i] += response[i];
        }

        // Mark exclusion zone around placed band
        mark_exclusion_zone(freq, peak_freq, exclusion_oct, &mut available);

        bands.push(band);
    }

    bands
}

/// Estimate Q from the half-power width of the error peak.
///
/// Measures how many octaves the error peak extends at half its amplitude,
/// then converts to Q: Q ≈ fc / bandwidth_hz
fn estimate_q_from_peak_width(freq: &[f64], error: &[f64], peak_idx: usize) -> f64 {
    let peak_val = error[peak_idx].abs();
    let half_val = peak_val * 0.5; // -6 dB (half amplitude in dB)
    let peak_freq = freq[peak_idx];

    // Search left for half-amplitude crossing
    let mut left_freq = freq[0];
    for i in (0..peak_idx).rev() {
        if error[i].abs() <= half_val {
            // Interpolate between i and i+1
            let f0 = freq[i];
            let f1 = freq[i + 1];
            let e0 = error[i].abs();
            let e1 = error[i + 1].abs();
            let t = if (e1 - e0).abs() > 1e-10 {
                (half_val - e0) / (e1 - e0)
            } else {
                0.5
            };
            left_freq = f0 + t * (f1 - f0);
            break;
        }
    }

    // Search right for half-amplitude crossing
    let mut right_freq = freq[freq.len() - 1];
    for i in (peak_idx + 1)..freq.len() {
        if error[i].abs() <= half_val {
            let f0 = freq[i - 1];
            let f1 = freq[i];
            let e0 = error[i - 1].abs();
            let e1 = error[i].abs();
            let t = if (e0 - e1).abs() > 1e-10 {
                (e0 - half_val) / (e0 - e1)
            } else {
                0.5
            };
            right_freq = f0 + t * (f1 - f0);
            break;
        }
    }

    // Bandwidth in Hz
    let bw_hz = (right_freq - left_freq).max(1.0);

    // Q = fc / bw (clamped)
    let q = (peak_freq / bw_hz).clamp(Q_MIN, Q_MAX);
    q
}

/// Mark an exclusion zone around a frequency (±octaves distance).
fn mark_exclusion_zone(freq: &[f64], center_freq: f64, octaves: f64, available: &mut [bool]) {
    let ratio = 2.0_f64.powf(octaves);
    let f_low = center_freq / ratio;
    let f_high = center_freq * ratio;
    for (i, &f) in freq.iter().enumerate() {
        if f >= f_low && f <= f_high {
            available[i] = false;
        }
    }
}

/// Find peak error with exclusion mask.
fn find_peak_error_masked(
    freq: &[f64],
    error: &[f64],
    peak_bias: f64,
    freq_range: (f64, f64),
    available: &[bool],
) -> Option<(usize, f64)> {
    let mut best_idx = 0;
    let mut best_weighted = 0.0_f64;
    let mut best_raw = 0.0_f64;
    let mut found = false;

    for (i, (&f, &e)) in freq.iter().zip(error).enumerate() {
        if f < freq_range.0 || f > freq_range.1 || !available[i] {
            continue;
        }
        let weighted = if e > 0.0 { e * peak_bias } else { e.abs() };
        if !found || weighted > best_weighted {
            best_weighted = weighted;
            best_raw = e;
            best_idx = i;
            found = true;
        }
    }

    if found && best_raw.abs() > 0.01 {
        Some((best_idx, best_raw))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Internal: Merge Nearby Bands
// ---------------------------------------------------------------------------

/// Merge PEQ bands that are within `merge_distance_oct` octaves of each other.
/// Combines them by weighted-average frequency and summed gain.
/// Keeps the wider Q (lower Q value).
fn merge_nearby_bands(bands: &mut Vec<PeqBand>, merge_distance_oct: f64) {
    if bands.len() < 2 {
        return;
    }

    // Sort by frequency for merging
    bands.sort_by(|a, b| a.freq_hz.partial_cmp(&b.freq_hz).unwrap_or(std::cmp::Ordering::Equal));

    let mut merged: Vec<PeqBand> = Vec::with_capacity(bands.len());
    let mut i = 0;

    while i < bands.len() {
        let mut group_freq_sum = bands[i].freq_hz * bands[i].gain_db.abs();
        let mut group_gain = bands[i].gain_db;
        let mut group_weight = bands[i].gain_db.abs();
        let mut group_q_min = bands[i].q; // keep widest Q (lowest value)
        let mut j = i + 1;

        while j < bands.len() {
            let octave_dist = (bands[j].freq_hz / bands[i].freq_hz).log2().abs();
            // Also check against the current group center
            let group_center = if group_weight > 0.0 {
                group_freq_sum / group_weight
            } else {
                bands[i].freq_hz
            };
            let dist_to_center = (bands[j].freq_hz / group_center).log2().abs();

            if octave_dist <= merge_distance_oct || dist_to_center <= merge_distance_oct {
                // Same-sign gains → sum; opposite-sign → sum (they partially cancel, that's OK)
                group_gain += bands[j].gain_db;
                group_freq_sum += bands[j].freq_hz * bands[j].gain_db.abs();
                group_weight += bands[j].gain_db.abs();
                if bands[j].q < group_q_min {
                    group_q_min = bands[j].q;
                }
                j += 1;
            } else {
                break;
            }
        }

        // Weighted average frequency
        let merged_freq = if group_weight > 0.0 {
            group_freq_sum / group_weight
        } else {
            bands[i].freq_hz
        };

        merged.push(PeqBand {
            freq_hz: merged_freq,
            gain_db: group_gain,
            q: group_q_min.clamp(Q_MIN, Q_MAX),
            enabled: true,
        });

        i = j;
    }

    *bands = merged;
}


// ---------------------------------------------------------------------------
// Internal: Helpers
// ---------------------------------------------------------------------------

/// Compute the maximum absolute error in the frequency range after applying correction.
fn compute_max_error_in_range(
    freq: &[f64],
    meas_mag: &[f64],
    target_mag: &[f64],
    correction: &[f64],
    freq_range: (f64, f64),
) -> f64 {
    let mut max_err = 0.0_f64;
    for (i, &f) in freq.iter().enumerate() {
        if f < freq_range.0 || f > freq_range.1 {
            continue;
        }
        let corrected = meas_mag[i] + correction[i];
        let err = (corrected - target_mag[i]).abs();
        if err > max_err {
            max_err = err;
        }
    }
    max_err
}

// ---------------------------------------------------------------------------
// Internal: Biquad Peaking EQ (RBJ Audio EQ Cookbook)
// ---------------------------------------------------------------------------

fn biquad_peaking_mag_db(f: f64, fc: f64, gain_db: f64, q: f64, sample_rate: f64) -> f64 {
    if gain_db.abs() < 1e-10 || q <= 0.0 || fc <= 0.0 || sample_rate <= 0.0 {
        return 0.0;
    }

    let w0 = 2.0 * PI * fc / sample_rate;
    let a_lin = 10.0_f64.powf(gain_db / 40.0);
    let alpha = w0.sin() / (2.0 * q);

    let b0 = 1.0 + alpha * a_lin;
    let b1 = -2.0 * w0.cos();
    let b2 = 1.0 - alpha * a_lin;
    let a0 = 1.0 + alpha / a_lin;
    let a1 = -2.0 * w0.cos();
    let a2 = 1.0 - alpha / a_lin;

    let w = 2.0 * PI * f / sample_rate;
    let cos_w = w.cos();
    let cos_2w = (2.0 * w).cos();
    let sin_w = w.sin();
    let sin_2w = (2.0 * w).sin();

    let num_re = b0 + b1 * cos_w + b2 * cos_2w;
    let num_im = -b1 * sin_w - b2 * sin_2w;
    let den_re = a0 + a1 * cos_w + a2 * cos_2w;
    let den_im = -a1 * sin_w - a2 * sin_2w;

    let num_mag_sq = num_re * num_re + num_im * num_im;
    let den_mag_sq = den_re * den_re + den_im * den_im;

    if den_mag_sq < 1e-30 {
        return 0.0;
    }

    10.0 * (num_mag_sq / den_mag_sq).log10()
}

// ---------------------------------------------------------------------------
// Public API: Above-LP Peak Reduction (Algorithm 2)
// ---------------------------------------------------------------------------

/// Compute median magnitude (dB) in a frequency range [f_low, f_high].
/// Returns None if no data points fall within the range.
pub fn compute_median_in_range(freq: &[f64], mag: &[f64], f_low: f64, f_high: f64) -> Option<f64> {
    let mut values: Vec<f64> = freq
        .iter()
        .zip(mag)
        .filter(|(&f, _)| f >= f_low && f <= f_high)
        .map(|(_, &m)| m)
        .collect();
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    Some(if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    })
}

/// Auto-PEQ above LP frequency: cut peaks above LP down to median in-band level.
/// Backward-compatible wrapper: delegates to unified LMA optimizer with cuts-only config.
pub fn auto_peq_above_lp(
    meas_mag: &[f64],
    freq: &[f64],
    config: &PeqConfig,
    lp_freq: f64,
    hp_freq: f64,
) -> Result<PeqResult, AppError> {
    info!("auto_peq_above_lp: delegating to auto_peq_lma");

    let n = freq.len();
    if n == 0 || meas_mag.len() != n {
        return Err(AppError::Dsp {
            message: format!(
                "auto_peq_above_lp: length mismatch: freq={}, meas={}",
                n, meas_mag.len()
            ),
        });
    }

    // Compute flat target at median level above LP
    let ref_level = compute_median_in_range(freq, meas_mag, hp_freq, lp_freq).ok_or_else(
        || AppError::Dsp {
            message: "auto_peq_above_lp: no data in HP-LP range for median".into(),
        },
    )?;

    let f_max = freq.last().copied().unwrap_or(20000.0).min(20000.0);

    // Target: flat at ref_level above LP, smoothed measurement below
    let smooth_cfg = SmoothingConfig {
        variable: false,
        fixed_fraction: Some(1.0 / 6.0),
    };
    let smoothed = variable_smoothing(freq, meas_mag, &smooth_cfg);
    let target: Vec<f64> = (0..n)
        .map(|i| if freq[i] > lp_freq { ref_level } else { smoothed[i] })
        .collect();

    // Configure for above-LP: cuts only (max_boost_db = 0), range from HP up
    let above_config = PeqConfig {
        max_bands: config.max_bands,
        tolerance_db: config.tolerance_db,
        peak_bias: config.peak_bias,
        max_boost_db: 0.0, // Cuts only
        max_cut_db: config.max_cut_db,
        freq_range: (lp_freq * 0.85, f_max), // start slightly below LP for wide filter overlap
        smoothing_fraction: Some(1.0 / 3.0), // 1/3 octave smoothing for above-LP
        min_band_distance_oct: config.min_band_distance_oct,
    };

    auto_peq_lma(meas_mag, Some(&target), freq, &above_config, hp_freq, lp_freq)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_log_freq(n: usize, f_min: f64, f_max: f64) -> Vec<f64> {
        let log_min = f_min.ln();
        let log_max = f_max.ln();
        (0..n)
            .map(|i| (log_min + (log_max - log_min) * i as f64 / (n - 1) as f64).exp())
            .collect()
    }

    #[test]
    fn test_biquad_at_center_frequency() {
        let gain = 6.0;
        let fc = 1000.0;
        let q = 4.0;
        let sr = 48000.0;
        let result = biquad_peaking_mag_db(fc, fc, gain, q, sr);
        assert!(
            (result - gain).abs() < 0.01,
            "At center freq, gain should be {gain} dB, got {result}"
        );
    }

    #[test]
    fn test_biquad_at_center_frequency_negative_gain() {
        let gain = -8.0;
        let fc = 500.0;
        let q = 3.0;
        let sr = 48000.0;
        let result = biquad_peaking_mag_db(fc, fc, gain, q, sr);
        assert!(
            (result - gain).abs() < 0.01,
            "At center freq, gain should be {gain} dB, got {result}"
        );
    }

    #[test]
    fn test_biquad_far_from_center() {
        let result = biquad_peaking_mag_db(100.0, 10000.0, 10.0, 5.0, 48000.0);
        assert!(
            result.abs() < 0.5,
            "Far from center, gain should be ~0 dB, got {result}"
        );
    }

    #[test]
    fn test_biquad_zero_gain() {
        let result = biquad_peaking_mag_db(1000.0, 1000.0, 0.0, 4.0, 48000.0);
        assert!(
            result.abs() < 1e-10,
            "Zero gain should give 0 dB, got {result}"
        );
    }

    #[test]
    fn test_apply_peq_empty() {
        let freq = make_log_freq(100, 20.0, 20000.0);
        let result = apply_peq(&freq, &[], 48000.0);
        assert_eq!(result.len(), 100);
        for v in &result {
            assert!(v.abs() < 1e-10, "Empty PEQ should return all zeros");
        }
    }

    #[test]
    fn test_apply_peq_single_band() {
        let freq = make_log_freq(200, 20.0, 20000.0);
        let band = PeqBand {
            freq_hz: 1000.0,
            gain_db: -6.0,
            q: 4.0,
            enabled: true,
        };
        let result = apply_peq(&freq, &[band], 48000.0);
        let idx = freq
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                ((**a - 1000.0).abs())
                    .partial_cmp(&((**b - 1000.0).abs()))
                    .unwrap()
            })
            .unwrap()
            .0;
        assert!(
            (result[idx] - (-6.0)).abs() < 0.1,
            "Single band at 1kHz should give -6dB at 1kHz, got {}",
            result[idx]
        );
    }

    #[test]
    fn test_auto_peq_flat_error() {
        let freq = make_log_freq(200, 20.0, 20000.0);
        let mag = vec![80.0; 200];
        let config = PeqConfig {
            max_bands: 20,
            tolerance_db: 1.0,
            peak_bias: 1.5,
            max_boost_db: 6.0,
            max_cut_db: 18.0,
            freq_range: (80.0, 15000.0),
            smoothing_fraction: None,
            min_band_distance_oct: None,
        };
        let result = auto_peq(&mag, &mag, &freq, &config).unwrap();
        assert!(
            result.bands.is_empty(),
            "Flat error should produce 0 bands, got {}",
            result.bands.len()
        );
    }

    #[test]
    fn test_auto_peq_single_peak() {
        let freq = make_log_freq(500, 20.0, 20000.0);
        let target = vec![80.0; 500];
        let mut meas = vec![80.0; 500];

        for (i, &f) in freq.iter().enumerate() {
            let log_ratio = (f / 1000.0).log2();
            meas[i] = 80.0 + 10.0 * (-log_ratio * log_ratio * 8.0).exp();
        }

        let config = PeqConfig {
            max_bands: 20,
            tolerance_db: 1.0,
            peak_bias: 1.5,
            max_boost_db: 6.0,
            max_cut_db: 18.0,
            freq_range: (80.0, 15000.0),
            smoothing_fraction: None,
            min_band_distance_oct: None,
        };
        let result = auto_peq(&meas, &target, &freq, &config).unwrap();

        assert!(
            !result.bands.is_empty(),
            "Should produce at least 1 band for a 10dB peak"
        );

        // All bands should have unique frequencies (no duplicates)
        for i in 0..result.bands.len() {
            for j in (i + 1)..result.bands.len() {
                let oct_dist = (result.bands[i].freq_hz / result.bands[j].freq_hz)
                    .log2()
                    .abs();
                assert!(
                    oct_dist > 0.1,
                    "Bands should not overlap: #{} at {:.0} Hz and #{} at {:.0} Hz",
                    i, result.bands[i].freq_hz, j, result.bands[j].freq_hz
                );
            }
        }

        // The dominant band should be near 1 kHz with negative gain
        let dominant = result
            .bands
            .iter()
            .max_by(|a, b| {
                a.gain_db
                    .abs()
                    .partial_cmp(&b.gain_db.abs())
                    .unwrap()
            })
            .unwrap();
        assert!(
            dominant.freq_hz > 500.0 && dominant.freq_hz < 2000.0,
            "Dominant PEQ should be near 1kHz, got {} Hz",
            dominant.freq_hz
        );
        assert!(
            dominant.gain_db < 0.0,
            "Dominant PEQ should have negative gain (cut), got {} dB",
            dominant.gain_db
        );

        assert!(
            result.max_error_db < 5.0,
            "Max error should be significantly reduced, got {} dB",
            result.max_error_db
        );
    }

    #[test]
    fn test_auto_peq_length_mismatch() {
        let freq = vec![100.0, 200.0, 300.0];
        let meas = vec![80.0, 80.0];
        let target = vec![80.0, 80.0, 80.0];
        let config = PeqConfig {
            max_bands: 10,
            tolerance_db: 1.0,
            peak_bias: 1.5,
            max_boost_db: 6.0,
            max_cut_db: 18.0,
            freq_range: (50.0, 500.0),
            smoothing_fraction: None,
            min_band_distance_oct: None,
        };
        assert!(auto_peq(&meas, &target, &freq, &config).is_err());
    }

    #[test]
    fn test_auto_peq_invalid_range() {
        let freq = vec![100.0, 200.0, 300.0];
        let mag = vec![80.0; 3];
        let config = PeqConfig {
            max_bands: 10,
            tolerance_db: 1.0,
            peak_bias: 1.5,
            max_boost_db: 6.0,
            max_cut_db: 18.0,
            freq_range: (500.0, 50.0),
            smoothing_fraction: None,
            min_band_distance_oct: None,
        };
        assert!(auto_peq(&mag, &mag, &freq, &config).is_err());
    }

    #[test]
    fn test_pruning_removes_redundant() {
        let freq = make_log_freq(300, 20.0, 20000.0);
        let target = vec![80.0; 300];
        let mut meas = vec![80.0; 300];

        for (i, &f) in freq.iter().enumerate() {
            let log_ratio = (f / 1000.0).log2();
            meas[i] = 80.0 + 5.0 * (-log_ratio * log_ratio * 16.0).exp();
        }

        let config = PeqConfig {
            max_bands: 20,
            tolerance_db: 2.0,
            peak_bias: 1.5,
            max_boost_db: 6.0,
            max_cut_db: 18.0,
            freq_range: (80.0, 15000.0),
            smoothing_fraction: None,
            min_band_distance_oct: None,
        };
        let result = auto_peq(&meas, &target, &freq, &config).unwrap();

        assert!(
            result.bands.len() <= 3,
            "Should keep few bands for moderate peak, got {}",
            result.bands.len()
        );
    }

    #[test]
    fn test_merge_nearby_bands() {
        let mut bands = vec![
            PeqBand { freq_hz: 100.0, gain_db: -3.0, q: 2.0, enabled: true },
            PeqBand { freq_hz: 105.0, gain_db: -2.0, q: 3.0, enabled: true },
            PeqBand { freq_hz: 1000.0, gain_db: 4.0, q: 5.0, enabled: true },
        ];
        merge_nearby_bands(&mut bands, MIN_BAND_DISTANCE_OCT);
        // 100 and 105 should merge (within 1/3 octave)
        // 1000 should stay separate
        assert_eq!(bands.len(), 2, "100+105 should merge, 1000 stays: got {} bands", bands.len());
        // Merged gain should be sum: -3 + -2 = -5
        assert!((bands[0].gain_db - (-5.0)).abs() < 0.1, "Merged gain should be -5, got {}", bands[0].gain_db);
    }

    #[test]
    fn test_no_duplicate_frequencies() {
        // Realistic scenario: complex error curve with multiple peaks
        let freq = make_log_freq(500, 20.0, 20000.0);
        let target = vec![80.0; 500];
        let mut meas = vec![80.0; 500];

        // Multiple peaks at different frequencies
        for (i, &f) in freq.iter().enumerate() {
            let p1 = 6.0 * (-(((f / 200.0).log2()) * 4.0).powi(2)).exp();
            let p2 = -4.0 * (-(((f / 800.0).log2()) * 3.0).powi(2)).exp();
            let p3 = 8.0 * (-(((f / 3000.0).log2()) * 5.0).powi(2)).exp();
            meas[i] = 80.0 + p1 + p2 + p3;
        }

        let config = PeqConfig {
            max_bands: 20,
            tolerance_db: 1.0,
            peak_bias: 1.5,
            max_boost_db: 6.0,
            max_cut_db: 18.0,
            freq_range: (80.0, 15000.0),
            smoothing_fraction: None,
            min_band_distance_oct: None,
        };
        let result = auto_peq(&meas, &target, &freq, &config).unwrap();

        // Check no duplicate frequencies (all pairs should be > 1/4 octave apart)
        for i in 0..result.bands.len() {
            for j in (i + 1)..result.bands.len() {
                let oct_dist = (result.bands[i].freq_hz / result.bands[j].freq_hz)
                    .log2()
                    .abs();
                assert!(
                    oct_dist > 0.15,
                    "Bands too close: #{} at {:.0} Hz and #{} at {:.0} Hz (dist={:.2} oct)",
                    i, result.bands[i].freq_hz, j, result.bands[j].freq_hz, oct_dist
                );
            }
        }
    }

    #[test]
    fn test_estimate_q_from_peak_width() {
        let freq = make_log_freq(500, 20.0, 20000.0);
        // Create a narrow peak error
        let error: Vec<f64> = freq
            .iter()
            .map(|&f| {
                let log_ratio = (f / 1000.0).log2();
                5.0 * (-log_ratio * log_ratio * 20.0).exp() // narrow peak
            })
            .collect();
        let peak_idx = freq
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                ((**a - 1000.0).abs())
                    .partial_cmp(&((**b - 1000.0).abs()))
                    .unwrap()
            })
            .unwrap()
            .0;

        let q = estimate_q_from_peak_width(&freq, &error, peak_idx);
        assert!(
            q >= Q_MIN && q <= Q_MAX,
            "Q should be within bounds, got {}",
            q
        );
        // Narrow peak should produce higher Q
        assert!(
            q > 1.5,
            "Narrow peak should produce Q > 1.5, got {}",
            q
        );
    }

    // ------- Algorithm 2: Above-LP tests -------

    #[test]
    fn test_compute_median_simple() {
        let freq = vec![100.0, 200.0, 300.0, 400.0, 500.0];
        let mag = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = compute_median_in_range(&freq, &mag, 100.0, 500.0);
        assert_eq!(result, Some(3.0));
    }

    #[test]
    fn test_compute_median_even() {
        let freq = vec![100.0, 200.0, 300.0, 400.0];
        let mag = vec![1.0, 2.0, 3.0, 4.0];
        let result = compute_median_in_range(&freq, &mag, 100.0, 400.0);
        assert_eq!(result, Some(2.5));
    }

    #[test]
    fn test_compute_median_empty_range() {
        let freq = vec![100.0, 200.0, 300.0];
        let mag = vec![1.0, 2.0, 3.0];
        let result = compute_median_in_range(&freq, &mag, 500.0, 1000.0);
        assert_eq!(result, None);
    }

    #[test]
    fn test_auto_peq_above_lp_cuts_only() {
        let freq = make_log_freq(500, 20.0, 20000.0);
        let mut meas = vec![80.0; 500];

        // Add peaks above 3000 Hz
        for (i, &f) in freq.iter().enumerate() {
            if f > 3000.0 {
                let p = 8.0 * (-(((f / 5000.0).log2()) * 3.0).powi(2)).exp();
                meas[i] = 80.0 + p;
            }
        }

        let config = PeqConfig {
            max_bands: 20,
            tolerance_db: 1.0,
            peak_bias: 1.5,
            max_boost_db: 6.0,
            max_cut_db: 18.0,
            freq_range: (80.0, 15000.0), // will be overridden
            smoothing_fraction: None,
            min_band_distance_oct: None,
        };
        let result = auto_peq_above_lp(&meas, &freq, &config, 3000.0, 80.0).unwrap();

        // All bands should have negative gain (cuts only)
        for b in &result.bands {
            assert!(
                b.gain_db <= 0.0,
                "Above-LP bands should be cuts only, got {} dB at {} Hz",
                b.gain_db, b.freq_hz
            );
        }
        // Should have at least 1 band for the peak
        assert!(
            !result.bands.is_empty(),
            "Should produce at least 1 band for peak above LP"
        );
    }

    #[test]
    fn test_auto_peq_above_lp_flat_no_bands() {
        let freq = make_log_freq(500, 20.0, 20000.0);
        let meas = vec![80.0; 500]; // flat

        let config = PeqConfig {
            max_bands: 20,
            tolerance_db: 1.0,
            peak_bias: 1.5,
            max_boost_db: 6.0,
            max_cut_db: 18.0,
            freq_range: (80.0, 15000.0),
            smoothing_fraction: None,
            min_band_distance_oct: None,
        };
        let result = auto_peq_above_lp(&meas, &freq, &config, 3000.0, 80.0).unwrap();

        assert!(
            result.bands.is_empty(),
            "Flat measurement should produce 0 bands above LP, got {}",
            result.bands.len()
        );
    }

    // ------- LMA-specific tests -------

    #[test]
    fn test_cholesky_solve_identity() {
        // A = I, b = [1, 2, 3] → x = [1, 2, 3]
        let a = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let b = vec![1.0, 2.0, 3.0];
        let x = cholesky_solve(&a, &b).unwrap();
        assert_eq!(x.len(), 3);
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
        assert!((x[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky_solve_2x2() {
        // A = [[4, 2], [2, 3]], b = [1, 2]
        // Solution: x = [-1/8, 3/4] = [-0.125, 0.75]
        let a = vec![
            vec![4.0, 2.0],
            vec![2.0, 3.0],
        ];
        let b = vec![1.0, 2.0];
        let x = cholesky_solve(&a, &b).unwrap();
        assert_eq!(x.len(), 2);
        assert!((x[0] - (-0.125)).abs() < 1e-10, "x[0] = {}, expected -0.125", x[0]);
        assert!((x[1] - 0.75).abs() < 1e-10, "x[1] = {}, expected 0.75", x[1]);
    }

    #[test]
    fn test_cholesky_not_positive_definite() {
        // Not positive definite matrix
        let a = vec![
            vec![-1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let b = vec![1.0, 2.0];
        assert!(cholesky_solve(&a, &b).is_none());
    }

    #[test]
    fn test_weight_function_erb() {
        let freq = make_log_freq(500, 20.0, 20000.0);
        let meas = vec![80.0; 500];
        let weights = compute_weights(&freq, &meas, 80.0, 3000.0);
        assert_eq!(weights.len(), 500);

        // Check weights in different bands
        for (i, &f) in freq.iter().enumerate() {
            if f < 150.0 {
                assert!((weights[i] - 1.0).abs() < 0.01, "Low freq weight should be 1.0, got {} at {:.0} Hz", weights[i], f);
            } else if f > 800.0 && f < 3000.0 {
                // Below LP, max ear sensitivity → 2.0
                assert!((weights[i] - 2.0).abs() < 0.01, "Mid freq weight should be 2.0, got {} at {:.0} Hz", weights[i], f);
            } else if f > 5000.0 {
                // Above LP: 0.5 × 0.6 = 0.3
                assert!((weights[i] - 0.3).abs() < 0.01, "HF above-LP weight should be 0.3, got {} at {:.0} Hz", weights[i], f);
            }
        }
    }

    #[test]
    fn test_weight_null_suppression() {
        let freq = make_log_freq(100, 20.0, 20000.0);
        let mut meas = vec![80.0; 100];
        // Create a deep null at point 50
        meas[50] = 60.0; // 20 dB below median

        let weights = compute_weights(&freq, &meas, 80.0, 3000.0);
        assert!(
            weights[50] < 0.01,
            "Deep null should have near-zero weight, got {}",
            weights[50]
        );
    }

    #[test]
    fn test_unified_target_blend() {
        let freq = make_log_freq(500, 20.0, 20000.0);
        let meas = vec![80.0; 500];
        let lp = 3000.0;

        let target = compute_unified_target(&freq, &meas, 80.0, lp);
        assert_eq!(target.len(), 500);

        // Below LP: should be close to smoothed measurement (~80)
        for (i, &f) in freq.iter().enumerate() {
            if f < lp * 0.5 {
                assert!(
                    (target[i] - 80.0).abs() < 2.0,
                    "Below LP: target should be ~80, got {:.1} at {:.0} Hz",
                    target[i], f
                );
            }
        }

        // Well above LP×2: should be flat at median
        for (i, &f) in freq.iter().enumerate() {
            if f > lp * 2.5 {
                assert!(
                    (target[i] - 80.0).abs() < 1.0,
                    "Above LP×2: target should be flat at ~80, got {:.1} at {:.0} Hz",
                    target[i], f
                );
            }
        }
    }

    #[test]
    fn test_lma_single_peak() {
        let freq = make_log_freq(500, 20.0, 20000.0);
        let target = vec![80.0; 500];
        let mut meas = vec![80.0; 500];

        // Add a single 8 dB peak at 1 kHz
        for (i, &f) in freq.iter().enumerate() {
            let log_ratio = (f / 1000.0).log2();
            meas[i] = 80.0 + 8.0 * (-log_ratio * log_ratio * 10.0).exp();
        }

        let config = PeqConfig {
            max_bands: 10,
            tolerance_db: 1.0,
            peak_bias: 1.5,
            max_boost_db: 6.0,
            max_cut_db: 18.0,
            freq_range: (80.0, 15000.0),
            smoothing_fraction: None,
            min_band_distance_oct: None,
        };
        let result = auto_peq_lma(&meas, Some(&target), &freq, &config, 80.0, 15000.0).unwrap();

        assert!(
            !result.bands.is_empty(),
            "Single peak should produce at least 1 band"
        );
        // Dominant band near 1 kHz with negative gain
        let dominant = result.bands.iter()
            .max_by(|a, b| a.gain_db.abs().partial_cmp(&b.gain_db.abs()).unwrap())
            .unwrap();
        assert!(
            dominant.freq_hz > 500.0 && dominant.freq_hz < 2000.0,
            "Dominant band should be near 1 kHz, got {:.0} Hz",
            dominant.freq_hz
        );
        assert!(
            dominant.gain_db < 0.0,
            "Dominant band should cut, got {:.1} dB",
            dominant.gain_db
        );
    }

    #[test]
    fn test_lma_flat_no_bands() {
        let freq = make_log_freq(500, 20.0, 20000.0);
        let meas = vec![80.0; 500];

        let config = PeqConfig {
            max_bands: 10,
            tolerance_db: 1.0,
            peak_bias: 1.5,
            max_boost_db: 6.0,
            max_cut_db: 18.0,
            freq_range: (80.0, 15000.0),
            smoothing_fraction: None,
            min_band_distance_oct: None,
        };
        let result = auto_peq_lma(&meas, Some(&meas), &freq, &config, 80.0, 15000.0).unwrap();

        assert!(
            result.bands.is_empty(),
            "Flat meas = flat target should produce 0 bands, got {}",
            result.bands.len()
        );
    }

    #[test]
    fn test_lma_q_constraint_above_lp() {
        let freq = make_log_freq(500, 20.0, 20000.0);
        let mut meas = vec![80.0; 500];

        // Add narrow peaks above 3 kHz (LP)
        for (i, &f) in freq.iter().enumerate() {
            if f > 3000.0 {
                let p1 = 10.0 * (-(((f / 5000.0).log2()) * 6.0).powi(2)).exp();
                let p2 = 8.0 * (-(((f / 8000.0).log2()) * 6.0).powi(2)).exp();
                meas[i] = 80.0 + p1 + p2;
            }
        }

        let config = PeqConfig {
            max_bands: 10,
            tolerance_db: 1.0,
            peak_bias: 1.5,
            max_boost_db: 0.0, // cuts only
            max_cut_db: 18.0,
            freq_range: (2500.0, 20000.0),
            smoothing_fraction: Some(1.0 / 3.0),
            min_band_distance_oct: None,
        };

        let target: Vec<f64> = freq.iter().map(|_| 80.0).collect();
        let result = auto_peq_lma(&meas, Some(&target), &freq, &config, 80.0, 3000.0).unwrap();

        // All bands above LP should have Q ≤ 2.5
        for b in &result.bands {
            if b.freq_hz > 3000.0 {
                assert!(
                    b.q <= Q_MAX_ABOVE_LP + 0.01,
                    "Band at {:.0} Hz above LP should have Q ≤ {}, got Q={:.2}",
                    b.freq_hz, Q_MAX_ABOVE_LP, b.q
                );
            }
        }
    }
}

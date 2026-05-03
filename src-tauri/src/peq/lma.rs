// LMA solver, weights, and shelf promotion

use rayon::prelude::*;
use tracing::info;

use super::types::*;
use super::biquad::*;

// ---------------------------------------------------------------------------
// Cholesky decomposition solver
// ---------------------------------------------------------------------------

/// Solve A*x = b via Cholesky decomposition (A must be symmetric positive definite).
/// Flat row-major layout: a[i * n + j]. Returns None if not positive definite.
pub(crate) fn cholesky_solve_flat(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    if n == 0 || b.len() != n || a.len() != n * n {
        return None;
    }

    // Cholesky decomposition: A = L*Lt (flat row-major)
    let mut l = vec![0.0_f64; n * n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i * n + k] * l[j * n + k];
            }
            if i == j {
                let diag = a[i * n + i] - sum;
                if diag <= 0.0 {
                    return None;
                }
                l[i * n + j] = diag.sqrt();
            } else {
                let ljj = l[j * n + j];
                if ljj.abs() < 1e-30 {
                    return None;
                }
                l[i * n + j] = (a[i * n + j] - sum) / ljj;
            }
        }
    }

    // Forward substitution: L*y = b
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[i * n + j] * y[j];
        }
        let lii = l[i * n + i];
        if lii.abs() < 1e-30 {
            return None;
        }
        y[i] = (b[i] - sum) / lii;
    }

    // Back substitution: Lt*x = y
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..n {
            sum += l[j * n + i] * x[j];
        }
        let lii = l[i * n + i];
        if lii.abs() < 1e-30 {
            return None;
        }
        x[i] = (y[i] - sum) / lii;
    }

    Some(x)
}

/// Legacy wrapper for tests — converts Vec<Vec<f64>> to flat layout.
#[cfg(test)]
pub(crate) fn cholesky_solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    let mut flat = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            flat[i * n + j] = a[i][j];
        }
    }
    cholesky_solve_flat(&flat, b, n)
}

// ---------------------------------------------------------------------------
// ERB-inspired frequency weights
// ---------------------------------------------------------------------------

/// Compute psychoacoustically-weighted frequency weights for LMA optimization.
///
/// Weight bands:
/// - 20-150 Hz: 1.0  (room modes, important but limited correction potential)
/// - 150-800 Hz: 1.5  (baffle step region)
/// - 800-5000 Hz: 2.0 (maximum ear sensitivity)
/// - 5000-20000 Hz: 0.5 (broad strokes, ignore narrow peaks)
///
/// Modifiers:
/// - Above LP: x0.6 (lower priority, correction above crossover)
/// - Deep nulls (>12 dB below median): weight -> 0 (don't try to fill acoustic nulls)
pub(crate) fn compute_weights(freq: &[f64], meas_mag: &[f64], hp_freq: f64, lp_freq: f64) -> Vec<f64> {
    let n = freq.len();

    // Compute median level in HP-LP band for null detection
    let median = super::compute_median_in_range(freq, meas_mag, hp_freq, lp_freq)
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

        // Null suppression: deep dips below median -> weight -> 0
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

/// Uniform weights for Hybrid mode: all frequencies equally important.
/// NO null suppression — hybrid PEQ must flatten everything including deep rolloffs.
/// Acoustic nulls are not a concern here because hybrid applies unlimited boost.
pub(crate) fn compute_uniform_weights(freq: &[f64], _meas_mag: &[f64], _hp_freq: f64, _lp_freq: f64) -> Vec<f64> {
    vec![1.0; freq.len()]
}

/// Zero out weights for frequencies inside exclusion zones.
pub(crate) fn apply_exclusion_zones(weights: &mut [f64], freq: &[f64], zones: &[ExclusionZone]) {
    for zone in zones {
        for (i, &f) in freq.iter().enumerate() {
            if f >= zone.start_hz && f <= zone.end_hz {
                weights[i] = 0.0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Unified target curve
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// LMA Solver
// ---------------------------------------------------------------------------

pub(crate) struct LmaSolver<'a> {
    freq: &'a [f64],
    meas_mag: &'a [f64],
    target_mag: &'a [f64],
    weights: &'a [f64],
    range_indices: Vec<usize>,
    lp_freq: f64,
    config: &'a PeqConfig,
}

impl<'a> LmaSolver<'a> {
    pub(crate) fn new(
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

    /// Pack bands into parameter vector theta = [f1, G1, Q1, f2, G2, Q2, ...]
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
                filter_type: PeqFilterType::Peaking,
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
            let freq = params[i * 3];
            let q_max = if freq > self.lp_freq {
                Q_MAX_ABOVE_LP
            } else {
                crate::peq::q_cap_at(freq)
            };
            params[i * 3 + 2] = params[i * 3 + 2].clamp(Q_MIN, q_max);
        }
    }

    /// Compute weighted residual vector: r_i = sqrt(W(f_i)) * sqrt(bias) * (meas[i] + correction[i] - target[i])
    fn compute_residuals(&self, params: &[f64]) -> Vec<f64> {
        let bands = Self::params_to_bands(params);
        let correction = apply_peq(self.freq, &bands, SAMPLE_RATE);

        let n_bands = params.len() / 3;
        let n_freq = self.range_indices.len();
        let extra = if self.config.gain_regularization > 0.0 { n_bands } else { 0 };
        let mut residuals = Vec::with_capacity(n_freq + extra);

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

        // L2 gain regularization: virtual residuals sqrt(lambda*m) * gain_i
        // Squaring gives lambda*m * gain_i^2, scaled to match frequency residual magnitude
        if self.config.gain_regularization > 0.0 {
            let scale = (self.config.gain_regularization * n_freq as f64).sqrt();
            for i in 0..n_bands {
                residuals.push(scale * params[i * 3 + 1]);
            }
        }

        residuals
    }

    /// Compute total cost = Sigma r_i^2 + Q penalties
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
            // In-band penalty aligned with the envelope (b137): pressure to
            // stay below q_warn_at(freq) instead of a freq-blind threshold.
            if freq <= self.lp_freq {
                let warn = crate::peq::q_warn_at(freq);
                if q > warn {
                    cost += (q - warn).powi(2) * 0.1 * m;
                }
            }
        }

        cost
    }

    /// Compute Jacobian via numerical finite differences (parallelized with rayon).
    /// Returns flat row-major Jacobian (m x n_params) and residuals.
    fn compute_jacobian(&self, params: &[f64]) -> (Vec<f64>, Vec<f64>, usize, usize) {
        let n_params = params.len();
        let r0 = self.compute_residuals(params);
        let m = r0.len();

        // Compute step sizes
        let steps: Vec<f64> = (0..n_params).map(|p| {
            match p % 3 {
                0 => (params[p] * 1e-3).max(0.1),
                1 => 0.01,
                2 => 0.01,
                _ => unreachable!(),
            }
        }).collect();

        // Parallel: compute perturbed residuals for each parameter
        let columns: Vec<(usize, Vec<f64>)> = (0..n_params).into_par_iter()
            .filter_map(|p| {
                let h = steps[p];
                if h.abs() < 1e-12 { return None; }
                let mut params_h = params.to_vec();
                params_h[p] += h;
                let r_h = self.compute_residuals(&params_h);
                Some((p, r_h))
            })
            .collect();

        // Assemble flat row-major Jacobian: jacobian[i * n_params + p]
        let mut jacobian = vec![0.0_f64; m * n_params];
        for (p, r_h) in columns {
            let h = steps[p];
            for i in 0..m {
                jacobian[i * n_params + p] = (r_h[i] - r0[i]) / h;
            }
        }

        (jacobian, r0, m, n_params)
    }

    /// Run LMA optimization on the given bands.
    /// Returns (optimized bands, iteration count).
    pub(crate) fn optimize(&self, initial_bands: &[PeqBand]) -> (Vec<PeqBand>, u32) {
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

            let (jacobian, residuals, m, n_params) = self.compute_jacobian(&params);

            // H = Jt*J  (flat row-major n_params x n_params)
            let mut h = vec![0.0_f64; n_params * n_params];
            for i in 0..n_params {
                for j in 0..=i {
                    let mut sum = 0.0;
                    for k in 0..m {
                        sum += jacobian[k * n_params + i] * jacobian[k * n_params + j];
                    }
                    h[i * n_params + j] = sum;
                    h[j * n_params + i] = sum;
                }
            }

            // g = Jt*r  (gradient)
            let mut g = vec![0.0_f64; n_params];
            for i in 0..n_params {
                let mut sum = 0.0;
                for k in 0..m {
                    sum += jacobian[k * n_params + i] * residuals[k];
                }
                g[i] = sum;
            }

            // Damped normal equations: (H + lambda*diag(H))*delta = -g
            let mut h_damped = h.clone();
            for i in 0..n_params {
                h_damped[i * n_params + i] += lambda * h[i * n_params + i].max(1e-8);
            }

            let neg_g: Vec<f64> = g.iter().map(|&gi| -gi).collect();

            let delta = match cholesky_solve_flat(&h_damped, &neg_g, n_params) {
                Some(d) => d,
                None => {
                    // Cholesky failed — increase damping
                    lambda *= 4.0;
                    continue;
                }
            };

            // Check convergence: ||delta|| / ||theta||
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

/// Try promoting the lowest/highest frequency bands to LowShelf/HighShelf.
/// If the shelf version produces equal or lower weighted error, keep it.
pub(crate) fn try_promote_to_shelves(
    bands: &mut Vec<PeqBand>,
    freq: &[f64],
    meas_mag: &[f64],
    target: &[f64],
    config: &PeqConfig,
) {
    if bands.is_empty() { return; }

    // Helper: weighted SSE in freq_range
    let compute_sse = |bs: &[PeqBand]| -> f64 {
        let corr = apply_peq(freq, bs, SAMPLE_RATE);
        freq.iter().enumerate()
            .filter(|(_, &f)| f >= config.freq_range.0 && f <= config.freq_range.1)
            .map(|(i, _)| {
                let e = meas_mag[i] + corr[i] - target[i];
                e * e
            })
            .sum::<f64>()
    };

    let baseline = compute_sse(bands);

    // Try lowest band -> LowShelf
    if bands[0].filter_type == PeqFilterType::Peaking {
        let mut trial = bands.clone();
        trial[0].filter_type = PeqFilterType::LowShelf;
        // Shelf with lower Q for broader effect
        trial[0].q = (trial[0].q * 0.7).max(Q_MIN);
        let trial_sse = compute_sse(&trial);
        if trial_sse <= baseline * 1.05 {
            bands[0].filter_type = PeqFilterType::LowShelf;
            bands[0].q = trial[0].q;
            info!("try_promote_to_shelves: band[0] at {:.0} Hz -> LowShelf", bands[0].freq_hz);
        }
    }

    // Try highest band -> HighShelf
    let last = bands.len() - 1;
    if bands[last].filter_type == PeqFilterType::Peaking {
        let mut trial = bands.clone();
        trial[last].filter_type = PeqFilterType::HighShelf;
        trial[last].q = (trial[last].q * 0.7).max(Q_MIN);
        let trial_sse = compute_sse(&trial);
        if trial_sse <= baseline * 1.05 {
            bands[last].filter_type = PeqFilterType::HighShelf;
            bands[last].q = trial[last].q;
            info!("try_promote_to_shelves: band[{}] at {:.0} Hz -> HighShelf", last, bands[last].freq_hz);
        }
    }
}

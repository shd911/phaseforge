use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterType {
    Butterworth,
    Bessel,
    LinkwitzRiley,
    Gaussian,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterConfig {
    pub filter_type: FilterType,
    pub order: u8,          // 1..8  (Butterworth/Bessel/LR)
    pub freq_hz: f64,
    pub shape: Option<f64>, // M coefficient (Gaussian only)
    #[serde(default)]
    pub linear_phase: bool, // true → magnitude-only (zero phase)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShelfConfig {
    pub freq_hz: f64,
    pub gain_db: f64,
    pub q: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetCurve {
    pub reference_level_db: f64,
    pub tilt_db_per_octave: f64,
    pub tilt_ref_freq: f64,
    pub high_pass: Option<FilterConfig>,
    pub low_pass: Option<FilterConfig>,
    pub low_shelf: Option<ShelfConfig>,
    pub high_shelf: Option<ShelfConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetResponse {
    pub magnitude: Vec<f64>,
    pub phase: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Public filter application (for IPC: compute_filter_response)
// ---------------------------------------------------------------------------

pub fn apply_filter_public(
    mag: &mut [f64],
    phase: &mut [f64],
    freq: &[f64],
    cfg: &FilterConfig,
    is_lowpass: bool,
) {
    apply_filter(mag, phase, freq, cfg, is_lowpass);
}

// ---------------------------------------------------------------------------
// Evaluation — returns both magnitude (dB) and phase (degrees)
// ---------------------------------------------------------------------------

pub fn evaluate(target: &TargetCurve, freq: &[f64]) -> TargetResponse {
    let n = freq.len();
    let mut magnitude = vec![target.reference_level_db; n];
    let mut phase = vec![0.0_f64; n];

    // 1. Tilt (all-pass-like: adds magnitude slope, zero additional phase for simplicity)
    if target.tilt_db_per_octave.abs() > 1e-12 {
        add_tilt(&mut magnitude, freq, target.tilt_db_per_octave, target.tilt_ref_freq);
    }

    // 2. High-pass
    if let Some(hp) = &target.high_pass {
        apply_filter(&mut magnitude, &mut phase, freq, hp, false);
    }

    // 3. Low-pass
    if let Some(lp) = &target.low_pass {
        apply_filter(&mut magnitude, &mut phase, freq, lp, true);
    }

    // 4. Low shelf
    if let Some(ls) = &target.low_shelf {
        apply_shelf(&mut magnitude, &mut phase, freq, ls, true);
    }

    // 5. High shelf
    if let Some(hs) = &target.high_shelf {
        apply_shelf(&mut magnitude, &mut phase, freq, hs, false);
    }

    // Wrap phase to [-180°, 180°] (REW convention)
    for p in phase.iter_mut() {
        *p = *p % 360.0;
        if *p > 180.0 { *p -= 360.0; }
        else if *p < -180.0 { *p += 360.0; }
    }

    TargetResponse { magnitude, phase }
}

// ---------------------------------------------------------------------------
// Gaussian LP kernel
// ---------------------------------------------------------------------------

/// Gaussian low-pass on LINEAR frequency scale (linear amplitude 0..1).
///
/// Formula:  H(f) = exp( -ln(2) · (f/fc)^(2M) )
///
/// At f = fc:  H = exp(-ln2) = 0.5  →  −6 dB
/// This ensures LP and HP both cross at −6 dB (like Linkwitz-Riley),
/// and LP + HP = 1.0 (flat sum) since HP = 1 − LP.
#[inline]
fn gaussian_lp_linear(f: f64, fc: f64, m: f64) -> f64 {
    let ratio = f / fc;
    let ln2 = 2.0_f64.ln();
    (-ln2 * ratio.powf(2.0 * m)).exp()
}

// ---------------------------------------------------------------------------
// Tilt
// ---------------------------------------------------------------------------

fn add_tilt(result: &mut [f64], freq: &[f64], db_per_octave: f64, ref_freq: f64) {
    for (r, &f) in result.iter_mut().zip(freq.iter()) {
        if f > 0.0 && ref_freq > 0.0 {
            *r += db_per_octave * (f / ref_freq).log2();
        }
    }
}

// ---------------------------------------------------------------------------
// Unified filter application (magnitude + phase)
// ---------------------------------------------------------------------------

fn apply_filter(
    mag: &mut [f64],
    phase: &mut [f64],
    freq: &[f64],
    cfg: &FilterConfig,
    is_lowpass: bool,
) {
    let fc = cfg.freq_hz;
    if fc <= 0.0 {
        return;
    }
    for i in 0..freq.len() {
        let f = freq[i];
        if f <= 0.0 {
            continue;
        }
        let (m_db, p_deg) = if is_lowpass {
            filter_lp_response(f, fc, cfg)
        } else {
            filter_hp_response(f, fc, cfg)
        };
        mag[i] += m_db;
        // Linear phase: magnitude-only filter, no phase rotation
        if !cfg.linear_phase {
            phase[i] += p_deg;
        }
    }
}

// ---------------------------------------------------------------------------
// Butterworth complex transfer function
// ---------------------------------------------------------------------------

/// Butterworth LP poles for order N (left-half-plane only).
/// Pole angles: θ_k = π(2k + N - 1) / (2N), k = 1..N
/// Only poles with Re < 0 (all of them for proper Butterworth).
fn butterworth_lp_complex(f: f64, fc: f64, n: u8) -> (f64, f64) {
    // Normalized frequency
    let w = f / fc;
    let mut re_prod = 1.0_f64;
    let mut im_prod = 0.0_f64;

    for k in 0..n {
        // Pole angle (on unit circle in s-plane)
        let theta = PI * (2 * k as u32 + n as u32 + 1) as f64 / (2 * n as u32) as f64;
        let pole_re = theta.cos(); // negative for left-half plane
        let pole_im = theta.sin();

        // H_k(jw) = 1 / (jw - s_k) where s_k = pole_re + j*pole_im
        // jw - s_k = -pole_re + j*(w - pole_im)
        let dr = -pole_re;
        let di = w - pole_im;

        // 1 / (dr + j*di) = (dr - j*di) / (dr² + di²)
        let denom = dr * dr + di * di;
        let inv_re = dr / denom;
        let inv_im = -di / denom;

        // Multiply into accumulator
        let new_re = re_prod * inv_re - im_prod * inv_im;
        let new_im = re_prod * inv_im + im_prod * inv_re;
        re_prod = new_re;
        im_prod = new_im;
    }

    // Magnitude in dB and phase in degrees
    let mag_linear = (re_prod * re_prod + im_prod * im_prod).sqrt();
    let mag_db = if mag_linear > 1e-30 { 20.0 * mag_linear.log10() } else { -600.0 };
    let phase_deg = im_prod.atan2(re_prod) * (180.0 / PI);

    (mag_db, phase_deg)
}

/// Butterworth HP: H_hp(s) = s^N / B_N(s)
/// At jw: H_hp(jw) = (jw)^N * H_lp(jw) / (jw)^0...
/// Simpler: HP(f) = LP(fc²/f) approach won't give phase.
/// Use the transform s → wc/s: H_hp(jw) = H_lp(wc/(jw))
fn butterworth_hp_complex(f: f64, fc: f64, n: u8) -> (f64, f64) {
    let w = f / fc;
    let mut re_prod = 1.0_f64;
    let mut im_prod = 0.0_f64;

    for k in 0..n {
        let theta = PI * (2 * k as u32 + n as u32 + 1) as f64 / (2 * n as u32) as f64;
        let pole_re = theta.cos();
        let pole_im = theta.sin();

        // For HP, substitute s → wc/s, so at s=jw: s_hp = wc/(jw) = -j*wc/w = -j/w_norm
        // H_k(s_hp) = 1/(s_hp - s_k) = 1/(-j/w - s_k)
        // = 1/(-pole_re + j*(-1/w - pole_im))
        let dr = -pole_re;
        let di = -1.0 / w - pole_im;

        let denom = dr * dr + di * di;
        let inv_re = dr / denom;
        let inv_im = -di / denom;

        let new_re = re_prod * inv_re - im_prod * inv_im;
        let new_im = re_prod * inv_im + im_prod * inv_re;
        re_prod = new_re;
        im_prod = new_im;
    }

    let mag_linear = (re_prod * re_prod + im_prod * im_prod).sqrt();
    let mag_db = if mag_linear > 1e-30 { 20.0 * mag_linear.log10() } else { -600.0 };
    let phase_deg = im_prod.atan2(re_prod) * (180.0 / PI);

    (mag_db, phase_deg)
}

// ---------------------------------------------------------------------------
// LP / HP response dispatchers
// ---------------------------------------------------------------------------

fn filter_lp_response(f: f64, fc: f64, cfg: &FilterConfig) -> (f64, f64) {
    match cfg.filter_type {
        FilterType::Butterworth => {
            butterworth_lp_complex(f, fc, cfg.order)
        }
        FilterType::LinkwitzRiley => {
            // LR = cascaded Butterworth: mag = 2*BW_mag, phase = 2*BW_phase
            let (m, p) = butterworth_lp_complex(f, fc, cfg.order);
            (2.0 * m, 2.0 * p)
        }
        FilterType::Bessel => {
            // Use Butterworth approximation for phase (Bessel has same magnitude formula
            // in our implementation). True Bessel phase is flatter in passband but
            // the magnitude model already uses the BW formula.
            butterworth_lp_complex(f, fc, cfg.order)
        }
        FilterType::Gaussian => {
            let m = cfg.shape.unwrap_or(1.0);
            let g = gaussian_lp_linear(f, fc, m);
            let mag_db = if g <= 1e-30 { -600.0 } else { 20.0 * g.log10() };
            // Gaussian filter is linear-phase (symmetric impulse response) → zero phase
            (mag_db, 0.0)
        }
    }
}

fn filter_hp_response(f: f64, fc: f64, cfg: &FilterConfig) -> (f64, f64) {
    match cfg.filter_type {
        FilterType::Butterworth => {
            butterworth_hp_complex(f, fc, cfg.order)
        }
        FilterType::LinkwitzRiley => {
            let (m, p) = butterworth_hp_complex(f, fc, cfg.order);
            (2.0 * m, 2.0 * p)
        }
        FilterType::Bessel => {
            butterworth_hp_complex(f, fc, cfg.order)
        }
        FilterType::Gaussian => {
            let m = cfg.shape.unwrap_or(1.0);
            let lp = gaussian_lp_linear(f, fc, m);
            let hp = 1.0 - lp;
            let mag_db = if hp <= 1e-30 { -600.0 } else { 20.0 * hp.log10() };
            // Gaussian HP is also linear-phase (complementary to zero-phase LP) → zero phase
            (mag_db, 0.0)
        }
    }
}

// ---------------------------------------------------------------------------
// Shelving filters (magnitude + phase)
// ---------------------------------------------------------------------------

fn apply_shelf(
    mag: &mut [f64],
    phase: &mut [f64],
    freq: &[f64],
    cfg: &ShelfConfig,
    is_low: bool,
) {
    let fc = cfg.freq_hz;
    let gain = cfg.gain_db;
    let q = cfg.q.max(0.1);
    if fc <= 0.0 || gain.abs() < 1e-12 {
        return;
    }
    for i in 0..freq.len() {
        let f = freq[i];
        if f <= 0.0 {
            continue;
        }
        let (m_db, p_deg) = shelf_response(f, fc, gain, q, is_low);
        mag[i] += m_db;
        phase[i] += p_deg;
    }
}

fn shelf_response(f: f64, fc: f64, gain_db: f64, q: f64, is_low: bool) -> (f64, f64) {
    let ratio = if is_low { f / fc } else { fc / f };
    let ratio2 = ratio * ratio;
    let denom = ratio2 + ratio / q + 1.0;
    let shelf_linear = 1.0 / denom;
    let mag_db = gain_db * shelf_linear;

    // Phase: shelf filters introduce phase shift around the transition frequency.
    // For a 2nd-order shelf: phase ≈ -atan2(ratio/q, 1 - ratio²) * (gain_sign)
    let phase_num = ratio / q;
    let phase_den = 1.0 - ratio2;
    let raw_phase = phase_num.atan2(phase_den);
    // Scale by gain direction: positive gain → negative phase shift (lag)
    let sign = if gain_db > 0.0 { -1.0 } else { 1.0 };
    // The phase magnitude scales with the shelf depth
    let phase_scale = (1.0 - shelf_linear).abs().min(1.0);
    let phase_deg = sign * raw_phase * phase_scale * (180.0 / PI);

    (mag_db, phase_deg)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_freqs() -> Vec<f64> {
        vec![20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0]
    }

    #[test]
    fn flat_target() {
        let target = TargetCurve {
            reference_level_db: 75.0,
            tilt_db_per_octave: 0.0,
            tilt_ref_freq: 1000.0,
            high_pass: None,
            low_pass: None,
            low_shelf: None,
            high_shelf: None,
        };
        let freq = test_freqs();
        let result = evaluate(&target, &freq);
        for &v in &result.magnitude {
            assert!((v - 75.0).abs() < 1e-10, "flat target should be constant");
        }
        for &p in &result.phase {
            assert!(p.abs() < 1e-10, "flat target should have zero phase");
        }
    }

    #[test]
    fn tilt_positive() {
        let target = TargetCurve {
            reference_level_db: 0.0,
            tilt_db_per_octave: 1.0,
            tilt_ref_freq: 1000.0,
            high_pass: None,
            low_pass: None,
            low_shelf: None,
            high_shelf: None,
        };
        let freq = test_freqs();
        let result = evaluate(&target, &freq);
        let idx_1k = freq.iter().position(|&f| (f - 1000.0).abs() < 1.0).unwrap();
        assert!((result.magnitude[idx_1k]).abs() < 1e-10);
        let idx_2k = freq.iter().position(|&f| (f - 2000.0).abs() < 1.0).unwrap();
        assert!((result.magnitude[idx_2k] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn highpass_butterworth_attenuation() {
        let target = TargetCurve {
            reference_level_db: 0.0,
            tilt_db_per_octave: 0.0,
            tilt_ref_freq: 1000.0,
            high_pass: Some(FilterConfig {
                filter_type: FilterType::Butterworth,
                order: 2,
                freq_hz: 100.0,
                shape: None,
                linear_phase: false,
            }),
            low_pass: None,
            low_shelf: None,
            high_shelf: None,
        };
        let freq = test_freqs();
        let result = evaluate(&target, &freq);
        // At fc, BW order 2 should be ~ -3 dB
        let idx_100 = freq.iter().position(|&f| (f - 100.0).abs() < 1.0).unwrap();
        assert!(
            (result.magnitude[idx_100] - (-3.0103)).abs() < 0.1,
            "HP BW2 at fc should be ~-3dB, got {}",
            result.magnitude[idx_100]
        );
        // Well above fc should be ~0 dB
        let idx_10k = freq.iter().position(|&f| (f - 10000.0).abs() < 1.0).unwrap();
        assert!(
            result.magnitude[idx_10k].abs() < 0.5,
            "HP BW2 well above fc should be ~0dB, got {}",
            result.magnitude[idx_10k]
        );
        // Well below fc should be heavily attenuated
        let idx_20 = freq.iter().position(|&f| (f - 20.0).abs() < 1.0).unwrap();
        assert!(result.magnitude[idx_20] < -15.0);
    }

    #[test]
    fn lowpass_butterworth_attenuation() {
        let target = TargetCurve {
            reference_level_db: 0.0,
            tilt_db_per_octave: 0.0,
            tilt_ref_freq: 1000.0,
            high_pass: None,
            low_pass: Some(FilterConfig {
                filter_type: FilterType::Butterworth,
                order: 2,
                freq_hz: 5000.0,
                shape: None,
                linear_phase: false,
            }),
            low_shelf: None,
            high_shelf: None,
        };
        let freq = test_freqs();
        let result = evaluate(&target, &freq);
        let idx_5k = freq.iter().position(|&f| (f - 5000.0).abs() < 1.0).unwrap();
        assert!(
            (result.magnitude[idx_5k] - (-3.0103)).abs() < 0.1,
            "LP BW2 at fc should be ~-3dB, got {}",
            result.magnitude[idx_5k]
        );
        let idx_100 = freq.iter().position(|&f| (f - 100.0).abs() < 1.0).unwrap();
        assert!(result.magnitude[idx_100].abs() < 0.5);
    }

    #[test]
    fn linkwitz_riley_double_slope() {
        let bw = FilterConfig {
            filter_type: FilterType::Butterworth,
            order: 2,
            freq_hz: 100.0,
            shape: None,
            linear_phase: false,
        };
        let lr = FilterConfig {
            filter_type: FilterType::LinkwitzRiley,
            order: 2,
            freq_hz: 100.0,
            shape: None,
            linear_phase: false,
        };
        let (bw_db, _) = filter_hp_response(100.0, 100.0, &bw);
        let (lr_db, _) = filter_hp_response(100.0, 100.0, &lr);
        assert!((lr_db - 2.0 * bw_db).abs() < 0.1);
    }

    #[test]
    fn gaussian_lp_plus_hp_equals_flat() {
        let fc = 1000.0;
        let m = 1.0;
        let freqs = vec![20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0];
        for &f in &freqs {
            let lp = gaussian_lp_linear(f, fc, m);
            let hp = 1.0 - lp;
            let sum = lp + hp;
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "LP + HP must equal 1.0 at f={}, got lp={}, hp={}, sum={}",
                f, lp, hp, sum
            );
        }
    }

    #[test]
    fn gaussian_lp_at_cutoff_is_minus_6db() {
        let fc = 1000.0;
        let m = 1.0;
        let g = gaussian_lp_linear(fc, fc, m);
        // exp(-ln2) = 0.5  →  −6 dB
        let expected = 0.5_f64;
        assert!(
            (g - expected).abs() < 1e-10,
            "LP at fc should be 0.5 (−6 dB), got {}",
            g
        );
    }

    #[test]
    fn gaussian_lp_at_zero_is_unity() {
        let fc = 1000.0;
        let m = 1.0;
        let g = gaussian_lp_linear(0.0, fc, m);
        assert!((g - 1.0).abs() < 1e-12);
    }

    #[test]
    fn gaussian_lp_steepness_m() {
        let fc = 1000.0;
        let f_test = 2000.0;
        let g_m1 = gaussian_lp_linear(f_test, fc, 1.0);
        let g_m2 = gaussian_lp_linear(f_test, fc, 2.0);
        let g_m4 = gaussian_lp_linear(f_test, fc, 4.0);
        assert!(g_m1 > g_m2 && g_m2 > g_m4);
    }

    #[test]
    fn butterworth_lp_phase_at_cutoff() {
        // At fc, BW order 1 should have -45° phase
        let (_, phase) = butterworth_lp_complex(1000.0, 1000.0, 1);
        assert!(
            (phase - (-45.0)).abs() < 2.0,
            "BW1 LP at fc should be ~-45°, got {}°",
            phase
        );
    }

    #[test]
    fn evaluate_returns_phase() {
        let target = TargetCurve {
            reference_level_db: 0.0,
            tilt_db_per_octave: 0.0,
            tilt_ref_freq: 1000.0,
            high_pass: None,
            low_pass: Some(FilterConfig {
                filter_type: FilterType::Butterworth,
                order: 2,
                freq_hz: 1000.0,
                shape: None,
                linear_phase: false,
            }),
            low_shelf: None,
            high_shelf: None,
        };
        let freq = test_freqs();
        let result = evaluate(&target, &freq);
        assert_eq!(result.magnitude.len(), freq.len());
        assert_eq!(result.phase.len(), freq.len());
        // Phase should be non-zero for frequencies near and above cutoff
        let idx_1k = freq.iter().position(|&f| (f - 1000.0).abs() < 1.0).unwrap();
        assert!(result.phase[idx_1k].abs() > 1.0, "Phase at fc should be significant");
    }

    // -----------------------------------------------------------------------
    // Diagnostic: 4-band crossover sum flatness test
    // -----------------------------------------------------------------------

    /// Helper: log-spaced frequency grid
    fn log_freq_grid(f_min: f64, f_max: f64, n: usize) -> Vec<f64> {
        let log_min = f_min.log10();
        let log_max = f_max.log10();
        (0..n).map(|i| {
            let t = i as f64 / (n - 1) as f64;
            10_f64.powf(log_min + t * (log_max - log_min))
        }).collect()
    }

    /// Simulate 4-band crossover sum: band1(LP@f1) + band2(HP@f1,LP@f2) + band3(HP@f2,LP@f3) + band4(HP@f3)
    /// Returns (max_deviation_db, freq_at_max_dev) for the coherent complex sum.
    fn check_4band_sum(
        filter_type: FilterType,
        order: u8,
        shape: Option<f64>,
        linear_phase: bool,
        xo_freqs: &[f64; 3], // 3 crossover frequencies
    ) -> (f64, f64) {
        let freq = log_freq_grid(5.0, 24000.0, 2000);

        // Define 4 bands: [LP@f0], [HP@f0 + LP@f1], [HP@f1 + LP@f2], [HP@f2]
        let bands: Vec<TargetCurve> = vec![
            TargetCurve {
                reference_level_db: 0.0,
                tilt_db_per_octave: 0.0,
                tilt_ref_freq: 1000.0,
                high_pass: None,
                low_pass: Some(FilterConfig {
                    filter_type: filter_type.clone(),
                    order, freq_hz: xo_freqs[0], shape, linear_phase,
                }),
                low_shelf: None, high_shelf: None,
            },
            TargetCurve {
                reference_level_db: 0.0,
                tilt_db_per_octave: 0.0,
                tilt_ref_freq: 1000.0,
                high_pass: Some(FilterConfig {
                    filter_type: filter_type.clone(),
                    order, freq_hz: xo_freqs[0], shape, linear_phase,
                }),
                low_pass: Some(FilterConfig {
                    filter_type: filter_type.clone(),
                    order, freq_hz: xo_freqs[1], shape, linear_phase,
                }),
                low_shelf: None, high_shelf: None,
            },
            TargetCurve {
                reference_level_db: 0.0,
                tilt_db_per_octave: 0.0,
                tilt_ref_freq: 1000.0,
                high_pass: Some(FilterConfig {
                    filter_type: filter_type.clone(),
                    order, freq_hz: xo_freqs[1], shape, linear_phase,
                }),
                low_pass: Some(FilterConfig {
                    filter_type: filter_type.clone(),
                    order, freq_hz: xo_freqs[2], shape, linear_phase,
                }),
                low_shelf: None, high_shelf: None,
            },
            TargetCurve {
                reference_level_db: 0.0,
                tilt_db_per_octave: 0.0,
                tilt_ref_freq: 1000.0,
                high_pass: Some(FilterConfig {
                    filter_type: filter_type.clone(),
                    order, freq_hz: xo_freqs[2], shape, linear_phase,
                }),
                low_pass: None,
                low_shelf: None, high_shelf: None,
            },
        ];

        // Evaluate each band and do coherent complex sum
        let mut sum_re = vec![0.0_f64; freq.len()];
        let mut sum_im = vec![0.0_f64; freq.len()];
        for band in &bands {
            let resp = evaluate(band, &freq);
            for j in 0..freq.len() {
                let amp = 10_f64.powf(resp.magnitude[j] / 20.0);
                let ph_rad = resp.phase[j] * PI / 180.0;
                sum_re[j] += amp * ph_rad.cos();
                sum_im[j] += amp * ph_rad.sin();
            }
        }

        // Find max deviation from 0 dB
        let mut max_dev = 0.0_f64;
        let mut max_dev_freq = 0.0_f64;
        for j in 0..freq.len() {
            let amplitude = (sum_re[j] * sum_re[j] + sum_im[j] * sum_im[j]).sqrt();
            let sum_db = if amplitude > 1e-30 { 20.0 * amplitude.log10() } else { -200.0 };
            let dev = sum_db.abs();
            if dev > max_dev {
                max_dev = dev;
                max_dev_freq = freq[j];
            }
        }

        (max_dev, max_dev_freq)
    }

    // Gaussian with lin-phase: HP = 1 - LP → sum = 1.0 (perfect for M≥2)
    // M=1 has ~0.13 dB deviation — inherent math property of Gaussian with 4 bands
    #[test]
    fn crossover_4band_sum_gaussian_m1_linphase() {
        let (dev, f) = check_4band_sum(
            FilterType::Gaussian, 4, Some(1.0), true,
            &[100.0, 800.0, 4000.0],
        );
        eprintln!("Gaussian M=1 lin-phase: max dev = {:.4} dB @ {:.1} Hz", dev, f);
        assert!(dev < 0.15, "Gaussian M=1: got {:.4} dB @ {:.1} Hz", dev, f);
    }

    #[test]
    fn crossover_4band_sum_gaussian_m2_linphase() {
        let (dev, f) = check_4band_sum(
            FilterType::Gaussian, 4, Some(2.0), true,
            &[100.0, 800.0, 4000.0],
        );
        eprintln!("Gaussian M=2 lin-phase: max dev = {:.4} dB @ {:.1} Hz", dev, f);
        assert!(dev < 0.01, "Gaussian M=2: got {:.4} dB @ {:.1} Hz", dev, f);
    }

    #[test]
    fn crossover_4band_sum_gaussian_m3_linphase() {
        let (dev, f) = check_4band_sum(
            FilterType::Gaussian, 4, Some(3.0), true,
            &[100.0, 800.0, 4000.0],
        );
        eprintln!("Gaussian M=3 lin-phase: max dev = {:.4} dB @ {:.1} Hz", dev, f);
        assert!(dev < 0.001, "Gaussian M=3: got {:.4} dB @ {:.1} Hz", dev, f);
    }

    #[test]
    fn crossover_4band_sum_gaussian_m4_linphase() {
        let (dev, f) = check_4band_sum(
            FilterType::Gaussian, 4, Some(4.0), true,
            &[100.0, 800.0, 4000.0],
        );
        eprintln!("Gaussian M=4 lin-phase: max dev = {:.4} dB @ {:.1} Hz", dev, f);
        assert!(dev < 0.0001, "Gaussian M=4: got {:.4} dB @ {:.1} Hz", dev, f);
    }

    // LR with lin-phase=true: LR_LP + LR_HP = BW² + BW² = 1 (power-complementary)
    // → FLAT sum in magnitude-only mode
    #[test]
    fn crossover_4band_sum_lr4_linphase() {
        let (dev, f) = check_4band_sum(
            FilterType::LinkwitzRiley, 4, None, true,
            &[100.0, 800.0, 4000.0],
        );
        eprintln!("LR4 lin-phase: max dev = {:.4} dB @ {:.1} Hz", dev, f);
        assert!(dev < 0.001, "LR4 lin-phase should be flat, got {:.4} dB", dev);
    }

    #[test]
    fn crossover_4band_sum_lr2_linphase() {
        let (dev, f) = check_4band_sum(
            FilterType::LinkwitzRiley, 2, None, true,
            &[100.0, 800.0, 4000.0],
        );
        eprintln!("LR2 lin-phase: max dev = {:.4} dB @ {:.1} Hz", dev, f);
        assert!(dev < 0.02, "LR2 lin-phase should be flat, got {:.4} dB", dev);
    }

    #[test]
    fn crossover_4band_sum_lr8_linphase() {
        let (dev, f) = check_4band_sum(
            FilterType::LinkwitzRiley, 8, None, true,
            &[100.0, 800.0, 4000.0],
        );
        eprintln!("LR8 lin-phase: max dev = {:.4} dB @ {:.1} Hz", dev, f);
        assert!(dev < 0.001, "LR8 lin-phase should be flat, got {:.4} dB", dev);
    }
}

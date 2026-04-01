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
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterConfig {
    pub filter_type: FilterType,
    pub order: u8,          // 1..8  (Butterworth/Bessel/LR)
    pub freq_hz: f64,
    pub shape: Option<f64>, // M coefficient (Gaussian only)
    #[serde(default)]
    pub linear_phase: bool, // true → magnitude-only (zero phase)
    #[serde(default)]
    pub q: Option<f64>,     // Q factor (Custom only, default 0.707 = Butterworth)
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
        *p = (*p + 180.0).rem_euclid(360.0) - 180.0;
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

    // Collect filter magnitude (dB) for all frequencies
    let mut filt_mag_db = Vec::with_capacity(freq.len());
    for i in 0..freq.len() {
        let f = freq[i];
        if f <= 0.0 {
            // DC: LP=0dB (unity), HP=-600dB (silence)
            filt_mag_db.push(if is_lowpass { 0.0 } else { -600.0 });
            continue;
        }
        let (m_db, _) = if is_lowpass {
            filter_lp_response(f, fc, cfg)
        } else {
            filter_hp_response(f, fc, cfg)
        };
        filt_mag_db.push(m_db);
    }

    // Apply magnitude
    for i in 0..freq.len() {
        mag[i] += filt_mag_db[i];
    }

    // Phase: skip for linear_phase mode
    if cfg.linear_phase {
        return;
    }

    // Gaussian minimum phase: resample mag onto linear grid, Hilbert, resample phase back
    if matches!(cfg.filter_type, FilterType::Gaussian) {
        let n_fft = {
            let desired = (freq.len() * 4).max(4096);
            desired.next_power_of_two()
        };
        let n_bins = n_fft / 2 + 1;
        let nyquist = freq.last().copied().unwrap_or(24000.0);

        // Resample filter magnitude from log freq grid onto linear FFT grid (DC..Nyquist)
        let mut lin_mag = vec![filt_mag_db[0]; n_bins];
        for k in 0..n_bins {
            let f_lin = nyquist * k as f64 / (n_bins - 1) as f64;
            // Find bracketing indices in log freq grid
            if f_lin <= freq[0] {
                lin_mag[k] = filt_mag_db[0];
            } else if f_lin >= *freq.last().unwrap() {
                lin_mag[k] = *filt_mag_db.last().unwrap();
            } else {
                let mut lo = 0usize;
                let mut hi = freq.len() - 1;
                while hi - lo > 1 {
                    let mid = (lo + hi) / 2;
                    if freq[mid] <= f_lin { lo = mid; } else { hi = mid; }
                }
                let dt = freq[hi] - freq[lo];
                let frac = if dt > 0.0 { (f_lin - freq[lo]) / dt } else { 0.0 };
                lin_mag[k] = filt_mag_db[lo] + frac * (filt_mag_db[hi] - filt_mag_db[lo]);
            }
        }

        let min_ph_rad = crate::dsp::minimum_phase_from_magnitude(&lin_mag, n_fft);

        // Resample phase from linear grid back to log freq grid
        for i in 0..freq.len() {
            let bin_f = freq[i] / nyquist * (n_bins - 1) as f64;
            let lo = (bin_f as usize).min(n_bins - 2);
            let hi = lo + 1;
            let frac = bin_f - lo as f64;
            let ph_rad = min_ph_rad[lo] * (1.0 - frac) + min_ph_rad[hi] * frac;
            phase[i] += ph_rad.to_degrees();
        }
    } else {
        // Other filter types: use analytical phase from filter_response
        for i in 0..freq.len() {
            let f = freq[i];
            if f <= 0.0 { continue; }
            let (_, p_deg) = if is_lowpass {
                filter_lp_response(f, fc, cfg)
            } else {
                filter_hp_response(f, fc, cfg)
            };
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
// Bessel complex transfer function (frequency-normalized poles)
// ---------------------------------------------------------------------------

/// Bessel filter poles (normalized to unit group delay at DC).
/// These are the standard frequency-normalized Bessel poles for orders 1–8.
/// Each entry is (real, imag) for left-half-plane poles with imag >= 0.
/// Conjugate pairs are implied for complex poles.
fn bessel_poles(order: u8) -> Vec<(f64, f64)> {
    match order {
        1 => vec![(-1.0, 0.0)],
        2 => vec![(-1.1016, 0.6368)],
        3 => vec![(-1.3226, 0.0), (-1.0474, 0.9992)],
        4 => vec![(-1.3700, 0.4102), (-0.9953, 1.2571)],
        5 => vec![(-1.5069, 0.0), (-1.3810, 0.7179), (-0.9576, 1.4711)],
        6 => vec![(-1.5735, 0.3213), (-1.3836, 0.9727), (-0.9307, 1.6620)],
        7 => vec![(-1.6853, 0.0), (-1.6130, 0.5896), (-1.3797, 1.1923), (-0.9104, 1.8364)],
        8 => vec![(-1.7575, 0.2737), (-1.6365, 0.8230), (-1.3690, 1.3883), (-0.8955, 1.9983)],
        _ => bessel_poles(8), // clamp to 8
    }
}

fn bessel_lp_complex(f: f64, fc: f64, n: u8) -> (f64, f64) {
    let poles = bessel_poles(n);
    // Frequency scaling: Bessel poles are normalized to ω₀=1 for group delay.
    // Scale by fc to move cutoff: s_scaled = s / (2π·fc), but we evaluate at s=j·2π·f.
    // Simplified: w = f/fc, evaluate H(jw) = K / product(jw - p_k) for all poles (including conjugates).
    let w = f / fc;
    let mut re_prod = 1.0_f64;
    let mut im_prod = 0.0_f64;

    for &(pr, pi) in &poles {
        if pi.abs() < 1e-15 {
            // Real pole: 1/(jw - pr)
            let dr = -pr;
            let di = w;
            let denom = dr * dr + di * di;
            let inv_re = dr / denom;
            let inv_im = -di / denom;
            let new_re = re_prod * inv_re - im_prod * inv_im;
            let new_im = re_prod * inv_im + im_prod * inv_re;
            re_prod = new_re;
            im_prod = new_im;
        } else {
            // Conjugate pair: 1/(jw - p) * 1/(jw - p*)
            // = 1/((jw - pr - j·pi)(jw - pr + j·pi))
            // = 1/((−pr + j(w − pi))(−pr + j(w + pi)))
            let dr1 = -pr;
            let di1 = w - pi;
            let dr2 = -pr;
            let di2 = w + pi;
            // Multiply the two denominators
            let d_re = dr1 * dr2 - di1 * di2;
            let d_im = dr1 * di2 + di1 * dr2;
            let d_mag2 = d_re * d_re + d_im * d_im;
            let inv_re = d_re / d_mag2;
            let inv_im = -d_im / d_mag2;
            let new_re = re_prod * inv_re - im_prod * inv_im;
            let new_im = re_prod * inv_im + im_prod * inv_re;
            re_prod = new_re;
            im_prod = new_im;
        }
    }

    // Normalize DC gain to 0 dB: at w=0, H(0) = 1/product(-p_k) for all poles.
    // Compute DC gain and divide.
    let mut dc_re = 1.0_f64;
    let mut dc_im = 0.0_f64;
    for &(pr, pi) in &poles {
        if pi.abs() < 1e-15 {
            let dr = -pr;
            let inv_re = 1.0 / dr;
            let new_re = dc_re * inv_re;
            let new_im = dc_im * inv_re;
            dc_re = new_re;
            dc_im = new_im;
        } else {
            let dr1 = -pr;
            let di1 = -pi;
            let dr2 = -pr;
            let di2 = pi;
            let d_re = dr1 * dr2 - di1 * di2;
            let d_im = dr1 * di2 + di1 * dr2;
            let d_mag2 = d_re * d_re + d_im * d_im;
            let inv_re = d_re / d_mag2;
            let inv_im = -d_im / d_mag2;
            let new_re = dc_re * inv_re - dc_im * inv_im;
            let new_im = dc_re * inv_im + dc_im * inv_re;
            dc_re = new_re;
            dc_im = new_im;
        }
    }
    // Normalize: H(0) is real positive for Bessel (symmetric poles). Divide by |H(0)|.
    let dc_mag = (dc_re * dc_re + dc_im * dc_im).sqrt();
    let final_re = re_prod / dc_mag;
    let final_im = im_prod / dc_mag;

    let mag_linear = (final_re * final_re + final_im * final_im).sqrt();
    let mag_db = if mag_linear > 1e-30 { 20.0 * mag_linear.log10() } else { -600.0 };
    let phase_deg = final_im.atan2(final_re) * (180.0 / PI);

    (mag_db, phase_deg)
}

fn bessel_hp_complex(f: f64, fc: f64, n: u8) -> (f64, f64) {
    // HP via LP-to-HP transform: s → wc/s (same approach as Butterworth HP)
    let poles = bessel_poles(n);
    let w = f / fc;
    let mut re_prod = 1.0_f64;
    let mut im_prod = 0.0_f64;

    for &(pr, pi) in &poles {
        if pi.abs() < 1e-15 {
            let dr = -pr;
            let di = -1.0 / w;
            let denom = dr * dr + di * di;
            let inv_re = dr / denom;
            let inv_im = -di / denom;
            let new_re = re_prod * inv_re - im_prod * inv_im;
            let new_im = re_prod * inv_im + im_prod * inv_re;
            re_prod = new_re;
            im_prod = new_im;
        } else {
            let dr1 = -pr;
            let di1 = -1.0 / w - pi;
            let dr2 = -pr;
            let di2 = -1.0 / w + pi;
            let d_re = dr1 * dr2 - di1 * di2;
            let d_im = dr1 * di2 + di1 * dr2;
            let d_mag2 = d_re * d_re + d_im * d_im;
            let inv_re = d_re / d_mag2;
            let inv_im = -d_im / d_mag2;
            let new_re = re_prod * inv_re - im_prod * inv_im;
            let new_im = re_prod * inv_im + im_prod * inv_re;
            re_prod = new_re;
            im_prod = new_im;
        }
    }

    // DC normalization (same as LP, but for HP gain at HF → 0 dB)
    // At w→∞, HP → 0 dB. The product form naturally gives this. Just normalize DC of underlying LP.
    let mut dc_re = 1.0_f64;
    let mut dc_im = 0.0_f64;
    for &(pr, pi) in &poles {
        if pi.abs() < 1e-15 {
            let inv_re = 1.0 / (-pr);
            let new_re = dc_re * inv_re;
            let new_im = dc_im * inv_re;
            dc_re = new_re;
            dc_im = new_im;
        } else {
            // Conjugate pair at HF limit (1/w → 0):
            // (-pr + j·(-pi))(-pr + j·(+pi)) = pr² + pi²
            let d_re = pr * pr + pi * pi;
            let d_im = 0.0;
            let d_mag2 = d_re * d_re + d_im * d_im;
            let inv_re = d_re / d_mag2;
            let inv_im = -d_im / d_mag2;
            let new_re = dc_re * inv_re - dc_im * inv_im;
            let new_im = dc_re * inv_im + dc_im * inv_re;
            dc_re = new_re;
            dc_im = new_im;
        }
    }
    let dc_mag = (dc_re * dc_re + dc_im * dc_im).sqrt();
    let final_re = re_prod / dc_mag;
    let final_im = im_prod / dc_mag;

    let mag_linear = (final_re * final_re + final_im * final_im).sqrt();
    let mag_db = if mag_linear > 1e-30 { 20.0 * mag_linear.log10() } else { -600.0 };
    let phase_deg = final_im.atan2(final_re) * (180.0 / PI);

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
            bessel_lp_complex(f, fc, cfg.order)
        }
        FilterType::Gaussian => {
            let m = cfg.shape.unwrap_or(1.0);
            let g = gaussian_lp_linear(f, fc, m);
            let mag_db = if g <= 1e-30 { -600.0 } else { 20.0 * g.log10() };
            (mag_db, 0.0)
        }
        FilterType::Custom => {
            let q = cfg.q.unwrap_or(0.707);
            custom_lp_complex(f, fc, q, cfg.order)
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
            bessel_hp_complex(f, fc, cfg.order)
        }
        FilterType::Gaussian => {
            let m = cfg.shape.unwrap_or(1.0);
            let lp = gaussian_lp_linear(f, fc, m);
            let hp = 1.0 - lp;
            let mag_db = if hp <= 1e-30 { -600.0 } else { 20.0 * hp.log10() };
            (mag_db, 0.0)
        }
        FilterType::Custom => {
            let q = cfg.q.unwrap_or(0.707);
            custom_hp_complex(f, fc, q, cfg.order)
        }
    }
}

// ---------------------------------------------------------------------------
// Custom LP / HP (2nd-order biquad sections with user Q, cascaded for higher orders)
// ---------------------------------------------------------------------------

/// Custom low-pass: cascades order/2 second-order LP biquad sections, each with the given Q.
/// Odd orders add a 1st-order section (Q irrelevant for 1st order).
fn custom_lp_complex(f: f64, fc: f64, q: f64, order: u8) -> (f64, f64) {
    let w = f / fc; // normalized frequency (ratio, same as Butterworth)
    let n_second_order = order / 2;
    let has_first_order = order % 2 == 1;

    let mut total_mag_db = 0.0;
    let mut total_phase_deg = 0.0;

    // 1st-order section: H(s) = 1/(s+1), |H(jw)| = 1/√(1+w²)
    if has_first_order {
        let mag_sq = 1.0 / (1.0 + w * w);
        total_mag_db += 10.0 * mag_sq.log10();
        total_phase_deg += -(w.atan()) * 180.0 / PI;
    }

    // 2nd-order sections: H(s) = 1/(s² + s/Q + 1)
    for _ in 0..n_second_order {
        let w2 = w * w;
        let denom_re = 1.0 - w2;
        let denom_im = w / q;
        let denom_sq = denom_re * denom_re + denom_im * denom_im;
        if denom_sq < 1e-30 {
            total_mag_db += -300.0;
        } else {
            total_mag_db += -10.0 * denom_sq.log10();
            total_phase_deg += -(denom_im.atan2(denom_re)) * 180.0 / PI;
        }
    }

    (total_mag_db, total_phase_deg)
}

/// Custom high-pass: transform s → fc/s gives HP from LP sections.
fn custom_hp_complex(f: f64, fc: f64, q: f64, order: u8) -> (f64, f64) {
    let w = f / fc;
    if w.abs() < 1e-15 {
        return (-600.0, 0.0);
    }
    // HP: each section uses the same w but with s²/(s²+s/Q+1) transfer function
    let n_second_order = order / 2;
    let has_first_order = order % 2 == 1;

    let mut total_mag_db = 0.0;
    let mut total_phase_deg = 0.0;

    // Each section contributes: H_hp = (jw)^n * H_lp(1/jw)
    // For 1st order: H_hp(jw) = jw/(jw+1) → |H| = w/√(1+w²), phase = 90° - atan(w)
    if has_first_order {
        let w2 = w * w;
        let mag_sq = w2 / (1.0 + w2);
        total_mag_db += 10.0 * mag_sq.log10();
        total_phase_deg += (PI / 2.0 - w.atan()) * 180.0 / PI;
    }

    // 2nd-order HP: H(s) = s²/(s² + s/Q + 1)
    for _ in 0..n_second_order {
        let w2 = w * w;
        let w4 = w2 * w2;
        // Numerator: (jw)² = -w² → |num|² = w⁴
        // Denominator: (jw)² + jw/Q + 1 = (1-w²) + j·w/Q
        let denom_re = 1.0 - w2;
        let denom_im = w / q;
        let denom_sq = denom_re * denom_re + denom_im * denom_im;
        if denom_sq < 1e-30 {
            total_mag_db += -300.0;
        } else {
            total_mag_db += 10.0 * (w4 / denom_sq).log10();
            // Phase: angle(num) - angle(denom)
            // num phase = angle(-w²) = π (for w>0)
            let denom_phase = denom_im.atan2(denom_re);
            total_phase_deg += (PI - denom_phase) * 180.0 / PI;
        }
    }

    (total_mag_db, total_phase_deg)
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

/// Analog 2nd-order shelf: exact s-domain transfer function.
///
/// Low shelf:  H(s) = A * (s² + s*(√A)/Q + A*ω₀²) / (A*s² + s*(√A)/Q + ω₀²)
///   where A = 10^(gain_dB/20), ω₀ = 2π·fc, and s = j·2π·f.
/// High shelf: dual — swap numerator/denominator roles.
///
/// This is the exact analog prototype used by the RBJ Audio EQ Cookbook
/// (before bilinear transform), ensuring consistency with PEQ biquad shelves
/// at low frequencies where warping is negligible.
fn shelf_response(f: f64, fc: f64, gain_db: f64, q: f64, is_low: bool) -> (f64, f64) {
    if gain_db.abs() < 1e-10 || q <= 0.0 || fc <= 0.0 {
        return (0.0, 0.0);
    }
    let a = 10.0_f64.powf(gain_db / 40.0); // A = sqrt(linear gain), per RBJ convention
    let sqrt_a = a.sqrt();
    let w0 = fc; // normalized frequency ratio
    let w = f;   // current frequency
    // Work with ratio s = j·(w/w0) for numerical stability
    let r = w / w0; // frequency ratio
    let r2 = r * r;

    // Low shelf H(jω) = A · (−r² + j·r·√A/Q + A) / (−A·r² + j·r·√A/Q + 1)
    // Simplify: numerator = (A − r²) + j·(r·√A/Q)
    //           denominator = (1 − A·r²) + j·(r·√A/Q)
    let (num_re, num_im, den_re, den_im) = if is_low {
        (
            a - r2,
            r * sqrt_a / q,
            1.0 - a * r2,
            r * sqrt_a / q,
        )
    } else {
        // High shelf: H(jω) = A · (A − r²) / (1 − A·r²) but with swapped shelf direction
        // H_hs(s) = A · (A·s² + s·√A/Q + ω₀²) / (s² + s·√A/Q + A·ω₀²)
        // → num = (A·(−r²) + 1) + j·(r·√A/Q) = (1 − A·r²) + j·(r·√A/Q)
        // → den = (−r² + A) + j·(r·√A/Q) = (A − r²) + j·(r·√A/Q)
        // Then multiply by A
        (
            1.0 - a * r2,
            r * sqrt_a / q,
            a - r2,
            r * sqrt_a / q,
        )
    };

    // H = A · (num_re + j·num_im) / (den_re + j·den_im)
    let den_mag2 = den_re * den_re + den_im * den_im;
    if den_mag2 < 1e-30 {
        return (gain_db, 0.0); // at singularity, return full gain
    }
    // Complex division: (num * conj(den)) / |den|²
    let h_re = a * (num_re * den_re + num_im * den_im) / den_mag2;
    let h_im = a * (num_im * den_re - num_re * den_im) / den_mag2;

    let mag_linear = (h_re * h_re + h_im * h_im).sqrt();
    let mag_db = if mag_linear > 1e-30 { 20.0 * mag_linear.log10() } else { -600.0 };
    let phase_deg = h_im.atan2(h_re) * (180.0 / PI);

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
                q: None,
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
                q: None,
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
            q: None,
        };
        let lr = FilterConfig {
            filter_type: FilterType::LinkwitzRiley,
            order: 2,
            freq_hz: 100.0,
            shape: None,
            linear_phase: false,
            q: None,
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
                q: None,
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
                    order, freq_hz: xo_freqs[0], shape, linear_phase, q: None,
                }),
                low_shelf: None, high_shelf: None,
            },
            TargetCurve {
                reference_level_db: 0.0,
                tilt_db_per_octave: 0.0,
                tilt_ref_freq: 1000.0,
                high_pass: Some(FilterConfig {
                    filter_type: filter_type.clone(),
                    order, freq_hz: xo_freqs[0], shape, linear_phase, q: None,
                }),
                low_pass: Some(FilterConfig {
                    filter_type: filter_type.clone(),
                    order, freq_hz: xo_freqs[1], shape, linear_phase, q: None,
                }),
                low_shelf: None, high_shelf: None,
            },
            TargetCurve {
                reference_level_db: 0.0,
                tilt_db_per_octave: 0.0,
                tilt_ref_freq: 1000.0,
                high_pass: Some(FilterConfig {
                    filter_type: filter_type.clone(),
                    order, freq_hz: xo_freqs[1], shape, linear_phase, q: None,
                }),
                low_pass: Some(FilterConfig {
                    filter_type: filter_type.clone(),
                    order, freq_hz: xo_freqs[2], shape, linear_phase, q: None,
                }),
                low_shelf: None, high_shelf: None,
            },
            TargetCurve {
                reference_level_db: 0.0,
                tilt_db_per_octave: 0.0,
                tilt_ref_freq: 1000.0,
                high_pass: Some(FilterConfig {
                    filter_type: filter_type.clone(),
                    order, freq_hz: xo_freqs[2], shape, linear_phase, q: None,
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

    #[test]
    fn bessel_lp_dc_gain_zero_and_rolloff() {
        // Bessel LP: DC gain should be 0 dB, at fc should be roughly -3 dB (frequency-normalized)
        let (mag_dc, _) = bessel_lp_complex(10.0, 10000.0, 2);
        assert!((mag_dc - 0.0).abs() < 0.01, "Bessel LP DC gain should be ~0 dB, got {}", mag_dc);
        // At fc, Bessel order 2 should be about -3 dB (group-delay-normalized poles)
        let (mag_fc, _) = bessel_lp_complex(1000.0, 1000.0, 2);
        assert!(mag_fc < -2.0 && mag_fc > -5.0, "Bessel LP at fc should be ~-3 dB, got {}", mag_fc);
        // Well above fc should be heavily attenuated
        let (mag_hi, _) = bessel_lp_complex(10000.0, 1000.0, 4);
        assert!(mag_hi < -20.0, "Bessel LP 10x above fc order 4 should be < -20 dB, got {}", mag_hi);
    }

    #[test]
    fn bessel_hp_hf_gain_zero_and_rolloff() {
        let (mag_hf, _) = bessel_hp_complex(10000.0, 100.0, 2);
        assert!((mag_hf - 0.0).abs() < 0.5, "Bessel HP HF gain should be ~0 dB, got {}", mag_hf);
        let (mag_fc, _) = bessel_hp_complex(1000.0, 1000.0, 2);
        assert!(mag_fc < -2.0 && mag_fc > -5.0, "Bessel HP at fc should be ~-3 dB, got {}", mag_fc);
        let (mag_lo, _) = bessel_hp_complex(10.0, 1000.0, 4);
        assert!(mag_lo < -20.0, "Bessel HP well below fc should be < -20 dB, got {}", mag_lo);
    }

    #[test]
    fn shelf_low_gains_at_limits() {
        // Low shelf +6 dB at fc=200 Hz, Q=0.707:
        // At DC (f << fc): should be ~+6 dB
        let (mag_dc, _) = shelf_response(10.0, 200.0, 6.0, 0.707, true);
        assert!((mag_dc - 6.0).abs() < 0.3, "Low shelf DC should be ~+6 dB, got {}", mag_dc);
        // At HF (f >> fc): should be ~0 dB
        let (mag_hf, _) = shelf_response(10000.0, 200.0, 6.0, 0.707, true);
        assert!(mag_hf.abs() < 0.3, "Low shelf HF should be ~0 dB, got {}", mag_hf);
        // At fc: should be ~+3 dB (half gain)
        let (mag_fc, _) = shelf_response(200.0, 200.0, 6.0, 0.707, true);
        assert!((mag_fc - 3.0).abs() < 1.0, "Low shelf at fc should be ~+3 dB, got {}", mag_fc);
    }

    #[test]
    fn shelf_high_gains_at_limits() {
        // High shelf +6 dB at fc=2000 Hz, Q=0.707:
        let (mag_hf, _) = shelf_response(10000.0, 2000.0, 6.0, 0.707, false);
        assert!((mag_hf - 6.0).abs() < 0.3, "High shelf HF should be ~+6 dB, got {}", mag_hf);
        let (mag_dc, _) = shelf_response(10.0, 2000.0, 6.0, 0.707, false);
        assert!(mag_dc.abs() < 0.3, "High shelf DC should be ~0 dB, got {}", mag_dc);
    }
}

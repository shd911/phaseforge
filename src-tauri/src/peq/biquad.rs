// Biquad filter evaluation for PEQ bands

use std::f64::consts::PI;

use super::types::*;

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
        .map(|&f| peq_band_mag_db(f, band, sample_rate))
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
            let (mag_db, phase_deg) = peq_band_complex(f, band, sample_rate);
            total_mag[i] += mag_db;
            total_phase[i] += phase_deg;
        }
    }
    // Wrap phase to [-180, 180] (REW convention)
    for p in total_phase.iter_mut() {
        *p = (*p + 180.0).rem_euclid(360.0) - 180.0;
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

/// Low shelf biquad (RBJ Audio EQ Cookbook) — returns (mag_db, phase_deg).
fn biquad_lowshelf_complex(f: f64, fc: f64, gain_db: f64, q: f64, sample_rate: f64) -> (f64, f64) {
    if gain_db.abs() < 1e-10 || q <= 0.0 || fc <= 0.0 || sample_rate <= 0.0 {
        return (0.0, 0.0);
    }
    let a_lin = 10.0_f64.powf(gain_db / 40.0); // sqrt of linear gain
    let w0 = 2.0 * PI * fc / sample_rate;
    let cos_w0 = w0.cos();
    let alpha = w0.sin() / (2.0 * q);
    let two_sqrt_a_alpha = 2.0 * a_lin.sqrt() * alpha;

    let b0 = a_lin * ((a_lin + 1.0) - (a_lin - 1.0) * cos_w0 + two_sqrt_a_alpha);
    let b1 = 2.0 * a_lin * ((a_lin - 1.0) - (a_lin + 1.0) * cos_w0);
    let b2 = a_lin * ((a_lin + 1.0) - (a_lin - 1.0) * cos_w0 - two_sqrt_a_alpha);
    let a0 = (a_lin + 1.0) + (a_lin - 1.0) * cos_w0 + two_sqrt_a_alpha;
    let a1 = -2.0 * ((a_lin - 1.0) + (a_lin + 1.0) * cos_w0);
    let a2 = (a_lin + 1.0) + (a_lin - 1.0) * cos_w0 - two_sqrt_a_alpha;

    biquad_eval_complex(f, sample_rate, b0, b1, b2, a0, a1, a2)
}

/// High shelf biquad (RBJ Audio EQ Cookbook) — returns (mag_db, phase_deg).
fn biquad_highshelf_complex(f: f64, fc: f64, gain_db: f64, q: f64, sample_rate: f64) -> (f64, f64) {
    if gain_db.abs() < 1e-10 || q <= 0.0 || fc <= 0.0 || sample_rate <= 0.0 {
        return (0.0, 0.0);
    }
    let a_lin = 10.0_f64.powf(gain_db / 40.0);
    let w0 = 2.0 * PI * fc / sample_rate;
    let cos_w0 = w0.cos();
    let alpha = w0.sin() / (2.0 * q);
    let two_sqrt_a_alpha = 2.0 * a_lin.sqrt() * alpha;

    let b0 = a_lin * ((a_lin + 1.0) + (a_lin - 1.0) * cos_w0 + two_sqrt_a_alpha);
    let b1 = -2.0 * a_lin * ((a_lin - 1.0) + (a_lin + 1.0) * cos_w0);
    let b2 = a_lin * ((a_lin + 1.0) + (a_lin - 1.0) * cos_w0 - two_sqrt_a_alpha);
    let a0 = (a_lin + 1.0) - (a_lin - 1.0) * cos_w0 + two_sqrt_a_alpha;
    let a1 = 2.0 * ((a_lin - 1.0) - (a_lin + 1.0) * cos_w0);
    let a2 = (a_lin + 1.0) - (a_lin - 1.0) * cos_w0 - two_sqrt_a_alpha;

    biquad_eval_complex(f, sample_rate, b0, b1, b2, a0, a1, a2)
}

/// Evaluate biquad transfer function H(z) at frequency f — shared helper.
fn biquad_eval_complex(f: f64, sample_rate: f64, b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) -> (f64, f64) {
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

    let mag_db = if den_mag_sq < 1e-30 { 0.0 } else { 10.0 * (num_mag_sq / den_mag_sq).log10() };
    let phase_deg = (num_im.atan2(num_re) - den_im.atan2(den_re)) * 180.0 / PI;

    (mag_db, phase_deg)
}

/// Dispatch to the correct biquad function based on PeqFilterType.
pub(crate) fn peq_band_complex(f: f64, band: &PeqBand, sample_rate: f64) -> (f64, f64) {
    match band.filter_type {
        PeqFilterType::Peaking => biquad_peaking_complex(f, band.freq_hz, band.gain_db, band.q, sample_rate),
        PeqFilterType::LowShelf => biquad_lowshelf_complex(f, band.freq_hz, band.gain_db, band.q, sample_rate),
        PeqFilterType::HighShelf => biquad_highshelf_complex(f, band.freq_hz, band.gain_db, band.q, sample_rate),
    }
}

/// Dispatch magnitude-only version.
pub(crate) fn peq_band_mag_db(f: f64, band: &PeqBand, sample_rate: f64) -> f64 {
    match band.filter_type {
        PeqFilterType::Peaking => biquad_peaking_mag_db(f, band.freq_hz, band.gain_db, band.q, sample_rate),
        PeqFilterType::LowShelf | PeqFilterType::HighShelf => {
            let (mag, _) = peq_band_complex(f, band, sample_rate);
            mag
        }
    }
}

// ---------------------------------------------------------------------------
// Internal: Biquad Peaking EQ (RBJ Audio EQ Cookbook)
// ---------------------------------------------------------------------------

pub(crate) fn biquad_peaking_mag_db(f: f64, fc: f64, gain_db: f64, q: f64, sample_rate: f64) -> f64 {
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

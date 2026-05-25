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
///
/// b140.16: complex-accumulator path. Pre-fix this summed `phase_deg += per_band_phase`
/// scalarly; each band's phase came wrapped to (-180, 180] from atan2 and the sum could
/// land in (-360 × N, 360 × N], wrapping incorrectly across freq bins when one band's
/// phase crossed ±180°. Same bug class as the apply_filter scalar sum fixed in b140.15.9.
/// Complex multiplication of unit phasors is phase addition modulo 360° — wrap-invariant —
/// so the final atan2 produces one clean (-180, 180] wrap per bin.
pub fn apply_peq_complex(freq: &[f64], bands: &[PeqBand], sample_rate: f64) -> (Vec<f64>, Vec<f64>) {
    let n = freq.len();
    let mut total_mag = vec![0.0_f64; n];
    let mut re_acc = vec![1.0_f64; n];
    let mut im_acc = vec![0.0_f64; n];
    for band in bands {
        if !band.enabled {
            continue;
        }
        for (i, &f) in freq.iter().enumerate() {
            let (mag_db, phase_deg) = peq_band_complex(f, band, sample_rate);
            total_mag[i] += mag_db;
            let p_rad = phase_deg.to_radians();
            let c = p_rad.cos();
            let s = p_rad.sin();
            let nr = re_acc[i] * c - im_acc[i] * s;
            let ni = re_acc[i] * s + im_acc[i] * c;
            re_acc[i] = nr;
            im_acc[i] = ni;
        }
    }
    let mut total_phase = vec![0.0_f64; n];
    for i in 0..n {
        total_phase[i] = im_acc[i].atan2(re_acc[i]).to_degrees();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::peq::types::{PeqBand, PeqFilterType};

    /// b140.16 regression: apply_peq_complex used to sum each band's
    /// wrapped phase scalarly. With 3+ peaking bands at moderate gain near
    /// overlapping frequencies, individual phase wraps would accumulate
    /// into ±540° / ±720° sums that wrap incorrectly across freq bins —
    /// visible as 1-bin spikes on PEQ phase trace. Complex-accumulator
    /// path eliminates this.
    #[test]
    fn apply_peq_complex_no_phase_spikes_multi_band() {
        let sample_rate = 48_000.0;
        // 5 PEQ bands at moderate gains spread across the audio band.
        // Each individual biquad's phase wraps at ±180° somewhere; the
        // scalar sum used to wrap-jump where multiple components crossed
        // at adjacent freq bins.
        let bands = vec![
            PeqBand { freq_hz: 80.0,   gain_db:  6.0, q: 2.0, enabled: true, filter_type: PeqFilterType::Peaking },
            PeqBand { freq_hz: 250.0,  gain_db: -8.0, q: 3.0, enabled: true, filter_type: PeqFilterType::Peaking },
            PeqBand { freq_hz: 800.0,  gain_db:  4.0, q: 5.0, enabled: true, filter_type: PeqFilterType::Peaking },
            PeqBand { freq_hz: 2500.0, gain_db: -6.0, q: 2.5, enabled: true, filter_type: PeqFilterType::Peaking },
            PeqBand { freq_hz: 7000.0, gain_db:  3.0, q: 1.5, enabled: true, filter_type: PeqFilterType::Peaking },
        ];
        let n = 1024;
        let freq: Vec<f64> = (0..n)
            .map(|i| 20.0 * (1000.0_f64).powf(i as f64 / (n - 1) as f64))
            .collect();

        let (_mag, phase) = apply_peq_complex(&freq, &bands, sample_rate);

        // Phase must be wrapped to (-180, 180] cleanly with no 1-bin spikes
        // (defined as |Δ| > 30° vs both neighbours where neighbours agree
        // within 10° modulo 360°).
        let wrap = |x: f64| x - 360.0 * (x / 360.0).round();
        let mut spikes = Vec::new();
        for i in 1..n - 1 {
            let a = phase[i - 1];
            let b = phase[i];
            let c = phase[i + 1];
            let ab = wrap(b - a).abs();
            let bc = wrap(b - c).abs();
            let ac = wrap(a - c).abs();
            if ab > 30.0 && bc > 30.0 && ac < 10.0 {
                spikes.push((i, freq[i], a, b, c));
            }
        }
        if !spikes.is_empty() {
            let mut msg = format!("\n{} phase spike(s) in 5-band PEQ sum:\n", spikes.len());
            for (i, f, a, b, c) in spikes.iter().take(6) {
                msg.push_str(&format!("  bin {} f={:.1}Hz prev={:.2} spike={:.2} next={:.2}°\n",
                    i, f, a, b, c));
            }
            panic!("{}", msg);
        }

        // Sanity: phase should remain in (-180, 180] (atan2 output range)
        for (i, &p) in phase.iter().enumerate() {
            assert!(p > -180.0 - 1e-9 && p <= 180.0 + 1e-9,
                "bin {} f={:.1}Hz phase={} out of (-180, 180]", i, freq[i], p);
        }
    }
}

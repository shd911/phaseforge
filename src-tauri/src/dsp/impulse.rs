use num_complex::Complex64;

use super::fft::FftEngine;
use super::interpolation::interpolate_linear_grid;

/// One band's frequency response feeding an impulse computation.
/// b141.59: `delay_s` (alignment, positive = later) and `sign` (polarity) are
/// applied EXACTLY on the linear FFT bins — never baked into the wrapped
/// phase of the sparse log grid, where a 3 ms ramp turns > 180° per step at
/// HF and the shortest-arc interpolation turned it into a 25 % pre-response
/// in Σ IR/Step (96 kHz, delays 0.8–3.1 ms).
pub struct SpectrumPart<'a> {
    pub magnitude: &'a [f64],
    pub phase: &'a [f64],
    pub delay_s: f64,
    pub sign: f64,
}

/// Bulk delay (s) of a wrapped phase curve: median local group delay over
/// steps that resolve it (|Δφ| < 90°) and carry energy (within 20 dB of the
/// maximum). Mirrors `bulkDelaySeconds` in band-evaluator/grid.ts.
fn bulk_delay(freq: &[f64], phase: &[f64], mag: &[f64]) -> f64 {
    let max_db = mag.iter().cloned().filter(|v| v.is_finite()).fold(f64::NEG_INFINITY, f64::max);
    let wrap = |d: f64| ((d + 180.0) % 360.0 + 360.0) % 360.0 - 180.0;
    let mut gds: Vec<f64> = Vec::new();
    for i in 1..freq.len() {
        if !(mag[i] > max_db - 20.0 && mag[i - 1] > max_db - 20.0) { continue; }
        let d = wrap(phase[i] - phase[i - 1]);
        let df = freq[i] - freq[i - 1];
        if d.abs() < 90.0 && df > 0.0 { gds.push(-d / (360.0 * df)); }
    }
    if gds.len() < 8 { return 0.0; }
    gds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let t = gds[gds.len() / 2];
    if t.abs() < 5e-4 { 0.0 } else { t }
}

/// Sum the parts on the linear grid at `fft_size`, IFFT, return the real
/// impulse (1/N-normalized). Each part's own bulk delay is removed before
/// interpolating its wrapped phase (slow residual) and restored exactly per
/// bin, together with its alignment delay and sign.
fn ifft_parts(freq: &[f64], parts: &[SpectrumPart], sample_rate: f64, fft_size: usize) -> Vec<f64> {
    let n_bins = fft_size / 2 + 1; // positive freq bins (DC to Nyquist)
    let nyq = sample_rate / 2.0;
    let mut acc = vec![Complex64::new(0.0, 0.0); n_bins];
    for p in parts {
        let tau_b = bulk_delay(freq, p.phase, p.magnitude);
        let wrap = |d: f64| ((d + 180.0) % 360.0 + 360.0) % 360.0 - 180.0;
        let resid: Vec<f64> = p.phase.iter().zip(freq)
            .map(|(&ph, &f)| wrap(ph + 360.0 * f * tau_b)).collect();
        // Interpolate onto linear grid: 0 Hz to Nyquist
        let (_grid_freq, mut grid_mag, grid_phase_opt) =
            interpolate_linear_grid(freq, p.magnitude, Some(&resid), n_bins, sample_rate);
        let mut grid_phase = grid_phase_opt.expect("phase must be present when Some(phase) was passed");
        extend_above_grid(freq, &resid, &mut grid_mag, &mut grid_phase, sample_rate);
        let tau = tau_b + p.delay_s;
        for i in 0..n_bins {
            let f = nyq * i as f64 / (n_bins - 1) as f64;
            let amp = p.sign * 10.0_f64.powf(grid_mag[i] / 20.0);
            let ph_rad = (grid_phase[i] - 360.0 * f * tau).to_radians();
            acc[i] += Complex64::from_polar(amp, ph_rad);
        }
    }

    // DC and Nyquist must be real for a real time-domain signal — project
    // them onto the real axis explicitly.
    let mut spectrum: Vec<Complex64> = Vec::with_capacity(fft_size);
    for (i, c) in acc.iter().enumerate() {
        if i == 0 || i == n_bins - 1 {
            spectrum.push(Complex64::new(c.re, 0.0));
        } else {
            spectrum.push(*c);
        }
    }
    // Mirror for negative frequencies (conjugate symmetry): bins n_bins..fft_size
    for i in 1..(fft_size - n_bins + 1) {
        let idx = n_bins - 1 - i;
        spectrum.push(spectrum[idx].conj());
    }

    // IFFT (rustfft does not normalize)
    let mut engine = FftEngine::new();
    engine.fft_inverse(&mut spectrum);
    let norm = 1.0 / fft_size as f64;
    spectrum.iter().map(|c| c.re * norm).collect()
}

/// b141.57 (audit 2026-10-01 M7): above the last measured frequency the
/// linear-grid interpolation held the magnitude flat and FROZE the phase up
/// to Nyquist — a band with zero group delay, i.e. a spike at t = 0 (20 % of
/// the peak for a 2 ms measurement ending at 20 kHz). Continue the phase with
/// the last group delay and fade the magnitude out over one octave
/// (raised cosine) instead. Callers that already extend to Nyquist (noise-
/// floor tail) are untouched: nothing lies above their last point.
fn extend_above_grid(freq: &[f64], phase: &[f64], mag: &mut [f64], ph: &mut [f64], sample_rate: f64) {
    let n = freq.len();
    if n < 2 { return; }
    let n_bins = mag.len();
    let nyq = sample_rate / 2.0;
    let f_last = freq[n - 1];
    if f_last >= nyq * 0.999 { return; }
    // Group delay from the highest steps that still resolve it (|Δφ| < 90°):
    // the last steps of a log grid can alias a few ms of delay (2 ms over a
    // 290 Hz step at 20 kHz is 209°, read as −151° — the wrong sign).
    let wrap = |d: f64| ((d + 180.0) % 360.0 + 360.0) % 360.0 - 180.0;
    let mut gds: Vec<f64> = Vec::new();
    for i in (1..n).rev() {
        let d = wrap(phase[i] - phase[i - 1]);
        let df = freq[i] - freq[i - 1];
        if d.abs() < 90.0 && df > 0.0 { gds.push(d / df); }
        if gds.len() >= 16 { break; }
    }
    if gds.is_empty() { return; }
    gds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let slope = gds[gds.len() / 2]; // deg per Hz
    let f_end = (2.0 * f_last).min(nyq);
    for k in 0..n_bins {
        let f = nyq * k as f64 / (n_bins - 1) as f64;
        if f <= f_last { continue; }
        ph[k] = phase[n - 1] + slope * (f - f_last);
        let fade = if f >= f_end { 0.0 } else {
            0.5 * (1.0 + (std::f64::consts::PI * (f - f_last) / (f_end - f_last)).cos())
        };
        mag[k] = mag[k] + 20.0 * fade.max(1e-15).log10();
    }
}

/// Result of impulse response computation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ImpulseResult {
    /// b141.23 (audit): the time axis is `(i - pre_peak_count) * dt` — a pure
    /// linear ramp that used to be serialised in full: 2.7 MB at 65536 taps,
    /// 11.3 MB at the 262144 the window-growth loop can reach, a third of the
    /// payload, ~15 times per IR render. The frontend rebuilds it from these
    /// two numbers (same treatment `time_ms` got in the FIR payload, b141.6).
    pub dt: f64,
    /// Samples before t=0 in `impulse` / `step` (the pre-peak region).
    pub pre_peak_count: usize,
    /// Impulse response amplitude (normalized: peak = 100%)
    pub impulse: Vec<f64>,
    /// Step response (cumulative sum of impulse, normalized)
    pub step: Vec<f64>,
    /// Raw impulse peak (before normalization). Used by frontend for shared normalization.
    pub raw_peak: f64,
    /// Raw step peak (before normalization). Used by frontend for shared normalization.
    pub step_raw_peak: f64,
}

/// Compute impulse and step response from frequency-domain measurement.
///
/// 1. Interpolate measurement onto linear frequency grid (0..Nyquist)
/// 2. Build complex spectrum H[k] = 10^(mag/20) * e^(j*phase_rad)
/// 3. Mirror for conjugate symmetry
/// 4. IFFT → time-domain impulse response
/// 5. Circular reorder: include pre-peak samples (negative time) from end of buffer
/// 6. Normalize peak to 100%, trim to 0.5% decay threshold
/// 7. Cumulative sum → step response
pub fn compute_impulse_response(
    freq: &[f64],
    magnitude: &[f64],
    phase: &[f64],
    sample_rate: f64,
) -> ImpulseResult {
    compute_sum_impulse_response(
        freq,
        &[SpectrumPart { magnitude, phase, delay_s: 0.0, sign: 1.0 }],
        sample_rate,
    )
}

/// Impulse / step of the coherent sum of `parts` (all on `freq`), each with
/// its own alignment delay and polarity applied on the linear bins (b141.59).
pub fn compute_sum_impulse_response(
    freq: &[f64],
    parts: &[SpectrumPart],
    sample_rate: f64,
) -> ImpulseResult {
    // Choose initial FFT size: next power of 2, at least 4096
    let initial_fft_size = {
        let min_size = 4096usize;
        let desired = (freq.len() * 4).max(min_size);
        desired.next_power_of_two()
    };

    // b141.6 (audit): the IFFT is circular — a causal tail that has not
    // decayed within the window wraps into the end of the buffer, which the
    // pre-peak collection below would render as fake "pre-ringing" before
    // t=0 (e.g. a Q=8 resonator at 30 Hz showed ~46% of peak at negative
    // time). Grow the window until the zone the layout never displays
    // ([half .. 3/4·fft_size]) is at residual level, so neither the post-peak
    // half nor the pre-peak zone contains wrapped energy. Capped: one 2^18
    // IFFT (5.5 s @ 48 kHz) is still ~ms-scale work.
    const MAX_FFT: usize = 1 << 18;
    let mut fft_size = initial_fft_size.min(MAX_FFT);
    let impulse_raw: Vec<f64>;
    let peak: f64;
    loop {
        let raw = ifft_parts(freq, parts, sample_rate, fft_size);
        let p = raw.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let mid_max = raw[fft_size / 2..fft_size * 3 / 4]
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        if mid_max <= p * 0.002 || fft_size >= MAX_FFT {
            impulse_raw = raw;
            peak = p;
            break;
        }
        fft_size = (fft_size * 4).min(MAX_FFT);
    }

    // Time step
    let dt = 1.0 / sample_rate;

    if peak <= 0.0 {
        return ImpulseResult {
            dt,
            pre_peak_count: 0,
            impulse: vec![0.0],
            step: vec![0.0],
            raw_peak: 0.0,
            step_raw_peak: 0.0,
        };
    }

    // Find peak index in the raw IFFT buffer
    let peak_idx = impulse_raw
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    // --- Determine pre-peak region (negative time) ---
    // The IFFT output is circular. Samples near the end of the buffer
    // represent the "pre-impulse" region (negative time).
    // We include these samples to show what happens before the peak.
    let threshold_raw = peak * 0.002; // 0.2% of peak (-54 dB) — catch subtle pre-ringing
    let max_pre = fft_size / 4; // max 25% of buffer as pre-peak
    let mut pre_peak_count = 0usize;

    for i in 0..max_pre {
        let idx = fft_size - 1 - i;
        if idx <= peak_idx + 1 {
            break; // don't overlap with forward data
        }
        if impulse_raw[idx].abs() > threshold_raw {
            pre_peak_count = i + 1;
        }
    }
    // Always include at least 50ms of pre-peak context (linear-phase filters and masking zone need this)
    let min_pre = ((sample_rate * 0.050) as usize).min(max_pre);
    pre_peak_count = pre_peak_count.max(min_pre);

    // --- Post-peak: full half buffer, no trimming ---
    // Frontend handles view range via fitData (±30ms around peak)
    let half = fft_size / 2;
    let impulse_norm_full: Vec<f64> = impulse_raw.iter().map(|v| (v / peak) * 100.0).collect();
    let trim_end = half;

    // --- Build output arrays ---
    // Layout: [pre-peak from end of buffer] + [0..trim_end from start of buffer]
    let total_len = pre_peak_count + trim_end;
    let pre_start = fft_size - pre_peak_count;

    let mut impulse_out = Vec::with_capacity(total_len);
    let mut raw_reordered = Vec::with_capacity(total_len);

    // Pre-peak samples: buffer indices [pre_start..fft_size]
    // Their "true" time = (index - fft_size) * dt (negative values)
    for i in 0..pre_peak_count {
        let buf_idx = pre_start + i;
        impulse_out.push(impulse_norm_full[buf_idx]);
        raw_reordered.push(impulse_raw[buf_idx]);
    }

    // Forward samples: buffer indices [0..trim_end]
    // Their time = index * dt
    for i in 0..trim_end {
        impulse_out.push(impulse_norm_full[i]);
        raw_reordered.push(impulse_raw[i]);
    }

    // --- Step response: cumulative sum of reordered raw impulse ---
    // Normalized by its own peak for readability (both IR and Step peak at 100%).
    // Time alignment (IR peak vs step 50%) handled on frontend.
    let mut step = Vec::with_capacity(total_len);
    let mut acc = 0.0;
    for v in &raw_reordered {
        acc += v;
        step.push(acc);
    }

    let step_peak = step.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let step_norm: Vec<f64> = if step_peak > 0.0 {
        step.iter().map(|v| (v / step_peak) * 100.0).collect()
    } else {
        step
    };

    ImpulseResult {
        dt,
        pre_peak_count,
        impulse: impulse_out,
        step: step_norm,
        raw_peak: peak,
        step_raw_peak: if step_peak > 0.0 { step_peak } else { 0.0 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_impulse_flat_spectrum() {
        // Flat magnitude, zero phase → should produce a delta-like impulse
        let n = 100;
        let freq: Vec<f64> = (0..n).map(|i| 20.0 + i as f64 * 200.0).collect();
        let mag: Vec<f64> = vec![0.0; n]; // 0 dB = unity
        let phase: Vec<f64> = vec![0.0; n]; // zero phase

        let result = compute_impulse_response(&freq, &mag, &phase, 48000.0);

        assert_eq!(result.step.len(), result.impulse.len());

        // Find peak in the output
        let peak_idx = result
            .impulse
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        // Peak value should be 100 (percent)
        let peak_val = result.impulse[peak_idx];
        assert!((peak_val - 100.0).abs() < 1.0, "Peak should be ~100%, got {}", peak_val);

        // Time at peak should be near 0
        let time_at = |i: usize| (i as f64 - result.pre_peak_count as f64) * result.dt;
        let peak_time = time_at(peak_idx);
        assert!(peak_time.abs() < 0.001, "Peak time should be near 0, got {}", peak_time);

        // Should have some negative time values (pre-peak region)
        assert!(time_at(0) < 0.0, "First time should be negative, got {}", time_at(0));
    }

    #[test]
    fn test_impulse_trimmed() {
        // Verify trimming: result should be shorter than full FFT size
        let n = 50;
        let freq: Vec<f64> = (0..n).map(|i| 20.0 + i as f64 * 400.0).collect();
        let mag: Vec<f64> = vec![0.0; n];
        let phase: Vec<f64> = vec![0.0; n];

        let result = compute_impulse_response(&freq, &mag, &phase, 48000.0);

        // Trimmed length should be less than full FFT size (4096 at minimum)
        assert!(result.impulse.len() < 4096, "Should be trimmed, got len={}", result.impulse.len());
        assert_eq!(result.step.len(), result.impulse.len());
        // Time axis starts before t=0 (pre-peak region)
        assert!(result.pre_peak_count > 0, "expected a pre-peak region");
    }

    /// b141.23: the axis is now derived on the frontend as
    /// `(i - pre_peak_count) * dt`. Pin the two numbers that define it.
    #[test]
    fn test_impulse_time_monotonic() {
        // Verify the derived time axis is monotonically increasing
        let n = 100;
        let freq: Vec<f64> = (0..n).map(|i| 20.0 + i as f64 * 200.0).collect();
        let mag: Vec<f64> = vec![0.0; n];
        let phase: Vec<f64> = vec![0.0; n];

        let result = compute_impulse_response(&freq, &mag, &phase, 48000.0);

        let time_at = |i: usize| (i as f64 - result.pre_peak_count as f64) * result.dt;
        assert!(result.dt > 0.0, "dt must be positive, got {}", result.dt);
        assert!((result.dt - 1.0 / 48_000.0).abs() < 1e-15, "dt must be 1/sr");
        for i in 1..result.impulse.len() {
            assert!(
                time_at(i) > time_at(i - 1),
                "Time should be monotonically increasing at index {}: {} <= {}",
                i, time_at(i), time_at(i - 1)
            );
        }
    }

    /// b141.6 (audit MEDIUM): a slowly-decaying causal system (high-Q LF
    /// resonator) whose tail exceeds the FFT window used to wrap around the
    /// IFFT buffer and get rendered as "pre-ringing" before t=0 — a false
    /// visual signal for users judging linear vs min phase by pre-ring.
    /// The FFT window must grow until the tail actually decays.
    #[test]
    fn test_causal_resonator_has_no_fake_preringing() {
        // 2nd-order LP resonator at f0=30 Hz, Q=8 → decay tau ≈ 2Q/w0 ≈ 85 ms,
        // comparable to the default 4096-sample window at 48 kHz (85.3 ms).
        let f0 = 30.0_f64;
        let q = 8.0_f64;
        let n = 500;
        let freq: Vec<f64> = (0..n)
            .map(|i| 10.0 * (20000.0_f64 / 10.0).powf(i as f64 / (n - 1) as f64))
            .collect();
        let mut mag = Vec::with_capacity(n);
        let mut phase = Vec::with_capacity(n);
        for &f in &freq {
            // H(jw) = w0^2 / (w0^2 - w^2 + j*w0*w/Q), normalized s-domain
            let r = f / f0;
            let re = 1.0 - r * r;
            let im = r / q;
            let denom = re * re + im * im;
            let h_re = re / denom;
            let h_im = -im / denom;
            let amp = (h_re * h_re + h_im * h_im).sqrt();
            mag.push(20.0 * amp.log10());
            phase.push(h_im.atan2(h_re).to_degrees());
        }

        let result = compute_impulse_response(&freq, &mag, &phase, 48000.0);

        // The system is strictly causal: anything before t = -10 ms must be
        // residual-level only. With the wrap bug the tail re-entered at up to
        // ~40% of peak.
        let time_at = |i: usize| (i as f64 - result.pre_peak_count as f64) * result.dt;
        let max_pre: f64 = result
            .impulse
            .iter()
            .enumerate()
            .filter(|(i, _)| time_at(*i) < -0.010)
            .map(|(_, v)| v.abs())
            .fold(0.0, f64::max);
        assert!(
            max_pre <= 1.0,
            "causal resonator shows fake pre-ringing: {:.2}% of peak before -10 ms",
            max_pre
        );
    }

    #[test]
    fn test_linphase_bandpass_step_at_ir_peak() {
        // Linear-phase bandpass: step should be ~50% at IR peak time
        let n = 512;
        let freq: Vec<f64> = (0..n).map(|i| 20.0 + i as f64 * (20000.0 - 20.0) / (n - 1) as f64).collect();
        let mut mag = vec![-60.0_f64; n];
        let mut phase = vec![0.0_f64; n];
        let delay = 0.005; // 5ms
        for i in 0..n {
            let f = freq[i];
            // Bandpass 200-3000 Hz
            if f >= 200.0 && f <= 3000.0 { mag[i] = 0.0; }
            // Linear phase (constant group delay)
            phase[i] = -360.0 * f * delay;
        }

        let result = compute_impulse_response(&freq, &mag, &phase, 48000.0);

        // Find IR peak
        let mut ir_peak_idx = 0;
        let mut ir_peak_val = 0.0_f64;
        for (i, &v) in result.impulse.iter().enumerate() {
            if v.abs() > ir_peak_val { ir_peak_val = v.abs(); ir_peak_idx = i; }
        }

        // Find Step peak
        let mut st_peak_idx = 0;
        let mut st_peak_val = 0.0_f64;
        for (i, &v) in result.step.iter().enumerate() {
            if v.abs() > st_peak_val { st_peak_val = v.abs(); st_peak_idx = i; }
        }

        let step_at_ir_peak = result.step[ir_peak_idx];
        let ir_at_step_peak = result.impulse[st_peak_idx].abs();

        let time_ms = |i: usize| (i as f64 - result.pre_peak_count as f64) * result.dt * 1000.0;
        eprintln!("IR peak: idx={} t={:.3}ms val={:.1}%", ir_peak_idx, time_ms(ir_peak_idx), ir_peak_val);
        eprintln!("Step peak: idx={} t={:.3}ms val={:.1}%", st_peak_idx, time_ms(st_peak_idx), st_peak_val);
        eprintln!("Step at IR peak: {:.1}% (should be ~50%)", step_at_ir_peak);
        eprintln!("IR at Step peak: {:.1}% (should be ~0 for symmetric)", ir_at_step_peak);
        eprintln!("IR peak time != Step peak time? {} != {} → {}", ir_peak_idx, st_peak_idx, ir_peak_idx != st_peak_idx);

        // Step at IR peak should be roughly 50% of step peak (=100%)
        // For linear-phase bandpass: cumsum reaches ~half at the symmetric center
        assert!(
            step_at_ir_peak > 5.0 && step_at_ir_peak < 50.0,
            "Step at IR peak should be ~10-50% of step peak, got {:.1}%",
            step_at_ir_peak
        );

        // IR and Step peaks should NOT be at the same time for bandpass
        assert_ne!(ir_peak_idx, st_peak_idx, "IR and Step peaks should be at different times for bandpass");
    }

    /// b141.57 (audit 2026-10-01 M7): a measurement that ends at 20 kHz and
    /// keeps a 2 ms time of flight must not grow a spike at t = 0. Reference:
    /// the same response measured up to 23.99 kHz (nothing to extend). The
    /// frozen phase gave 22.7 % at t = 0; the reference has 1.1 % (wrap of the
    /// 40 Hz ringing, unrelated).
    #[test]
    fn no_false_spike_at_zero_for_a_grid_ending_below_nyquist() {
        let sr = 48_000.0;
        let tau = 0.002;
        let near_zero = |top: f64| -> f64 {
            let nn = (48.0 * (top / 20.0f64).log2()).round() as usize + 1;
            let freq: Vec<f64> = (0..nn).map(|i| 20.0 * (top / 20.0f64).powf(i as f64 / (nn - 1) as f64)).collect();
            let (mag, ph): (Vec<f64>, Vec<f64>) = freq.iter().map(|&f| {
                // BW4 HP 40 Hz magnitude and phase, plus the delay.
                let w = f / 40.0;
                let m = w.powi(4) / (1.0 + w.powi(8)).sqrt();
                let mut p = 0.0;
                for k in 0..2 {
                    let q = 1.0 / (2.0 * (std::f64::consts::PI * (2 * k + 1) as f64 / 8.0).sin());
                    p += 180.0 - (w / q).atan2(1.0 - w * w).to_degrees();
                }
                let p = p - 360.0 * f * tau;
                (20.0 * m.log10(), ((p + 180.0) % 360.0 + 360.0) % 360.0 - 180.0)
            }).unzip();
            let r = compute_impulse_response(&freq, &mag, &ph, sr);
            let t0 = r.pre_peak_count;
            r.impulse[t0.saturating_sub(5)..t0 + 5].iter().fold(0.0_f64, |a, &v| a.max(v.abs()))
        };
        let (short, full) = (near_zero(20_000.0), near_zero(23_990.0));
        println!("|impulse| near t=0: grid to 20 kHz {short:.2} %, to 23.99 kHz {full:.2} %");
        assert!(short < full + 1.0, "spike at t=0: {short:.2} % vs {full:.2} % reference");
    }
}

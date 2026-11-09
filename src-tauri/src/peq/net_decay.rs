//! b141.139: what a PEQ filter does to the decay of the corrected response.
//!
//! A filter's own ring (q_envelope::ring_ms) is not the defect: a cut on a
//! resonance of the speaker or the room cancels that resonance's ring, a boost
//! into a dip made by a reflection adds a ring nothing cancels. So the decay is
//! judged on measurement × PEQ, with the filter and without it, through the
//! ear's own filter at the filter's resonance — a real 4th-order gammatone of
//! bandwidth 1.019·ERB (Glasberg & Moore 1990) — as the time from the energy
//! peak until the Schroeder integral has fallen by 20 and 30 dB. The ear's own
//! decay (a flat response through the same filter) is the yardstick.
//! Calibration (`tests/peq_net_decay.rs`, to −30 dB, ear 31 ms at 50 Hz): an
//! exact cut on a +6 dB Q 5 resonance at 50 Hz takes 133 ms back to 31; a
//! +8 dB Q 3 boost into the dip of a weaker reflection (minimum phase, so the
//! boost is its inverse) shortens 47 → 29, into the dip of a louder one
//! (excess phase) lengthens 32 → 112, and 4 Hz off the dip's centre 45 → 80.

use num_complex::Complex64;
use serde::Serialize;

use super::{PeqBand, PeqFilterType};
use crate::dsp::fft::FftEngine;

/// Analysis rate and length: 2.7 s of decay, 0.37 Hz bins.
const SR: f64 = 96_000.0;
const N: usize = 1 << 18;
/// Decay levels below the energy peak, dB.
pub const LEVELS_DB: [f64; 2] = [20.0, 30.0];

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct NetDecay {
    /// The filter's resonance: f, or fp for a Linkwitz transform.
    pub freq_hz: f64,
    /// Decay to −20 / −30 dB, ms: the ear alone, the corrected response
    /// without this filter, with it.
    pub ear_ms: [f64; 2],
    pub without_ms: [f64; 2],
    pub with_ms: [f64; 2],
}

fn erb(f: f64) -> f64 { 24.7 * (4.37 * f / 1000.0 + 1.0) }

/// Real gammatone t³·e^(−2πbt)·cos(2πfc·t): both halves of its spectrum —
/// at LF b is comparable to fc.
fn gammatone(f: f64, fc: f64) -> Complex64 {
    let b = 1.019 * erb(fc);
    let half = |x: f64| Complex64::new(1.0, x / b).powi(-4);
    0.5 * (half(f - fc) + half(f + fc))
}

fn resonance_hz(p: &PeqBand) -> f64 {
    match p.filter_type {
        PeqFilterType::LinkwitzTransform => p.freq_hz / 10f64.powf(p.gain_db / 40.0),
        _ => p.freq_hz,
    }
}

fn bins() -> Vec<f64> { (0..=N / 2).map(|k| k as f64 * SR / N as f64).collect() }

/// The measurement on the analysis bins: magnitude and unwrapped phase
/// interpolated over log f; beyond its ends both hold — a roll-off added
/// there would be a high-pass of its own, ringing into the analysis (−24 dB/oct
/// below 10 Hz took a 50 Hz decay from 43 to 49 ms).
fn measurement_on_bins(freq: &[f64], mag_db: &[f64], phase_deg: &[f64], f: &[f64]) -> Vec<Complex64> {
    let mut ph: Vec<f64> = phase_deg.iter().map(|p| p.to_radians()).collect();
    for i in 1..ph.len() {
        let d = ph[i] - ph[i - 1];
        ph[i] -= (d / std::f64::consts::TAU).round() * std::f64::consts::TAU;
    }
    let lf: Vec<f64> = freq.iter().map(|x| x.ln()).collect();
    let (f0, f1) = (freq[0], freq[freq.len() - 1]);
    let n = freq.len();
    f.iter().map(|&x| {
        if x <= 0.0 { return Complex64::new(0.0, 0.0); }
        let (m, p) = if x < f0 {
            (mag_db[0], ph[0])
        } else if x > f1 {
            (mag_db[n - 1], ph[n - 1])
        } else {
            let l = x.ln();
            let j = lf.partition_point(|&v| v <= l).clamp(1, n - 1);
            let t = (l - lf[j - 1]) / (lf[j] - lf[j - 1]);
            (mag_db[j - 1] + t * (mag_db[j] - mag_db[j - 1]), ph[j - 1] + t * (ph[j] - ph[j - 1]))
        };
        Complex64::from_polar(10f64.powf(m / 20.0), p)
    }).collect()
}

/// One filter's complex response on the bins, as the file realises it; the
/// export rate's Nyquist caps the evaluation.
fn peq_on_bins(p: &PeqBand, f: &[f64], export_sr: f64) -> Vec<Complex64> {
    let fmax = export_sr / 2.0 * 0.999;
    let capped: Vec<f64> = f.iter().map(|&x| x.min(fmax)).collect();
    let (mag, ph) = crate::fir::band::peq_response_as_exported(&capped, std::slice::from_ref(p), export_sr);
    mag.iter().zip(&ph).map(|(m, d)| Complex64::from_polar(10f64.powf(m / 20.0), d.to_radians())).collect()
}

/// Decay (ms) of `h` through the ear's filter at `fc` to each of LEVELS_DB.
fn decay_ms(fft: &mut FftEngine, h: &[Complex64], f: &[f64], fc: f64) -> [f64; 2] {
    let mut buf = vec![Complex64::new(0.0, 0.0); N];
    for (k, (&hk, &fk)) in h.iter().zip(f).enumerate() {
        buf[k] = hk * gammatone(fk, fc);
    }
    buf[N / 2].im = 0.0;
    buf[0].im = 0.0;
    for k in 1..N / 2 { buf[N - k] = buf[k].conj(); }
    fft.fft_inverse(&mut buf);
    let e: Vec<f64> = buf.iter().map(|c| c.re * c.re).collect();
    let ipk = e.iter().enumerate().fold(0, |b, (i, &v)| if v > e[b] { i } else { b });
    // The half after the peak, read circularly: the other half is the wrapped
    // pre-response. b141.144: a measurement whose delay removal overshot puts
    // the peak before t = 0, i.e. at the end of the buffer — cut at N, the
    // tail was 99 samples and the −30 dB decay came back as infinity (JSON
    // null), which threw in the «мс» tooltip and froze the plot.
    let tail: Vec<f64> = (0..N / 2).map(|i| e[(ipk + i) % N]).collect();
    let mut edc = vec![0.0; tail.len()];
    let mut acc = 0.0;
    for i in (0..tail.len()).rev() { acc += tail[i]; edc[i] = acc; }
    let total = edc[0].max(f64::MIN_POSITIVE);
    LEVELS_DB.map(|db| {
        let lim = total * 10f64.powf(-db / 10.0);
        // Not reached in half the window: at least that long — never infinity.
        edc.iter().position(|&v| v < lim).map_or(tail.len() as f64 / SR * 1000.0, |i| i as f64 / SR * 1000.0)
    })
}

/// Per filter (same order as `peq`): its effect on the decay, or None when it
/// is disabled, has no gain, or its resonance is not at least half an octave
/// inside the measurement (or above 0.45 of the analysis rate).
pub fn net_decay(
    freq: &[f64], mag_db: &[f64], phase_deg: &[f64], peq: &[PeqBand], export_sr: f64,
) -> Result<Vec<Option<NetDecay>>, String> {
    let n = freq.len();
    if n < 2 || mag_db.len() != n || phase_deg.len() != n {
        return Err("net_decay: measurement arrays differ in length".into());
    }
    if freq.windows(2).any(|w| !(w[1] > w[0]) || !(w[0] > 0.0)) {
        return Err("net_decay: measurement frequencies must rise above 0".into());
    }
    let f = bins();
    let meas = measurement_on_bins(freq, mag_db, phase_deg, &f);
    let active = |p: &PeqBand| p.enabled && p.gain_db.abs() >= 0.05;
    let each: Vec<Option<Vec<Complex64>>> = peq.iter()
        .map(|p| active(p).then(|| peq_on_bins(p, &f, export_sr))).collect();
    let mut all = meas.clone();
    for r in each.iter().flatten() {
        for (a, x) in all.iter_mut().zip(r) { *a *= x; }
    }
    let flat = vec![Complex64::new(1.0, 0.0); f.len()];
    let (lo, hi) = (freq[0] * 2f64.sqrt(), (freq[n - 1] / 2f64.sqrt()).min(0.45 * SR));
    let mut fft = FftEngine::new();
    Ok(peq.iter().zip(&each).map(|(p, r)| {
        let r = r.as_ref()?;
        let fc = resonance_hz(p);
        if !(fc >= lo && fc <= hi) { return None; }
        let without: Vec<Complex64> = all.iter().zip(r).map(|(a, x)| a / x).collect();
        Some(NetDecay {
            freq_hz: fc,
            ear_ms: decay_ms(&mut fft, &flat, &f, fc),
            without_ms: decay_ms(&mut fft, &without, &f, fc),
            with_ms: decay_ms(&mut fft, &all, &f, fc),
        })
    }).collect())
}

/// Test hook: decay of an arbitrary response (given on the analysis bins)
/// through the ear's filter at `fc`.
#[doc(hidden)]
pub fn decay_of(h: impl Fn(f64) -> Complex64, fc: f64) -> [f64; 2] {
    let f = bins();
    let hv: Vec<Complex64> = f.iter().map(|&x| h(x)).collect();
    decay_ms(&mut FftEngine::new(), &hv, &f, fc)
}

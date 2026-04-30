use chrono::{SecondsFormat, Utc};
use serde::{Deserialize, Serialize};

use crate::io::Measurement;

// Analysis tuning constants. Picked deliberately tight to avoid false
// positives on real measurements; users tweak by ignoring findings, not by
// retuning. Don't expose these as user-facing parameters.
const NOISE_FLOOR_STD_DB: f64 = 2.0;
const NOISE_FLOOR_OCTAVE_RUN: f64 = 1.0; // need ≥1 oct of "flat" run
// Flat region only counts as noise floor if its mean is at least this many
// dB below the median of the rest of the band — otherwise a flat speaker
// response would be misidentified as noise.
const NOISE_FLOOR_DROP_DB: f64 = 15.0;
const LF_ROLLOFF_SLOPE_DB_PER_OCT: f64 = 18.0;
const HF_CLIFF_SLOPE_DB_PER_OCT: f64 = 24.0;
const HF_CLIFF_TAIL_OCTAVES: f64 = 1.5;
const RESONANCE_LOCAL_OCTAVE: f64 = 1.0 / 3.0;
const RESONANCE_MIN_AMPLITUDE_DB: f64 = 1.5;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum Severity {
    Info,
    Warning,
    Error,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type", content = "value")]
pub enum ActionType {
    SetOptLowerBound(f64),
    SetOptUpperBound(f64),
    AddExclusionZone { low_hz: f64, high_hz: f64 },
    ApplySmoothing(String),
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Recommendation {
    pub action: ActionType,
    pub label: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Finding {
    pub id: String,
    pub severity: Severity,
    pub title: String,
    pub description: String,
    pub freq_range: Option<(f64, f64)>,
    pub recommendations: Vec<Recommendation>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AnalysisResult {
    pub timestamp: String,
    pub app_version: String,
    pub findings: Vec<Finding>,
}

#[derive(Debug, Clone)]
pub struct NoiseFloorResult {
    pub low: Option<Finding>,
    pub high: Option<Finding>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn round_freq(f: f64) -> f64 {
    if f >= 1000.0 {
        (f / 10.0).round() * 10.0
    } else if f >= 100.0 {
        f.round()
    } else {
        (f * 10.0).round() / 10.0
    }
}

fn mean_std(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

/// Slice [lo_idx, hi_idx) where freq is within [lo, hi].
fn freq_window(freq: &[f64], lo: f64, hi: f64) -> (usize, usize) {
    let lo_idx = freq.partition_point(|&f| f < lo);
    let hi_idx = freq.partition_point(|&f| f <= hi);
    (lo_idx, hi_idx)
}

// ---------------------------------------------------------------------------
// Detector 1: noise floor (low and high)
// ---------------------------------------------------------------------------

pub fn detect_noise_floor(freq: &[f64], magnitude: &[f64]) -> NoiseFloorResult {
    if freq.len() < 16 {
        return NoiseFloorResult { low: None, high: None };
    }
    NoiseFloorResult {
        low: detect_noise_floor_low(freq, magnitude),
        high: detect_noise_floor_high(freq, magnitude),
    }
}

fn window_std(freq: &[f64], magnitude: &[f64], center: f64) -> Option<f64> {
    let half = 2f64.sqrt(); // ±1/2 octave
    let lo = center / half;
    let hi = center * half;
    let (a, b) = freq_window(freq, lo, hi);
    if b.saturating_sub(a) < 4 {
        return None;
    }
    let (_, std) = mean_std(&magnitude[a..b]);
    Some(std)
}

fn window_mean(freq: &[f64], magnitude: &[f64], center: f64) -> Option<f64> {
    let half = 2f64.sqrt();
    let lo = center / half;
    let hi = center * half;
    let (a, b) = freq_window(freq, lo, hi);
    if b.saturating_sub(a) < 4 {
        return None;
    }
    let (m, _) = mean_std(&magnitude[a..b]);
    Some(m)
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v: Vec<f64> = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

fn detect_noise_floor_low(freq: &[f64], magnitude: &[f64]) -> Option<Finding> {
    let f_min = freq.first().copied().unwrap_or(20.0).max(1.0);
    let mut log = f_min.log2();
    let log_max = freq[freq.len() / 2].log2();
    let step = 0.05;
    let mut last_flat: Option<f64> = None;
    while log <= log_max {
        let center = 2f64.powf(log);
        match window_std(freq, magnitude, center) {
            Some(s) if s < NOISE_FLOOR_STD_DB => last_flat = Some(center),
            Some(_) => break,
            None => break,
        }
        log += step;
    }
    let f_low = last_flat?;
    if (f_low / f_min).log2() < NOISE_FLOOR_OCTAVE_RUN {
        return None;
    }
    let mean = window_mean(freq, magnitude, f_low).unwrap_or(0.0);
    // Reject if the flat region isn't actually below the rest of the band —
    // otherwise a perfectly flat measurement would falsely trigger.
    let rest_idx = freq.partition_point(|&f| f <= f_low * 1.2);
    if rest_idx < freq.len() {
        let rest_median = median(&magnitude[rest_idx..]);
        if rest_median - mean < NOISE_FLOOR_DROP_DB {
            return None;
        }
    } else {
        return None;
    }
    let f_round = round_freq(f_low);
    Some(Finding {
        id: "noise_floor_low".into(),
        severity: Severity::Warning,
        title: format!("Шумовой пол ниже {:.0} Гц", f_round),
        description: format!(
            "SPL стабилизируется около {:.1} дБ — ниже этой частоты данные не несут полезной информации.",
            mean
        ),
        freq_range: Some((f_min, f_low)),
        recommendations: vec![Recommendation {
            action: ActionType::SetOptLowerBound(f_round),
            label: format!("Установить нижнюю границу оптимизации = {:.0} Гц", f_round),
        }],
    })
}

fn detect_noise_floor_high(freq: &[f64], magnitude: &[f64]) -> Option<Finding> {
    let f_max = freq.last().copied().unwrap_or(20000.0);
    let mut log = f_max.log2();
    let log_min = freq[freq.len() / 2].log2();
    let step = 0.05;
    let mut first_flat: Option<f64> = None;
    while log >= log_min {
        let center = 2f64.powf(log);
        match window_std(freq, magnitude, center) {
            Some(s) if s < NOISE_FLOOR_STD_DB => first_flat = Some(center),
            Some(_) => break,
            None => break,
        }
        log -= step;
    }
    let f_high = first_flat?;
    if (f_max / f_high).log2() < NOISE_FLOOR_OCTAVE_RUN {
        return None;
    }
    let mean = window_mean(freq, magnitude, f_high).unwrap_or(0.0);
    let rest_idx_end = freq.partition_point(|&f| f < f_high / 1.2);
    if rest_idx_end > 0 {
        let rest_median = median(&magnitude[..rest_idx_end]);
        if rest_median - mean < NOISE_FLOOR_DROP_DB {
            return None;
        }
    } else {
        return None;
    }
    let f_round = round_freq(f_high);
    Some(Finding {
        id: "noise_floor_high".into(),
        severity: Severity::Warning,
        title: format!("Шумовой пол выше {:.0} Гц", f_round),
        description: format!(
            "SPL стабилизируется около {:.1} дБ — выше этой частоты данные не несут полезной информации.",
            mean
        ),
        freq_range: Some((f_high, f_max)),
        recommendations: vec![Recommendation {
            action: ActionType::SetOptUpperBound(f_round),
            label: format!("Установить верхнюю границу оптимизации = {:.0} Гц", f_round),
        }],
    })
}

// ---------------------------------------------------------------------------
// Detector 2: low-frequency rolloff (window-induced)
// ---------------------------------------------------------------------------

/// Average slope (dB/octave) over a freq range using a least-squares fit.
fn slope_db_per_oct(freq: &[f64], magnitude: &[f64], lo_idx: usize, hi_idx: usize) -> f64 {
    if hi_idx <= lo_idx + 1 {
        return 0.0;
    }
    let n = (hi_idx - lo_idx) as f64;
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for i in lo_idx..hi_idx {
        let x = freq[i].max(1e-9).log2();
        let y = magnitude[i];
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-12 {
        return 0.0;
    }
    (n * sxy - sx * sy) / denom
}

/// True if [lo_idx, hi_idx) has no oscillatory peak/trough wider than
/// RESONANCE_LOCAL_OCTAVE with amplitude swing > RESONANCE_MIN_AMPLITUDE_DB.
/// Tiny ripple from measurement noise must not suppress legitimate cliff or
/// rolloff detection on real (rippled) data.
fn no_resonance(freq: &[f64], magnitude: &[f64], lo_idx: usize, hi_idx: usize) -> bool {
    if hi_idx <= lo_idx + 4 {
        return true;
    }
    let mut sign_changes = 0;
    let mut last_sign: i32 = 0;
    let mut last_extremum_idx = lo_idx;
    let mut last_extremum_y = magnitude[lo_idx];
    for i in (lo_idx + 1)..hi_idx {
        let dy = magnitude[i] - magnitude[i - 1];
        let s = if dy > 1e-3 { 1 } else if dy < -1e-3 { -1 } else { 0 };
        if s != 0 && last_sign != 0 && s != last_sign {
            let span = (freq[i] / freq[last_extremum_idx].max(1e-9)).log2().abs();
            let amp = (magnitude[i - 1] - last_extremum_y).abs();
            if span < RESONANCE_LOCAL_OCTAVE && amp > RESONANCE_MIN_AMPLITUDE_DB {
                sign_changes += 1;
                if sign_changes >= 2 {
                    return false;
                }
            }
            last_extremum_idx = i - 1;
            last_extremum_y = magnitude[i - 1];
        }
        if s != 0 {
            last_sign = s;
        }
    }
    true
}

pub fn detect_lf_rolloff(measurement: &Measurement, noise_floor_low: Option<f64>) -> Option<Finding> {
    let freq = &measurement.freq;
    let mag = &measurement.magnitude;
    if freq.len() < 16 {
        return None;
    }
    // Look at the bottom 2 octaves above f_min for steep monotonic slope.
    let f_min = freq[0].max(1.0);
    let f_lo = noise_floor_low.unwrap_or(f_min);
    let f_hi = (f_lo * 4.0).min(freq[freq.len() / 2]);
    if f_hi <= f_lo * 1.5 {
        return None;
    }
    let (lo_idx, hi_idx) = freq_window(freq, f_lo, f_hi);
    if hi_idx.saturating_sub(lo_idx) < 8 {
        return None;
    }
    let slope = slope_db_per_oct(freq, mag, lo_idx, hi_idx);
    // Rolloff at low f means SPL rises with f → positive slope vs log2(f).
    if slope < LF_ROLLOFF_SLOPE_DB_PER_OCT {
        return None;
    }
    if !no_resonance(freq, mag, lo_idx, hi_idx) {
        return None;
    }
    // f_min usable ≈ where slope crosses out of rolloff. Conservative: top of
    // the analysed window.
    let f_round = round_freq(f_hi);
    if let Some(nf) = noise_floor_low {
        if f_round <= nf * 1.05 {
            return None; // dominated by noise floor warning
        }
    }
    Some(Finding {
        id: "lf_rolloff_window".into(),
        severity: Severity::Warning,
        title: format!("Низкочастотный спад до {:.0} Гц", f_round),
        description: format!(
            "Скорость спада {:.0} дБ/окт без резонансных особенностей — вероятно, ограничение окна замера.",
            slope
        ),
        freq_range: Some((f_lo, f_hi)),
        recommendations: vec![Recommendation {
            action: ActionType::SetOptLowerBound(f_round),
            label: format!("Установить нижнюю границу оптимизации = {:.0} Гц", f_round),
        }],
    })
}

// ---------------------------------------------------------------------------
// Detector 3: high-frequency cliff (anti-aliasing / mic rolloff)
// ---------------------------------------------------------------------------

pub fn detect_hf_cliff(freq: &[f64], magnitude: &[f64], sample_rate: f64) -> Option<Finding> {
    if freq.len() < 16 {
        return None;
    }
    let f_max = freq.last().copied().unwrap_or(20000.0);
    let nyquist = (sample_rate / 2.0).max(1.0);
    let f_top = (nyquist * 0.95).min(f_max);
    let f_start = (f_top / 2f64.powf(HF_CLIFF_TAIL_OCTAVES)).max(8000.0);
    if f_top <= f_start * 1.5 {
        return None;
    }
    // Walk up from f_start: cliff starts where the LOCAL slope (next 1/3
    // octave) first drops below the threshold. Looking only at a short window
    // makes the detection insensitive to flat pre-cliff regions diluting the
    // average. Require the steep slope to persist all the way to f_top.
    let mut log = f_start.log2();
    let log_max = f_top.log2();
    let step = 0.05;
    let mut found: Option<f64> = None;
    while log <= log_max {
        let f_cliff = 2f64.powf(log);
        let f_local = (f_cliff * 2f64.powf(1.0 / 3.0)).min(f_top);
        let (lo_idx, hi_idx) = freq_window(freq, f_cliff, f_local);
        let (full_lo, full_hi) = freq_window(freq, f_cliff, f_top);
        if hi_idx.saturating_sub(lo_idx) < 4 || full_hi.saturating_sub(full_lo) < 6 {
            log += step;
            continue;
        }
        let local_slope = slope_db_per_oct(freq, magnitude, lo_idx, hi_idx);
        let full_slope = slope_db_per_oct(freq, magnitude, full_lo, full_hi);
        if local_slope <= -HF_CLIFF_SLOPE_DB_PER_OCT
            && full_slope <= -HF_CLIFF_SLOPE_DB_PER_OCT
            && no_resonance(freq, magnitude, full_lo, full_hi)
        {
            found = Some(f_cliff);
            break;
        }
        log += step;
    }
    let f_cliff = found?;
    let f_round = round_freq(f_cliff);
    Some(Finding {
        id: "hf_cliff".into(),
        severity: Severity::Info,
        title: format!("Высокочастотный обрыв на {:.0} Гц", f_round),
        description: "Резкий монотонный спад без резонансов — типичная anti-aliasing или микрофонная характеристика.".into(),
        freq_range: Some((f_cliff, f_max)),
        recommendations: vec![Recommendation {
            action: ActionType::SetOptUpperBound(f_round),
            label: format!("Установить верхнюю границу оптимизации = {:.0} Гц", f_round),
        }],
    })
}

// ---------------------------------------------------------------------------
// Tauri command
// ---------------------------------------------------------------------------

#[tauri::command]
pub fn analyze_measurement(measurement: Measurement) -> Result<AnalysisResult, String> {
    let freq = &measurement.freq;
    let mag = &measurement.magnitude;
    if freq.is_empty() || mag.len() != freq.len() {
        return Err("invalid measurement: empty or length mismatch".into());
    }
    let mut findings: Vec<Finding> = Vec::new();
    let nf = detect_noise_floor(freq, mag);
    let nf_low_freq = nf.low.as_ref().and_then(|f| f.freq_range.map(|r| r.1));
    if let Some(f) = nf.low {
        findings.push(f);
    }
    if let Some(f) = nf.high {
        findings.push(f);
    }
    if let Some(f) = detect_lf_rolloff(&measurement, nf_low_freq) {
        findings.push(f);
    }
    let sr = measurement.sample_rate.unwrap_or(48000.0);
    if let Some(f) = detect_hf_cliff(freq, mag, sr) {
        findings.push(f);
    }
    Ok(AnalysisResult {
        timestamp: Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
        app_version: env!("CARGO_PKG_VERSION").into(),
        findings,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::MeasurementMetadata;

    fn log_grid(n: usize, fmin: f64, fmax: f64) -> Vec<f64> {
        let lmin = fmin.log10();
        let lmax = fmax.log10();
        (0..n)
            .map(|i| 10f64.powf(lmin + (lmax - lmin) * (i as f64) / (n as f64 - 1.0)))
            .collect()
    }

    fn meas(freq: Vec<f64>, magnitude: Vec<f64>) -> Measurement {
        Measurement {
            name: "test".into(),
            source_path: None,
            sample_rate: Some(48000.0),
            freq,
            magnitude,
            phase: None,
            metadata: MeasurementMetadata::default(),
        }
    }

    #[test]
    fn flat_response_has_no_findings() {
        let freq = log_grid(512, 20.0, 20000.0);
        let mag: Vec<f64> = freq.iter().map(|_| 0.0).collect();
        let nf = detect_noise_floor(&freq, &mag);
        // Flat ≠ noise floor: drop-vs-rest check rejects it.
        assert!(nf.low.is_none(), "flat response triggered noise_floor_low");
        assert!(nf.high.is_none(), "flat response triggered noise_floor_high");
        let m = meas(freq.clone(), mag.clone());
        assert!(detect_lf_rolloff(&m, None).is_none());
        assert!(detect_hf_cliff(&freq, &mag, 48000.0).is_none());
    }

    #[test]
    fn noise_floor_low_is_detected() {
        // Floor at -50 dB below 30 Hz, then rises to 0 dB at 100 Hz, flat to 20k.
        let freq = log_grid(1024, 5.0, 20000.0);
        let mag: Vec<f64> = freq
            .iter()
            .map(|&f| {
                if f < 30.0 {
                    -50.0
                } else if f < 100.0 {
                    -50.0 + 50.0 * (f.log10() - 30.0_f64.log10()) / (100.0_f64.log10() - 30.0_f64.log10())
                } else {
                    0.0
                }
            })
            .collect();
        let res = detect_noise_floor(&freq, &mag);
        assert!(res.low.is_some(), "expected noise_floor_low finding");
        let f = res.low.unwrap();
        let (lo, hi) = f.freq_range.unwrap();
        assert!(lo < 10.0, "low edge {lo}");
        assert!(hi >= 20.0 && hi <= 35.0, "high edge {hi}");
        assert_eq!(f.id, "noise_floor_low");
    }

    #[test]
    fn lf_rolloff_is_detected() {
        // Steep rolloff below 50 Hz: -30 dB/oct, no noise floor (random walk
        // doesn't trigger noise-floor detector).
        let freq = log_grid(1024, 5.0, 20000.0);
        let mag: Vec<f64> = freq
            .iter()
            .enumerate()
            .map(|(i, &f)| {
                let osc = (i as f64 * 0.05).sin() * 0.5; // tiny ripple to avoid flat
                if f < 50.0 {
                    -30.0 * (50.0_f64.log2() - f.log2()) + osc
                } else {
                    osc
                }
            })
            .collect();
        let m = meas(freq.clone(), mag.clone());
        let lf = detect_lf_rolloff(&m, None);
        assert!(lf.is_some(), "expected lf_rolloff finding");
        let f = lf.unwrap();
        assert_eq!(f.id, "lf_rolloff_window");
    }

    #[test]
    fn hf_cliff_is_detected() {
        // Cliff above 18 kHz: -40 dB/oct.
        let freq = log_grid(2048, 20.0, 24000.0);
        let mag: Vec<f64> = freq
            .iter()
            .enumerate()
            .map(|(i, &f)| {
                let osc = (i as f64 * 0.03).sin() * 0.5;
                if f < 18000.0 {
                    osc
                } else {
                    -40.0 * (f.log2() - 18000.0_f64.log2()) + osc
                }
            })
            .collect();
        let cliff = detect_hf_cliff(&freq, &mag, 48000.0);
        assert!(cliff.is_some(), "expected hf_cliff finding");
        let f = cliff.unwrap();
        assert_eq!(f.id, "hf_cliff");
        let (lo, _hi) = f.freq_range.unwrap();
        assert!(lo >= 16000.0 && lo <= 19000.0, "cliff edge {lo}");
    }
}

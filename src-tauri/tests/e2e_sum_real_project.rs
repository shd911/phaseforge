// b140.2.1.2 — diagnostic only. NOT for CI; runs via:
//   cargo test --test e2e_sum_real_project -- --ignored --nocapture
//
// Compares the Legacy-style (renderSumMode) and New-style (evaluateSum)
// aggregation paths on the real 5wayNew project. Reports point-by-point
// diff per Σ curve and per-band so the visual mismatch on the SUM tab
// can be localised. No production code is touched, no asserts fire —
// the test prints findings and returns Ok.

use std::f64::consts::PI;
use std::path::Path;

use phaseforge_lib::dsp::{interp_1d, interpolate_log_grid};
use phaseforge_lib::io::Measurement;
use phaseforge_lib::peq::{apply_peq_complex, PeqBand};
use phaseforge_lib::project::{BandData, ProjectFile};
use phaseforge_lib::target::{self, TargetCurve};

#[test]
#[ignore]
fn diff_legacy_vs_new_5wayNew() {
    let project_dir = "/Users/olegryzhikov/phaseforge/test-fixtures/5wayNew";
    let pfproj_path = format!("{}/5wayNew.pfproj", project_dir);

    if !Path::new(&pfproj_path).exists() {
        eprintln!(
            "SKIP: 5wayNew fixture не найден ({}). Скопируй проект руками — тест в #[ignore].",
            pfproj_path
        );
        return;
    }

    let raw = std::fs::read_to_string(&pfproj_path).expect("read pfproj");
    let project: ProjectFile = serde_json::from_str(&raw).expect("parse pfproj");
    eprintln!(
        "\nLoaded project '{}': {} bands.",
        project.project_name.as_deref().unwrap_or("<unnamed>"),
        project.bands.len()
    );

    // -----------------------------------------------------------------
    // Step 1. Load measurements and prepare per-band data.
    // -----------------------------------------------------------------
    let mut prepared: Vec<PreparedBand> = Vec::with_capacity(project.bands.len());
    for band in &project.bands {
        prepared.push(prepare_band(project_dir, band));
    }
    eprintln!(
        "Loaded measurements: {}/{}.",
        prepared.iter().filter(|b| b.measurement.is_some()).count(),
        prepared.len()
    );

    // -----------------------------------------------------------------
    // Step 2. Common grid.
    //   Legacy: union extended to at least 20 Hz – 20 kHz, n = max bands' n.
    //   New:    union as-is, n = max bands' n.
    // -----------------------------------------------------------------
    let (mut f_min, mut f_max, mut n_max) = (f64::INFINITY, f64::NEG_INFINITY, 0usize);
    for b in &prepared {
        if let Some(m) = &b.measurement {
            if m.freq.first().copied().unwrap_or(f64::INFINITY) < f_min {
                f_min = m.freq[0];
            }
            if m.freq.last().copied().unwrap_or(f64::NEG_INFINITY) > f_max {
                f_max = *m.freq.last().unwrap();
            }
            if m.freq.len() > n_max {
                n_max = m.freq.len();
            }
        }
    }
    if !f_min.is_finite() || !f_max.is_finite() || n_max < 2 {
        eprintln!("ABORT: no measurements to aggregate.");
        return;
    }
    let new_f_min = f_min;
    let new_f_max = f_max;
    let legacy_f_min = f_min.min(20.0);
    let legacy_f_max = f_max.max(20000.0);
    let n_pts = n_max;
    let new_grid = log_grid(new_f_min, new_f_max, n_pts);
    let legacy_grid = log_grid(legacy_f_min, legacy_f_max, n_pts);

    eprintln!(
        "Grid: legacy {}..{:.0} Hz × {} pts | new {:.1}..{:.0} Hz × {} pts",
        legacy_f_min as i64, legacy_f_max, n_pts, new_f_min, new_f_max, n_pts
    );

    // -----------------------------------------------------------------
    // Step 3. Resample each band's measurement onto both grids.
    // -----------------------------------------------------------------
    for b in &mut prepared {
        if let Some(m) = &b.measurement {
            let (_, leg_mag, leg_phase) = interpolate_log_grid(
                &m.freq,
                &m.magnitude,
                m.phase.as_deref(),
                n_pts,
                legacy_f_min,
                legacy_f_max,
            );
            let (_, new_mag, new_phase) = interpolate_log_grid(
                &m.freq,
                &m.magnitude,
                m.phase.as_deref(),
                n_pts,
                new_f_min,
                new_f_max,
            );
            b.legacy_meas_mag = leg_mag;
            b.legacy_meas_phase = leg_phase;
            b.new_meas_mag = new_mag;
            b.new_meas_phase = new_phase;
        }
    }

    // -----------------------------------------------------------------
    // Step 4. Legacy global ref level vs each band's autoRef.
    //   Legacy: globalRef = max passband-avg (200..2000) across resampled meas.
    //   New (band-evaluator): each band uses its own autoRef on band's grid.
    // -----------------------------------------------------------------
    let global_ref = {
        let mut best = f64::NEG_INFINITY;
        for b in &prepared {
            if b.legacy_meas_mag.is_empty() {
                continue;
            }
            let avg = passband_avg(&legacy_grid, &b.legacy_meas_mag, 200.0, 2000.0);
            if let Some(v) = avg {
                if v > best {
                    best = v;
                }
            }
        }
        if best.is_finite() {
            best
        } else {
            0.0
        }
    };
    eprintln!("Legacy globalRef: {:.2} dB (max passband avg 200–2000 across bands).", global_ref);

    eprintln!("\nPer-band breakdown:");
    eprintln!(
        "  {:<32} {:>5} {:>9} {:>10} {:>10} {:>10} {:>10}",
        "name", "phase", "inverted", "delay (ms)", "passAvg", "autoRef", "Δref"
    );
    for b in &prepared {
        let auto = if let Some(m) = &b.measurement {
            band_auto_ref(m, &b.target).unwrap_or(0.0)
        } else {
            0.0
        };
        let pass_avg = passband_avg(&legacy_grid, &b.legacy_meas_mag, 200.0, 2000.0).unwrap_or(0.0);
        let has_phase = b
            .measurement
            .as_ref()
            .and_then(|m| m.phase.as_ref())
            .is_some();
        eprintln!(
            "  {:<32} {:>5} {:>9} {:>10.3} {:>10.2} {:>10.2} {:>+10.2}",
            shorten(&b.name, 32),
            if has_phase { "yes" } else { "no" },
            if b.inverted { "yes" } else { "no" },
            b.alignment_delay * 1000.0,
            pass_avg,
            auto,
            global_ref - auto
        );
    }

    // -----------------------------------------------------------------
    // Step 5. Σ Target — Legacy vs New.
    //   Legacy: target_curve.reference_level += globalRef before evaluate.
    //   New:    target uses band's autoRef (passband avg of its OWN meas).
    //   Both: coherent sum with polarity + alignment_delay.
    // -----------------------------------------------------------------
    let mut leg_targets: Vec<Option<SumBand>> = Vec::new();
    let mut new_targets: Vec<Option<SumBand>> = Vec::new();
    for b in &prepared {
        if !b.target_enabled {
            leg_targets.push(None);
            new_targets.push(None);
            continue;
        }
        // Legacy target: ref += globalRef, evaluate on legacy grid.
        let mut leg_curve = b.target.clone();
        leg_curve.reference_level_db += global_ref;
        let leg_resp = target::evaluate(&leg_curve, &legacy_grid);
        let leg_with_peq = add_peq(&legacy_grid, &leg_resp.magnitude, &leg_resp.phase, &b.peq_bands);
        leg_targets.push(Some(SumBand {
            mag: leg_with_peq.0,
            phase: leg_with_peq.1,
            sign: if b.inverted { -1.0 } else { 1.0 },
            delay: b.alignment_delay,
        }));

        // New target: ref += band's autoRef, evaluate on new grid.
        if let Some(m) = &b.measurement {
            let auto = band_auto_ref(m, &b.target).unwrap_or(0.0);
            let mut new_curve = b.target.clone();
            new_curve.reference_level_db += auto;
            let new_resp = target::evaluate(&new_curve, &new_grid);
            let new_with_peq = add_peq(&new_grid, &new_resp.magnitude, &new_resp.phase, &b.peq_bands);
            new_targets.push(Some(SumBand {
                mag: new_with_peq.0,
                phase: new_with_peq.1,
                sign: if b.inverted { -1.0 } else { 1.0 },
                delay: b.alignment_delay,
            }));
        } else {
            new_targets.push(None);
        }
    }

    let leg_target_sum = coherent_sum(&legacy_grid, &leg_targets);
    let new_target_sum = coherent_sum(&new_grid, &new_targets);
    report_diff_log(
        "Σ Target",
        &legacy_grid,
        leg_target_sum.as_ref().map(|s| &s.mag).map(|v| &v[..]),
        leg_target_sum.as_ref().map(|s| &s.phase).map(|v| &v[..]),
        &new_grid,
        new_target_sum.as_ref().map(|s| &s.mag).map(|v| &v[..]),
        new_target_sum.as_ref().map(|s| &s.phase).map(|v| &v[..]),
    );

    // -----------------------------------------------------------------
    // Step 6. Σ Measurement — both pipelines should match at this stage
    // (no normalisation). Diff isolates resampling + grid effects only.
    // -----------------------------------------------------------------
    let mut leg_meas: Vec<Option<SumBand>> = Vec::new();
    let mut new_meas: Vec<Option<SumBand>> = Vec::new();
    for b in &prepared {
        if let Some(m) = &b.measurement {
            let has_phase = m.phase.is_some();
            if has_phase {
                leg_meas.push(Some(SumBand {
                    mag: b.legacy_meas_mag.clone(),
                    phase: b.legacy_meas_phase.clone().unwrap_or_else(|| vec![0.0; n_pts]),
                    sign: if b.inverted { -1.0 } else { 1.0 },
                    delay: b.alignment_delay,
                }));
                new_meas.push(Some(SumBand {
                    mag: b.new_meas_mag.clone(),
                    phase: b.new_meas_phase.clone().unwrap_or_else(|| vec![0.0; n_pts]),
                    sign: if b.inverted { -1.0 } else { 1.0 },
                    delay: b.alignment_delay,
                }));
            } else {
                leg_meas.push(None);
                new_meas.push(None);
            }
        } else {
            leg_meas.push(None);
            new_meas.push(None);
        }
    }
    let leg_meas_sum = coherent_sum(&legacy_grid, &leg_meas);
    let new_meas_sum = coherent_sum(&new_grid, &new_meas);
    report_diff_log(
        "Σ Measurement (no normalisation)",
        &legacy_grid,
        leg_meas_sum.as_ref().map(|s| &s.mag).map(|v| &v[..]),
        leg_meas_sum.as_ref().map(|s| &s.phase).map(|v| &v[..]),
        &new_grid,
        new_meas_sum.as_ref().map(|s| &s.mag).map(|v| &v[..]),
        new_meas_sum.as_ref().map(|s| &s.phase).map(|v| &v[..]),
    );

    // -----------------------------------------------------------------
    // Step 7. b140.2.1.4 diagnostic: per-band freq ranges + extrapolation
    //         behaviour of interpolate_log on out-of-range queries.
    // -----------------------------------------------------------------
    eprintln!("\n--- Per-band measurement freq ranges ---");
    for (i, b) in prepared.iter().enumerate() {
        if let Some(m) = &b.measurement {
            let f_lo = m.freq.first().copied().unwrap_or(0.0);
            let f_hi = m.freq.last().copied().unwrap_or(0.0);
            let m_lo = m.magnitude.first().copied().unwrap_or(f64::NAN);
            let m_hi = m.magnitude.last().copied().unwrap_or(f64::NAN);
            eprintln!(
                "  Band {} ({:<32}): freq [{:>7.1}, {:>7.1}] Hz, {:>4} pts | mag@lo {:+6.1} dB | mag@hi {:+6.1} dB",
                i, shorten(&b.name, 32), f_lo, f_hi, m.freq.len(), m_lo, m_hi
            );
        } else {
            eprintln!("  Band {} ({}): no measurement", i, b.name);
        }
    }

    eprintln!(
        "\nNew (evaluateSum-style) common grid: [{:.2}, {:.1}] Hz, {} pts",
        new_grid[0], new_grid[new_grid.len() - 1], new_grid.len()
    );

    eprintln!("\n--- Extrapolation behaviour: interpolate_log on out-of-range freqs ---");
    eprintln!("(`in-range: true`  → query inside band's [f_lo, f_hi])");
    eprintln!("(`in-range: false` → boundary-clamped magnitude — phantom contribution to Σ)");
    let probes = [5.0_f64, 30.0, 100.0, 300.0, 500.0, 1000.0, 5000.0, 10000.0, 22000.0];
    for (i, b) in prepared.iter().enumerate() {
        let Some(m) = &b.measurement else { continue };
        let band_lo = m.freq[0];
        let band_hi = m.freq[m.freq.len() - 1];
        eprintln!(
            "\n  Band {} ({}) — native [{:.1}, {:.1}] Hz",
            i, shorten(&b.name, 32), band_lo, band_hi
        );
        let probe_freqs: Vec<f64> = probes.to_vec();
        let interp = interp_1d(&m.freq, &m.magnitude, &probe_freqs);
        for (j, &f) in probe_freqs.iter().enumerate() {
            let inside = f >= band_lo && f <= band_hi;
            eprintln!(
                "    {:>7.1} Hz: {:+6.2} dB    (in-range: {})",
                f, interp[j], inside
            );
        }
    }

    eprintln!("\n=== Done. No production code changed. ===\n");
}

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

struct PreparedBand {
    name: String,
    measurement: Option<Measurement>,
    target: TargetCurve,
    target_enabled: bool,
    inverted: bool,
    alignment_delay: f64,
    peq_bands: Vec<PeqBand>,

    legacy_meas_mag: Vec<f64>,
    legacy_meas_phase: Option<Vec<f64>>,
    new_meas_mag: Vec<f64>,
    new_meas_phase: Option<Vec<f64>>,
}

struct SumBand {
    mag: Vec<f64>,
    phase: Vec<f64>,
    sign: f64,
    delay: f64,
}

struct SumOut {
    mag: Vec<f64>,
    phase: Vec<f64>,
}

fn prepare_band(project_dir: &str, band: &BandData) -> PreparedBand {
    let measurement = band.measurement_file.as_ref().and_then(|file| {
        let p = format!("{}/{}", project_dir, file);
        match phaseforge_lib::io::import_measurement(Path::new(&p)) {
            Ok(m) => Some(m),
            Err(e) => {
                eprintln!("  ⚠ load {}: {:?}", file, e);
                None
            }
        }
    });
    PreparedBand {
        name: band.name.clone(),
        measurement,
        target: band.target.clone(),
        target_enabled: band.target_enabled,
        inverted: band.inverted,
        alignment_delay: band.alignment_delay.unwrap_or(0.0),
        peq_bands: band.peq_bands.clone(),
        legacy_meas_mag: Vec::new(),
        legacy_meas_phase: None,
        new_meas_mag: Vec::new(),
        new_meas_phase: None,
    }
}

fn log_grid(f_min: f64, f_max: f64, n: usize) -> Vec<f64> {
    let lo = f_min.ln();
    let hi = f_max.ln();
    (0..n)
        .map(|i| (lo + (hi - lo) * i as f64 / (n - 1) as f64).exp())
        .collect()
}

fn passband_avg(freq: &[f64], mag: &[f64], lo: f64, hi: f64) -> Option<f64> {
    if mag.is_empty() {
        return None;
    }
    let mut s = 0.0_f64;
    let mut c = 0usize;
    for (i, &f) in freq.iter().enumerate() {
        if f >= lo && f <= hi {
            s += mag[i];
            c += 1;
        }
    }
    if c > 0 {
        Some(s / c as f64)
    } else {
        None
    }
}

/// New band-evaluator-style autoRef: passband average of measurement on its
/// own grid, with the passband adapted to HP/LP corners.
fn band_auto_ref(m: &Measurement, target: &TargetCurve) -> Option<f64> {
    let hp = target.high_pass.as_ref().map(|f| f.freq_hz).unwrap_or(20.0);
    let lp = target.low_pass.as_ref().map(|f| f.freq_hz).unwrap_or(20000.0);
    let pb_lo = (hp * 1.5).max(20.0);
    let pb_hi = (lp * 0.7).min(20000.0);
    let (lo, hi) = if pb_lo < pb_hi { (pb_lo, pb_hi) } else { (200.0, 2000.0) };
    passband_avg(&m.freq, &m.magnitude, lo, hi)
}

fn add_peq(freq: &[f64], mag: &[f64], phase: &[f64], bands: &[PeqBand]) -> (Vec<f64>, Vec<f64>) {
    let enabled: Vec<PeqBand> = bands.iter().filter(|p| p.enabled).cloned().collect();
    if enabled.is_empty() {
        return (mag.to_vec(), phase.to_vec());
    }
    let (peq_mag, peq_phase) = apply_peq_complex(freq, &enabled, 48000.0);
    let out_mag: Vec<f64> = mag.iter().zip(peq_mag.iter()).map(|(m, p)| m + p).collect();
    let out_phase: Vec<f64> = phase.iter().zip(peq_phase.iter()).map(|(m, p)| m + p).collect();
    (out_mag, out_phase)
}

fn coherent_sum(freq: &[f64], bands: &[Option<SumBand>]) -> Option<SumOut> {
    let n = freq.len();
    let mut re = vec![0.0_f64; n];
    let mut im = vec![0.0_f64; n];
    let mut any = false;
    for entry in bands {
        let Some(b) = entry else { continue };
        any = true;
        for j in 0..n {
            let amp = 10f64.powf((b.mag[j].max(-200.0)) / 20.0) * b.sign;
            let ph_rad = (b.phase[j] + 360.0 * freq[j] * b.delay) * PI / 180.0;
            re[j] += amp * ph_rad.cos();
            im[j] += amp * ph_rad.sin();
        }
    }
    if !any {
        return None;
    }
    let mut mag = vec![0.0_f64; n];
    let mut phase = vec![0.0_f64; n];
    for j in 0..n {
        let amplitude = (re[j] * re[j] + im[j] * im[j]).sqrt();
        mag[j] = if amplitude > 0.0 { 20.0 * amplitude.log10() } else { -200.0 };
        phase[j] = im[j].atan2(re[j]) * 180.0 / PI;
    }
    Some(SumOut { mag, phase })
}

/// Compares two curves on (potentially) different grids by interpolating the
/// new curve onto the legacy grid for a fair point-by-point comparison.
fn report_diff_log(
    label: &str,
    legacy_freq: &[f64],
    legacy_mag: Option<&[f64]>,
    legacy_phase: Option<&[f64]>,
    new_freq: &[f64],
    new_mag: Option<&[f64]>,
    new_phase: Option<&[f64]>,
) {
    eprintln!("\n--- {} ---", label);
    let (Some(lm), Some(nm)) = (legacy_mag, new_mag) else {
        eprintln!("  ABSENT: legacy={} new={}",
            legacy_mag.is_some(), new_mag.is_some());
        return;
    };

    // Re-sample new onto legacy grid for comparison.
    let nm_on_leg = interp_1d(new_freq, nm, legacy_freq);
    let mut max_diff = 0.0_f64;
    let mut max_idx = 0usize;
    for j in 0..legacy_freq.len() {
        let d = (lm[j] - nm_on_leg[j]).abs();
        if d > max_diff {
            max_diff = d;
            max_idx = j;
        }
    }
    let mean_diff = {
        let mut s = 0.0_f64;
        let mut c = 0usize;
        for j in 0..legacy_freq.len() {
            if legacy_freq[j] >= 50.0 && legacy_freq[j] <= 15000.0 {
                s += (lm[j] - nm_on_leg[j]).abs();
                c += 1;
            }
        }
        if c > 0 { s / c as f64 } else { 0.0 }
    };
    eprintln!(
        "  mag: max diff {:.3} dB at {:.1} Hz   |   mean abs diff {:.3} dB (50–15k)",
        max_diff, legacy_freq[max_idx], mean_diff
    );

    if let (Some(lp), Some(np)) = (legacy_phase, new_phase) {
        let np_on_leg = interp_1d(new_freq, np, legacy_freq);
        let mut max_pdiff = 0.0_f64;
        let mut max_pidx = 0usize;
        for j in 0..legacy_freq.len() {
            let dp = wrap_deg(lp[j] - np_on_leg[j]).abs();
            if dp > max_pdiff {
                max_pdiff = dp;
                max_pidx = j;
            }
        }
        eprintln!(
            "  phase: max wrapped diff {:.2}° at {:.1} Hz",
            max_pdiff, legacy_freq[max_pidx]
        );
    }

    // Show offset at a few signpost frequencies.
    eprintln!("  signpost (legacy − new on legacy grid):");
    for &f in &[30.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0] {
        let idx = nearest_idx(legacy_freq, f);
        let d = lm[idx] - nm_on_leg[idx];
        eprintln!("    {:>6.0} Hz: {:+.3} dB", legacy_freq[idx], d);
    }
}

fn nearest_idx(freq: &[f64], target: f64) -> usize {
    let mut best = 0;
    let mut best_d = f64::INFINITY;
    for (i, &f) in freq.iter().enumerate() {
        let d = (f - target).abs();
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

fn wrap_deg(p: f64) -> f64 {
    ((p + 180.0).rem_euclid(360.0)) - 180.0
}

fn shorten(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n - 1])
    }
}

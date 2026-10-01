//! b141.53 (audit 2026-10-01 M2): «Макс. подъём» caps the TOTAL target + PEQ
//! once. It used to cap the PEQ alone at max_boost, so a +30 dB PEQ under an
//! LR4 HP came out 5.97 dB below the plot (−25.72 vs −19.75 dB at 1 kHz)
//! although the total never reached the limit.

use phaseforge_lib::fir::{generate_model_fir, FirConfig, PhaseMode, WindowType};
use phaseforge_lib::peq::{apply_peq, PeqBand, PeqFilterType};
use phaseforge_lib::target::{self, FilterConfig, FilterType, TargetCurve};

const SR: f64 = 48_000.0;

fn run(max_boost: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let freq: Vec<f64> = (0..512).map(|i| 5.0 * (SR / 2.0 * 0.95 / 5.0f64).powf(i as f64 / 511.0)).collect();
    let t = TargetCurve {
        reference_level_db: 0.0, tilt_db_per_octave: 0.0, tilt_ref_freq: 1000.0,
        high_pass: Some(FilterConfig { filter_type: FilterType::LinkwitzRiley, order: 4, freq_hz: 2000.0,
            shape: None, linear_phase: true, q: None, subsonic_protect: None }),
        low_pass: None, low_shelf: None, high_shelf: None,
    };
    let r = target::evaluate(&t, &freq);
    let peq = [PeqBand { freq_hz: 1000.0, gain_db: 30.0, q: 1.0, enabled: true, filter_type: PeqFilterType::Peaking }];
    let pm = apply_peq(&freq, &peq, SR);
    let cfg = FirConfig {
        taps: 65_536, sample_rate: SR, max_boost_db: max_boost, noise_floor_db: -150.0,
        window: WindowType::Blackman, phase_mode: PhaseMode::Composite, iterations: 0,
        freq_weighting: false, narrowband_limit: false, nb_smoothing_oct: 0.333,
        nb_max_excess_db: 6.0, linear_phase_main: true, subsonic_cutoff_hz: None,
    };
    let out = generate_model_fir(&freq, &r.magnitude, &pm, &vec![0.0; 512], &cfg).unwrap();
    let model: Vec<f64> = r.magnitude.iter().zip(&pm).map(|(a, b)| a + b).collect();
    (freq, model, out.realized_mag)
}

fn at(f: &[f64], v: &[f64], x: f64) -> f64 {
    let i = f.iter().position(|&q| q >= x).unwrap();
    v[i - 1] + (v[i] - v[i - 1]) * (x - f[i - 1]) / (f[i] - f[i - 1])
}

#[test]
fn peq_below_the_limit_is_realised_in_full() {
    let (f, model, real) = run(24.0);
    // Compare shapes (realised is normalised to a 0 dB peak).
    let dm = at(&f, &model, 1000.0) - at(&f, &model, 8000.0);
    let dr = at(&f, &real, 1000.0) - at(&f, &real, 8000.0);
    println!("model Δ(1k−8k) {dm:.2} dB, realised {dr:.2} dB");
    assert!((dm - dr).abs() < 0.2, "realised {dr:.2} vs model {dm:.2} dB");
}

#[test]
fn total_above_the_limit_is_capped() {
    // Limit 3 dB: the PEQ peak (+30 on a HP at −20 dB → about +6 dB total near 1.5 kHz) is cut to 3.
    let (f, model, real) = run(3.0);
    let peak_model = f.iter().zip(&model).filter(|(&q, _)| q > 200.0 && q < 20_000.0)
        .map(|(_, &m)| m).fold(f64::MIN, f64::max);
    assert!(peak_model > 3.0, "fixture must exceed the limit, peak {peak_model:.2}");
    // Realised is normalised to its (capped, 3 dB) peak: 8 kHz must sit at
    // model(8 kHz) − 3, not model(8 kHz) − peak_model.
    let dr = at(&f, &real, 8000.0);
    let want = at(&f, &model, 8000.0) - 3.0;
    println!("model peak {peak_model:.2} dB, realised 8 kHz {dr:.2} dB re peak, want {want:.2}");
    assert!((dr - want).abs() < 0.3, "8 kHz {dr:.2} dB, want {want:.2} dB under a 3 dB cap");
}

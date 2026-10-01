//! b141.52 (audit 2026-10-01 H4): the phase the plot reconstructs for a
//! Gaussian min-phase band (compute_minimum_phase on the FIR grid with the
//! FIR's absolute floor — what `reconstructTargetPhase` now asks for) must
//! match the phase the cepstral FIR realises. Before: 65° apart at 5 kHz on
//! a Gaussian LP 2 kHz, 101° at 20 Hz on a Gaussian HP 100 + subsonic.

use phaseforge_lib::dsp::minimum_phase_on_log_grid;
use phaseforge_lib::fir::{generate_model_fir, FirConfig, PhaseMode, WindowType};
use phaseforge_lib::target::{self, FilterConfig, FilterType, TargetCurve};

const SR: f64 = 48_000.0;

fn grid(n: usize) -> Vec<f64> {
    let (a, b) = (5.0_f64, SR / 2.0 * 0.95);
    (0..n).map(|i| a * (b / a).powf(i as f64 / (n - 1) as f64)).collect()
}

fn gauss(fc: f64, subsonic: bool) -> FilterConfig {
    FilterConfig {
        filter_type: FilterType::Gaussian, order: 4, freq_hz: fc, shape: Some(1.0),
        linear_phase: false, q: None, subsonic_protect: Some(subsonic),
    }
}

fn curve(hp: Option<FilterConfig>, lp: Option<FilterConfig>) -> TargetCurve {
    TargetCurve { reference_level_db: 0.0, tilt_db_per_octave: 0.0, tilt_ref_freq: 1000.0,
        high_pass: hp, low_pass: lp, low_shelf: None, high_shelf: None }
}

fn check(t: &TargetCurve, subsonic: Option<f64>, probes: &[f64], tol: f64) {
    // FIR as the frontend sends it: 512-point FIR grid, Composite min-phase.
    let fg = grid(512);
    let r = target::evaluate(t, &fg);
    let cfg = FirConfig {
        taps: 65_536, sample_rate: SR, max_boost_db: 24.0, noise_floor_db: -150.0,
        window: WindowType::Blackman, phase_mode: PhaseMode::Composite, iterations: 0,
        freq_weighting: false, narrowband_limit: false, nb_smoothing_oct: 0.333,
        nb_max_excess_db: 6.0, linear_phase_main: false, subsonic_cutoff_hz: subsonic,
    };
    let fir = generate_model_fir(&fg, &r.magnitude, &[], &r.phase, &cfg).unwrap();

    // Plot: Hilbert on the 1024-point canonical grid, absolute −150 dB floor.
    let pg = grid(1024);
    let pm = target::evaluate(t, &pg).magnitude;
    let plot = minimum_phase_on_log_grid(&pg, &pm, Some(SR), Some(-150.0)).unwrap();
    let at = |f: &[f64], v: &[f64], x: f64| {
        let i = f.iter().position(|&q| q >= x).unwrap();
        v[i - 1] + (v[i] - v[i - 1]) * (x - f[i - 1]) / (f[i] - f[i - 1])
    };
    for &f in probes {
        let d = at(&pg, &plot, f) - at(&fg, &fir.realized_phase, f);
        let d = ((d + 180.0) % 360.0 + 360.0) % 360.0 - 180.0;
        println!("f={f:>7}: plot − FIR = {d:7.2}°");
        assert!(d.abs() < tol, "plot vs FIR {d:.2}° at {f} Hz");
    }
}

#[test]
fn gaussian_lp_plot_phase_matches_fir() {
    check(&curve(None, Some(gauss(2000.0, false))), None, &[500.0, 1000.0, 2000.0, 5000.0], 2.0);
}

#[test]
fn gaussian_hp_with_subsonic_plot_phase_matches_fir() {
    check(&curve(Some(gauss(100.0, true)), None), Some(100.0 / 8.0), &[20.0, 50.0, 100.0, 400.0], 2.0);
}

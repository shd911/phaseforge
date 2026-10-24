// b141.107: Linkwitz transform as a PEQ type — a theoretical correction of a
// sealed box's assumed 12 dB/oct roll-off. The bottom of a real sub
// measurement («wall» 10x4) is the REW sweep edge (start 20.29 Hz: 35–97
// dB/oct, phase turning the wrong way), so the box's f0/Q0 cannot be fitted
// from it; the transform is entered on an assumed model instead.
//
// The transform must turn the model box into the BU2 target exactly, on both
// FIR routes, and ring as that target — not as a resonant boost at the edge
// (24 Hz +4.5 dB Q 1.9: 215 ms to −60 dB).
use phaseforge_lib::fir::band::{generate_band_fir, BandFirSettings};
use phaseforge_lib::fir::WindowType;
use phaseforge_lib::peq::q_envelope::ring_ms;
use phaseforge_lib::peq::{peq_band_response, PeqBand, PeqFilterType};
use phaseforge_lib::target::{FilterConfig, FilterType, TargetCurve};

fn lt(f0: f64, q0: f64, gain: f64) -> PeqBand {
    PeqBand { freq_hz: f0, gain_db: gain, q: q0, enabled: true, filter_type: PeqFilterType::LinkwitzTransform }
}

/// |HP2(f; fc, q)| in dB.
fn hp2_db(f: f64, fc: f64, q: f64) -> f64 {
    let x = f / fc;
    let (re, im) = (1.0 - x * x, x / q);
    20.0 * (x * x / (re * re + im * im).sqrt()).log10()
}

#[test]
fn transform_turns_the_box_into_the_bu2_target() {
    let (f0, q0, gain) = (25.3, 0.707, 6.0);
    let fp = f0 / 10f64.powf(gain / 40.0);
    let freq: Vec<f64> = (0..400).map(|i| 5.0 * 40f64.powf(i as f64 / 399.0)).collect();
    for sr in [44_100.0, 48_000.0, 176_400.0] {
        let t = peq_band_response(&freq, &lt(f0, q0, gain), sr);
        let worst = freq.iter().zip(&t).map(|(&f, &g)| {
            (hp2_db(f, f0, q0) + g - hp2_db(f, fp, std::f64::consts::FRAC_1_SQRT_2)).abs()
        }).fold(0.0, f64::max);
        println!("{sr} Hz: fp {fp:.2} Hz, box × LT vs BU2 worst {worst:.4} dB; DC-ish lift {:.2} dB", t[0]);
        assert!(worst < 0.02, "{sr}: box × transform deviates {worst:.3} dB from the target");
        assert!((t[0] - gain).abs() < 0.3, "{sr}: lift at 5 Hz {:.2} dB, expected ~{gain}", t[0]);
    }
}

#[test]
fn both_fir_routes_realise_the_transform() {
    let lp = |lin: bool| FilterConfig { filter_type: FilterType::LinkwitzRiley, order: 1, freq_hz: 130.0,
        shape: None, linear_phase: lin, q: None, subsonic_protect: None };
    let peq = [lt(25.3, 0.707, 6.0)];
    for lin in [false, true] {
        let target = TargetCurve { reference_level_db: 0.0, tilt_db_per_octave: 0.0, tilt_ref_freq: 1000.0,
            high_pass: None, low_pass: Some(lp(lin)), low_shelf: None, high_shelf: None };
        let s = BandFirSettings { taps: 131_072, sample_rate: 176_400.0, window: WindowType::Blackman,
            max_boost_db: 24.0, noise_floor_db: -150.0, iterations: 3, freq_weighting: true,
            narrowband_limit: true, nb_smoothing_oct: 0.333, nb_max_excess_db: 6.0 };
        let r = generate_band_fir(&target, &peq, &s, false).unwrap();
        assert!(r.fir.impulse.iter().all(|v| v.is_finite()), "non-finite impulse");
        let dev = r.freq.iter().zip(&r.dev_mag).filter(|(f, _)| **f >= 10.0 && **f <= 100.0)
            .map(|(_, d)| d.abs()).fold(0.0, f64::max);
        println!("{:?} route: |dev_mag| 10–100 Hz ≤ {dev:.3} dB, peak boost {:.2} dB", r.route, r.peak_boost_db);
        assert!(dev < 0.1, "{:?}: the FIR misses the transform by {dev:.2} dB", r.route);
        assert!((r.peak_boost_db - 6.0).abs() < 0.5, "peak boost {:.2} dB", r.peak_boost_db);
    }
}

#[test]
fn transform_rings_like_its_target_not_like_an_edge_boost() {
    let sr = 176_400.0;
    let t = ring_ms(25.3, 6.0, 0.707, &PeqFilterType::LinkwitzTransform, sr, 60.0);
    let edge = ring_ms(24.0, 4.5, 1.92, &PeqFilterType::Peaking, sr, 60.0);
    println!("LT 25.3 Hz Q0.71 +6 dB: {t:.0} ms; edge boost 24 Hz +4.5 Q1.92: {edge:.0} ms (to −60 dB)");
    assert!(t.is_finite() && t > 0.0);
    assert!(t < 0.6 * edge, "the transform rings {t:.0} ms vs the edge boost {edge:.0} ms");
}

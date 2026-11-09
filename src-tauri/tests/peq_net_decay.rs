// b141.139: the PEQ «мс» column is judged on the corrected response, not on
// the filter alone. A cut on a resonance cancels that resonance's ring (the
// filter's own ring is not heard), a boost into a dip made by a reflection
// adds a ring nothing cancels. Synthetic measurements, 96 kHz.
use num_complex::Complex64;
use phaseforge_lib::fir::band::peq_response_as_exported;
use phaseforge_lib::peq::net_decay::net_decay;
use phaseforge_lib::peq::{PeqBand, PeqFilterType};

const SR: f64 = 96_000.0;

fn pk(f: f64, g: f64, q: f64) -> PeqBand {
    PeqBand { freq_hz: f, gain_db: g, q, enabled: true, filter_type: PeqFilterType::Peaking }
}

fn log_grid(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n).map(|i| lo * (hi / lo).powf(i as f64 / (n - 1) as f64)).collect()
}

fn as_measurement(f: &[f64], h: impl Fn(f64) -> Complex64) -> (Vec<f64>, Vec<f64>) {
    f.iter().map(|&x| { let c = h(x); (20.0 * c.norm().log10(), c.arg().to_degrees()) }).unzip()
}

#[test]
fn a_cut_on_a_resonance_brings_the_decay_back_to_the_ear() {
    let f = log_grid(10.0, 20_000.0, 1100);
    let (mag, ph) = peq_response_as_exported(&f, &[pk(50.0, 6.0, 5.0)], SR);
    let r = net_decay(&f, &mag, &ph, &[pk(50.0, -6.0, 5.0)], SR).unwrap();
    let d = r[0].as_ref().unwrap();
    eprintln!("{d:?}");
    for i in 0..2 {
        assert!(d.without_ms[i] > 2.5 * d.ear_ms[i], "resonance rings past the ear: {d:?}");
        assert!((d.with_ms[i] / d.ear_ms[i] - 1.0).abs() < 0.15, "cut back to the ear: {d:?}");
    }
}

fn reflection(direct: f64, refl: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // A reflection 10 ms late: a dip right at 50 Hz.
    let f = log_grid(10.0, 20_000.0, 16_000);
    let (mag, ph) = as_measurement(&f, |x| {
        direct + refl * Complex64::from_polar(1.0, -std::f64::consts::TAU * x * 0.01)
    });
    (f, mag, ph)
}

#[test]
fn a_boost_into_a_weaker_reflections_dip_shortens_it() {
    // Direct louder than the reflection: the dip is minimum-phase, a boost on
    // it is its inverse and takes its ring away too.
    let (f, mag, ph) = reflection(1.0, 0.7);
    let d = net_decay(&f, &mag, &ph, &[pk(50.0, 8.0, 3.0)], SR).unwrap()[0].clone().unwrap();
    eprintln!("{d:?}");
    assert!(d.with_ms[1] < d.without_ms[1], "{d:?}");
}

#[test]
fn a_boost_into_an_excess_phase_dip_rings_longer() {
    // The reflection louder than the direct sound (or a boost off the dip's
    // centre): nothing in the response cancels what the boost adds.
    let (f, mag, ph) = reflection(0.7, 1.0);
    let d = net_decay(&f, &mag, &ph, &[pk(50.0, 8.0, 3.0)], SR).unwrap()[0].clone().unwrap();
    eprintln!("{d:?}");
    assert!(d.with_ms[1] > 1.3 * d.without_ms[1], "{d:?}");
    let (f, mag, ph) = reflection(1.0, 0.7);
    let d = net_decay(&f, &mag, &ph, &[pk(54.4, 8.0, 3.0)], SR).unwrap()[0].clone().unwrap();
    eprintln!("{d:?}");
    assert!(d.with_ms[1] > 1.3 * d.without_ms[1], "{d:?}");
}

#[test]
fn nothing_for_disabled_flat_or_outside_the_measurement() {
    let f = log_grid(30.0, 2_000.0, 600);
    let (mag, ph): (Vec<f64>, Vec<f64>) = (vec![0.0; f.len()], vec![0.0; f.len()]);
    let mut off = pk(100.0, 3.0, 2.0);
    off.enabled = false;
    let peq = [off, pk(100.0, 0.0, 2.0), pk(35.0, 3.0, 2.0), pk(1_800.0, 3.0, 2.0), pk(100.0, 3.0, 2.0)];
    let r = net_decay(&f, &mag, &ph, &peq, SR).unwrap();
    assert!(r[..4].iter().all(Option::is_none), "{r:?}");
    assert!(r[4].is_some());
    assert!(net_decay(&f, &mag[1..], &ph, &peq, SR).is_err());
}

#[test]
fn a_linkwitz_transform_is_judged_at_its_new_corner() {
    let f = log_grid(10.0, 2_000.0, 600);
    let (mag, ph): (Vec<f64>, Vec<f64>) = (vec![0.0; f.len()], vec![0.0; f.len()]);
    let lt = PeqBand { freq_hz: 60.0, gain_db: 12.0, q: 0.7, enabled: true, filter_type: PeqFilterType::LinkwitzTransform };
    let d = net_decay(&f, &mag, &ph, &[lt], SR).unwrap()[0].clone().unwrap();
    assert!((d.freq_hz - 60.0 / 10f64.powf(0.3)).abs() < 1e-9, "{d:?}");
}

/// b141.144: a measurement whose delay removal overshot (response starts
/// before t = 0 — a real tweeter, residual GD −5 ms) puts the energy peak at
/// the END of the analysis buffer. The tail must be read circularly: cut at
/// the buffer end it was ~1 ms (the ear filter adds ~3.5 ms of its own delay, so 6 ms early is past zero), the −30 dB decay came back as infinity, JSON
/// wrote null and the «мс» tooltip threw — the plot froze on band switches.
#[test]
fn a_measurement_starting_before_zero_decays_as_if_it_did_not() {
    let f: Vec<f64> = (0..50_000).map(|i| 500.0 + i as f64 * 0.02).collect();
    let mag = vec![0.0; f.len()];
    let run = |tau: f64| {
        let ph: Vec<f64> = f.iter().map(|&x| -360.0 * x * tau).collect();
        net_decay(&f, &mag, &ph, &[pk(1_000.0, -3.0, 2.0)], SR).unwrap()[0].clone().unwrap()
    };
    let (early, on_time) = (run(-0.006), run(0.0));
    let all = [early.ear_ms, early.without_ms, early.with_ms].concat();
    assert!(all.iter().all(|v| v.is_finite()), "{early:?}");
    for (a, b) in early.with_ms.iter().zip(&on_time.with_ms).chain(early.without_ms.iter().zip(&on_time.without_ms)) {
        assert!((a - b).abs() < 0.05, "a 6 ms early start changed the decay: {early:?} vs {on_time:?}");
    }
    assert!(!serde_json::to_string(&early).unwrap().contains("null"));
}

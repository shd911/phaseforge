// b140.0 — synthetic fixtures for the E2E export harness.
//
// Note on file location: Rust integration tests treat every `tests/*.rs` as a
// separate test binary. To share helpers between tests, the standard idiom is
// `tests/common/mod.rs` (sub-directory modules are NOT auto-treated as test
// binaries). The TZ called for `tests/fixtures.rs` — same content, this is
// just the Cargo-friendly placement.

#![allow(dead_code)]

use phaseforge_lib::io::{Measurement, MeasurementMetadata};
use phaseforge_lib::peq::{PeqBand, PeqFilterType};
use phaseforge_lib::target::{FilterConfig, FilterType, TargetCurve};

/// Plain flat measurement: 0 dB, 0° phase, log freq grid.
pub fn flat_measurement(f_min: f64, f_max: f64, n: usize) -> Measurement {
    let freq: Vec<f64> = (0..n)
        .map(|i| f_min * (f_max / f_min).powf(i as f64 / (n - 1) as f64))
        .collect();
    Measurement {
        name: "flat".into(),
        source_path: None,
        sample_rate: Some(48000.0),
        freq,
        magnitude: vec![0.0; n],
        phase: Some(vec![0.0; n]),
        metadata: MeasurementMetadata::default(),
    }
}

/// One acceptance-matrix entry: name + target + linear/min main + PEQ bands.
pub struct ExportConfig {
    pub name: &'static str,
    pub target: TargetCurve,
    pub linear_phase_main: bool,
    pub peq_bands: Vec<PeqBand>,
}

/// 8 canonical configurations — the Cartesian product
///   { linear_main, min_main } × { subsonic_off, subsonic_on } × { no_peq, with_peq }.
pub fn acceptance_configs() -> Vec<ExportConfig> {
    vec![
        ExportConfig {
            name: "linear_no_subsonic_no_peq",
            target: gaussian_hp(632.0, true, false),
            linear_phase_main: true,
            peq_bands: vec![],
        },
        ExportConfig {
            name: "linear_subsonic_no_peq",
            target: gaussian_hp(632.0, true, true),
            linear_phase_main: true,
            peq_bands: vec![],
        },
        ExportConfig {
            name: "min_no_subsonic_no_peq",
            target: gaussian_hp(632.0, false, false),
            linear_phase_main: false,
            peq_bands: vec![],
        },
        ExportConfig {
            name: "min_subsonic_no_peq",
            target: gaussian_hp(632.0, false, true),
            linear_phase_main: false,
            peq_bands: vec![],
        },
        ExportConfig {
            name: "linear_no_subsonic_with_peq",
            target: gaussian_hp(632.0, true, false),
            linear_phase_main: true,
            peq_bands: sample_peq_bands(),
        },
        ExportConfig {
            name: "linear_subsonic_with_peq",
            target: gaussian_hp(632.0, true, true),
            linear_phase_main: true,
            peq_bands: sample_peq_bands(),
        },
        ExportConfig {
            name: "min_no_subsonic_with_peq",
            target: gaussian_hp(632.0, false, false),
            linear_phase_main: false,
            peq_bands: sample_peq_bands(),
        },
        ExportConfig {
            name: "min_subsonic_with_peq",
            target: gaussian_hp(632.0, false, true),
            linear_phase_main: false,
            peq_bands: sample_peq_bands(),
        },
    ]
}

fn gaussian_hp(fc: f64, linear: bool, subsonic: bool) -> TargetCurve {
    TargetCurve {
        reference_level_db: 0.0,
        tilt_db_per_octave: 0.0,
        tilt_ref_freq: 1000.0,
        high_pass: Some(FilterConfig {
            filter_type: FilterType::Gaussian,
            order: 4,
            freq_hz: fc,
            shape: Some(1.0),
            linear_phase: linear,
            q: None,
            subsonic_protect: Some(subsonic),
        }),
        low_pass: None,
        low_shelf: None,
        high_shelf: None,
    }
}

/// Three narrow PEQ bands with known frequency / Q / gain. Each has |gain| ≥ 6 dB
/// and Q ≥ 3 so the min-phase rotation around the band centre is ≥ 30° —
/// large enough that the b140.0 PEQ-rotation assertion can detect a missing
/// PEQ phase contribution unambiguously.
pub fn sample_peq_bands() -> Vec<PeqBand> {
    vec![
        PeqBand {
            freq_hz: 200.0,
            gain_db: -6.0,
            q: 4.0,
            enabled: true,
            filter_type: PeqFilterType::Peaking,
        },
        PeqBand {
            freq_hz: 1000.0,
            gain_db: 6.0,
            q: 3.0,
            enabled: true,
            filter_type: PeqFilterType::Peaking,
        },
        PeqBand {
            freq_hz: 5000.0,
            gain_db: -6.0,
            q: 3.0,
            enabled: true,
            filter_type: PeqFilterType::Peaking,
        },
    ]
}

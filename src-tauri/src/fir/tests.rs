//! b140.13.3 — FIR-pipeline integration tests extracted from `fir/mod.rs`.
//!
//! Verbatim move: same `#[cfg(test)] mod tests` body, now its own file.
//! `super::*` still resolves to `crate::fir::*` so the windowing /
//! helpers / minimum_phase re-exports gated behind `#[cfg(test)]` in
//! mod.rs remain the path the tests use to find their dependencies.
//!
//! Behaviour unchanged — Rust suite stays at 184 in-lib + 13 integration
//! tests, golden_fir / pipeline_contract baselines hold.

use super::*;

    #[test]
    fn test_recommend_taps() {
        // For 20 Hz at 48000 Hz: 3 * 48000 / 20 = 7200 → next pow2 = 8192
        let taps = recommend_taps(20.0, 48000.0);
        assert_eq!(taps, 8192);

        // For 80 Hz at 48000 Hz: 3 * 48000 / 80 = 1800 → next pow2 = 2048 → clamp to 4096
        let taps = recommend_taps(80.0, 48000.0);
        assert_eq!(taps, 4096);
    }

    #[test]
    fn test_flat_correction_produces_dirac() {
        // 0 dB correction everywhere should produce a near-dirac impulse
        let n = 100;
        let freq: Vec<f64> = (0..n).map(|i| 20.0 + i as f64 * 200.0).collect();
        let mag: Vec<f64> = vec![80.0; n];
        let target: Vec<f64> = vec![80.0; n]; // same as measurement = 0 correction
        let peq: Vec<f64> = vec![0.0; n];

        let config = FirConfig {
            taps: 4096,
            sample_rate: 48000.0,
            max_boost_db: 18.0,
            noise_floor_db: -60.0,
            window: WindowType::Blackman,
            phase_mode: PhaseMode::MinimumPhase,
            iterations: 0, freq_weighting: false, narrowband_limit: false,
            nb_smoothing_oct: 0.333, nb_max_excess_db: 6.0,
            gaussian_min_phase_filters: vec![],
        linear_phase_main: false,
        subsonic_cutoff_hz: None,
        };

        let result = generate_fir(&freq, &mag, &target, &peq, &config, (20.0, 20000.0)).unwrap();
        assert_eq!(result.impulse.len(), 4096);
        assert_eq!(result.time_ms.len(), 4096);

        // Passband normalization should be near 0 dB (unity gain)
        assert!(result.norm_db.abs() < 3.0, "norm_db should be near 0 dB, got {}", result.norm_db);
    }

    #[test]
    fn test_boost_limiting() {
        // Large correction should be clamped
        let n = 100;
        let freq: Vec<f64> = (0..n).map(|i| 20.0 + i as f64 * 200.0).collect();
        let mag: Vec<f64> = vec![60.0; n];
        let target: Vec<f64> = vec![120.0; n]; // +60 dB correction!
        let peq: Vec<f64> = vec![0.0; n];

        let config = FirConfig {
            taps: 4096,
            sample_rate: 48000.0,
            max_boost_db: 18.0,
            noise_floor_db: -60.0,
            window: WindowType::Blackman,
            phase_mode: PhaseMode::MinimumPhase,
            iterations: 0, freq_weighting: false, narrowband_limit: false,
            nb_smoothing_oct: 0.333, nb_max_excess_db: 6.0,
            gaussian_min_phase_filters: vec![],
        linear_phase_main: false,
        subsonic_cutoff_hz: None,
        };

        let result = generate_fir(&freq, &mag, &target, &peq, &config, (20.0, 20000.0)).unwrap();
        // After passband normalization, norm_db reflects the passband level offset
        // With uniform +60dB correction clamped to +18dB, passband is 18dB → norm_db ≈ 18
        assert!(result.norm_db < 25.0, "norm_db should be limited, got {}", result.norm_db);
    }

    #[test]
    fn test_window_symmetry() {
        let n = 1024;
        let all_types = vec![
            WindowType::Rectangular,
            WindowType::Bartlett,
            WindowType::Hann,
            WindowType::Hamming,
            WindowType::Blackman,
            WindowType::ExactBlackman,
            WindowType::BlackmanHarris,
            WindowType::Nuttall3,
            WindowType::Nuttall4,
            WindowType::FlatTop,
            WindowType::Kaiser,
            WindowType::DolphChebyshev,
            WindowType::Gaussian,
            WindowType::Tukey,
            WindowType::Lanczos,
            WindowType::Poisson,
            WindowType::HannPoisson,
            WindowType::Bohman,
            WindowType::Cauchy,
            WindowType::Riesz,
        ];

        for wtype in &all_types {
            let w = generate_window(n, wtype);
            assert_eq!(w.len(), n, "{:?}: wrong length", wtype);

            // Check symmetry (some windows like DolphChebyshev have small numerical noise)
            for i in 0..n / 2 {
                let diff = (w[i] - w[n - 1 - i]).abs();
                assert!(diff < 1e-6, "{:?}: not symmetric at i={}: {} vs {}", wtype, i, w[i], w[n - 1 - i]);
            }

            // Center should be near 1.0 (except FlatTop which peaks > 1)
            let center = w[n / 2];
            assert!(center > 0.5, "{:?}: center too low = {}", wtype, center);
        }
    }

    #[test]
    fn test_kaiser_window() {
        let n = 512;
        let w = kaiser_window(n, 10.0);
        assert_eq!(w.len(), n);
        // Symmetric
        for i in 0..n / 2 {
            assert!((w[i] - w[n - 1 - i]).abs() < 1e-10);
        }
        // Center should be 1.0
        assert!((w[n / 2] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_sigmoid_blend() {
        // Well below center: should be ~0
        assert!(sigmoid_blend(10.0, 100.0, 0.5) < 0.01);
        // At center: should be ~0.5
        assert!((sigmoid_blend(100.0, 100.0, 0.5) - 0.5).abs() < 0.01);
        // Well above center: should be ~1
        assert!(sigmoid_blend(1000.0, 100.0, 0.5) > 0.99);
    }

    #[test]
    fn test_wav_export_f64() {
        let impulse = vec![0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5];
        let tmp = std::env::temp_dir().join("phaseforge_test_fir_f64.wav");
        export_wav_f64(&impulse, 48000.0, &tmp).unwrap();

        // Check file exists and has correct size (8 bytes per sample)
        let meta = std::fs::metadata(&tmp).unwrap();
        let expected_size = 44 + impulse.len() * 8; // header + data (64-bit)
        assert_eq!(meta.len(), expected_size as u64);

        // Read back and verify header
        let data = std::fs::read(&tmp).unwrap();
        assert_eq!(&data[0..4], b"RIFF");
        assert_eq!(&data[8..12], b"WAVE");
        assert_eq!(&data[12..16], b"fmt ");
        // Format = 3 (IEEE float)
        assert_eq!(u16::from_le_bytes([data[20], data[21]]), 3);
        // Channels = 1
        assert_eq!(u16::from_le_bytes([data[22], data[23]]), 1);
        // Sample rate = 48000
        assert_eq!(u32::from_le_bytes([data[24], data[25], data[26], data[27]]), 48000);
        // Bits per sample = 64
        assert_eq!(u16::from_le_bytes([data[34], data[35]]), 64);

        // Verify first sample is 0.0 f64
        let sample0 = f64::from_le_bytes(data[44..52].try_into().unwrap());
        assert!((sample0 - 0.0).abs() < 1e-15);
        // Second sample is 0.5 f64
        let sample1 = f64::from_le_bytes(data[52..60].try_into().unwrap());
        assert!((sample1 - 0.5).abs() < 1e-15);

        // Cleanup
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_minimum_phase_flat() {
        // Flat correction → phase should be ~0
        let correction = vec![0.0; 513]; // n_bins for n_fft=1024
        let phase = minimum_phase_from_magnitude(&correction, 1024);
        assert_eq!(phase.len(), 513);
        for &p in &phase {
            assert!(p.abs() < 0.01, "Phase should be ~0 for flat correction, got {}", p);
        }
    }

    #[test]
    fn test_bessel_i0() {
        // I₀(0) = 1
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-12);
        // I₀ is always ≥ 1 for x ≥ 0
        assert!(bessel_i0(5.0) > 1.0);
        assert!(bessel_i0(10.0) > bessel_i0(5.0));
    }

    #[test]
    fn test_generate_model_fir_flat() {
        // Flat 0 dB model → FIR realized should be near 0 dB everywhere
        let n = 256;
        let freq: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                (20.0_f64.ln() + t * (20000.0_f64.ln() - 20.0_f64.ln())).exp()
            })
            .collect();
        let mag = vec![0.0; n];
        let phase = vec![0.0; n];

        let config = FirConfig {
            taps: 4096,
            sample_rate: 48000.0,
            max_boost_db: 18.0,
            noise_floor_db: -60.0,
            window: WindowType::Blackman,
            phase_mode: PhaseMode::MinimumPhase,
            iterations: 0, freq_weighting: false, narrowband_limit: false,
            nb_smoothing_oct: 0.333, nb_max_excess_db: 6.0,
            gaussian_min_phase_filters: vec![],
        linear_phase_main: false,
        subsonic_cutoff_hz: None,
        };

        let result = generate_model_fir(&freq, &mag, &[], &phase, &config).unwrap();
        assert_eq!(result.impulse.len(), 4096);
        assert_eq!(result.realized_mag.len(), n);
        assert_eq!(result.realized_phase.len(), n);

        // Realized magnitude should be near 0 dB in the midband (100-10000 Hz)
        for (i, &f) in freq.iter().enumerate() {
            if f >= 100.0 && f <= 10000.0 {
                assert!(
                    result.realized_mag[i].abs() < 3.0,
                    "Realized mag at {:.0} Hz should be near 0 dB, got {:.1}",
                    f, result.realized_mag[i]
                );
            }
        }
    }

    #[test]
    fn test_generate_model_fir_returns_valid() {
        // Low-pass model at -6dB/oct: verify structure
        let n = 128;
        let freq: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                (20.0_f64.ln() + t * (20000.0_f64.ln() - 20.0_f64.ln())).exp()
            })
            .collect();
        let mag: Vec<f64> = freq.iter().map(|&f| {
            // Simple low-pass rolloff above 1kHz
            if f <= 1000.0 { 0.0 } else { -20.0 * (f / 1000.0).log10() }
        }).collect();
        let phase = vec![0.0; n];

        let config = FirConfig {
            taps: 8192,
            sample_rate: 48000.0,
            max_boost_db: 18.0,
            noise_floor_db: -60.0,
            window: WindowType::Hann,
            phase_mode: PhaseMode::LinearPhase,
            iterations: 0,
            freq_weighting: false,
            narrowband_limit: false,
            nb_smoothing_oct: 0.333,
            nb_max_excess_db: 6.0,
            gaussian_min_phase_filters: vec![],
        linear_phase_main: false,
        subsonic_cutoff_hz: None,
        };

        let result = generate_model_fir(&freq, &mag, &[], &phase, &config).unwrap();
        assert_eq!(result.taps, 8192);
        assert_eq!(result.sample_rate, 48000.0);
        // b141.6: time_ms removed from the payload — frontend derives the
        // ramp from taps + sample_rate. Length contract moves to impulse.
        assert_eq!(result.impulse.len(), 8192);
    }

    #[test]
    fn test_interp_1d() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![10.0, 20.0, 30.0, 40.0];
        let q = vec![0.5, 1.5, 2.5, 3.5, 5.0];
        let result = crate::dsp::interp_1d(&x, &y, &q);
        assert!((result[0] - 10.0).abs() < 0.01); // clamped
        assert!((result[1] - 15.0).abs() < 0.01);
        assert!((result[2] - 25.0).abs() < 0.01);
        assert!((result[3] - 35.0).abs() < 0.01);
        assert!((result[4] - 40.0).abs() < 0.01); // clamped
    }

    #[test]
    fn test_linear_phase_fir_symmetry() {
        // Linear phase FIR should produce impulse symmetric around N/2
        let n = 128;
        let freq: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                (20.0_f64.ln() + t * (20000.0_f64.ln() - 20.0_f64.ln())).exp()
            })
            .collect();
        // Simple low-pass: flat to 1kHz, then -20dB/dec rolloff
        let mag: Vec<f64> = freq.iter().map(|&f| {
            if f <= 1000.0 { 0.0 } else { -20.0 * (f / 1000.0).log10() }
        }).collect();
        let phase = vec![0.0; n]; // zero phase model

        let config = FirConfig {
            taps: 8192,
            sample_rate: 48000.0,
            max_boost_db: 18.0,
            noise_floor_db: -60.0,
            window: WindowType::Blackman,
            phase_mode: PhaseMode::LinearPhase,
            iterations: 0,
            freq_weighting: false,
            narrowband_limit: false,
            nb_smoothing_oct: 0.333,
            nb_max_excess_db: 6.0,
            gaussian_min_phase_filters: vec![],
        linear_phase_main: false,
        subsonic_cutoff_hz: None,
        };

        let result = generate_model_fir(&freq, &mag, &[], &phase, &config).unwrap();
        let impulse = &result.impulse;
        let n_fft = impulse.len();
        let center = n_fft / 2;

        // Peak should be at or very near center
        let peak_idx = impulse.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(i, _)| i).unwrap();
        assert!(
            (peak_idx as i64 - center as i64).unsigned_abs() <= 1,
            "Peak at {} should be near center {}", peak_idx, center
        );

        // Check symmetry: impulse[center+k] ≈ impulse[center-k]
        let peak = impulse[center].abs();
        for k in 1..center.min(512) {
            let left = impulse[center - k];
            let right = impulse[center + k];
            let diff = (left - right).abs();
            let thresh = peak * 1e-6; // relative tolerance
            assert!(
                diff < thresh.max(1e-12),
                "Symmetry broken at k={}: left={}, right={}, diff={}",
                k, left, right, diff
            );
        }

        // Realized phase should be near zero (excess phase after delay removal)
        for (i, &f) in freq.iter().enumerate() {
            if f >= 100.0 && f <= 10000.0 {
                assert!(
                    result.realized_phase[i].abs() < 5.0,
                    "Excess phase at {:.0} Hz should be near 0°, got {:.1}°",
                    f, result.realized_phase[i]
                );
            }
        }
    }

    #[test]
    fn test_4band_all_filter_types() {
        use crate::target::{self, TargetCurve, FilterConfig, FilterType};

        // 1. Log frequency grid: 512 points, 20-20000 Hz
        let n = 512;
        let freq: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                (20.0_f64.ln() + t * (20000.0_f64.ln() - 20.0_f64.ln())).exp()
            })
            .collect();

        // 2. All filter type configurations: (label, FilterType, order, shape, linear_phase)
        let filter_types: Vec<(&str, FilterType, u8, Option<f64>, bool)> = vec![
            ("BW4", FilterType::Butterworth, 4, None, false),
            ("BS4", FilterType::Bessel, 4, None, false),
            ("LR4", FilterType::LinkwitzRiley, 4, None, false),
            ("GS2", FilterType::Gaussian, 4, Some(2.0), true),
        ];

        // 3. Band definitions: (name, hp_freq, lp_freq)
        let band_defs: Vec<(&str, Option<f64>, Option<f64>)> = vec![
            ("Sub", None, Some(80.0)),
            ("LowMid", Some(80.0), Some(500.0)),
            ("MidHigh", Some(500.0), Some(3500.0)),
            ("Tweeter", Some(3500.0), None),
        ];

        // Base FIR config (MinimumPhase for analog-type filters)
        let base_config = FirConfig {
            taps: 65536,
            sample_rate: 48000.0,
            max_boost_db: 24.0,
            noise_floor_db: -150.0,
            window: WindowType::Blackman,
            phase_mode: PhaseMode::MinimumPhase,
            iterations: 0,
            freq_weighting: false,
            narrowband_limit: false,
            nb_smoothing_oct: 0.333,
            nb_max_excess_db: 6.0,
            gaussian_min_phase_filters: vec![],
        linear_phase_main: false,
        subsonic_cutoff_hz: None,
        };

        fn make_filter(
            ft: &FilterType, order: u8, freq_hz: f64,
            shape: Option<f64>, linear_phase: bool,
        ) -> FilterConfig {
            FilterConfig {
                filter_type: ft.clone(),
                order,
                freq_hz,
                shape,
                linear_phase,
                q: None,
                subsonic_protect: None,
            }
        }

        let tmp_dir = std::env::temp_dir().join("phaseforge_TEST");
        let _ = std::fs::create_dir_all(&tmp_dir);

        println!("\n=== 4-Band x 4-FilterType FIR Test ===\n");

        let mut total = 0u32;

        for (ft_name, ft, order, shape, linear) in &filter_types {
            // Gaussian with linear_phase → use LinearPhase FIR mode
            let fir_config = if *linear {
                FirConfig {
                    phase_mode: PhaseMode::LinearPhase,
                    ..base_config.clone()
                }
            } else {
                base_config.clone()
            };

            println!("--- {} (order={}, linear={}) ---", ft_name, order, linear);

            for (band_name, hp_freq, lp_freq) in &band_defs {
                let hp = hp_freq.map(|f| make_filter(ft, *order, f, *shape, *linear));
                let lp = lp_freq.map(|f| make_filter(ft, *order, f, *shape, *linear));

                let target_curve = TargetCurve {
                    reference_level_db: 0.0,
                    tilt_db_per_octave: 0.0,
                    tilt_ref_freq: 1000.0,
                    high_pass: hp,
                    low_pass: lp,
                    low_shelf: None,
                    high_shelf: None,
                };

                let response = target::evaluate(&target_curve, &freq);

                // Generate model FIR (no PEQ, no measurement correction)
                let result = generate_model_fir(
                    &freq,
                    &response.magnitude,
                    &[],
                    &response.phase,
                    &fir_config,
                )
                .unwrap_or_else(|e| panic!("{}/{}: generate_model_fir failed: {}", ft_name, band_name, e));

                // --- Assertions ---

                // Structure
                assert_eq!(result.taps, 65536, "{}/{}: wrong taps", ft_name, band_name);
                assert_eq!(result.impulse.len(), 65536, "{}/{}: wrong impulse len", ft_name, band_name);
                assert_eq!(result.realized_mag.len(), n, "{}/{}: wrong realized_mag len", ft_name, band_name);

                // norm_db finite and reasonable
                assert!(
                    result.norm_db.is_finite(),
                    "{}/{}: norm_db is not finite",
                    ft_name, band_name
                );
                assert!(
                    result.norm_db.abs() < 30.0,
                    "{}/{}: norm_db={:.1} too large",
                    ft_name, band_name, result.norm_db
                );

                // No NaN/Inf in impulse
                for (i, &v) in result.impulse.iter().enumerate() {
                    assert!(
                        v.is_finite(),
                        "{}/{}: impulse[{}] is not finite ({})",
                        ft_name, band_name, i, v
                    );
                }

                // Passband realized_mag near 0 dB
                let pb_lo = hp_freq.unwrap_or(20.0) * 1.5;
                let pb_hi = lp_freq.unwrap_or(20000.0) / 1.5;
                if pb_lo < pb_hi {
                    for (i, &f) in freq.iter().enumerate() {
                        if f >= pb_lo && f <= pb_hi {
                            assert!(
                                result.realized_mag[i].abs() < 5.0,
                                "{}/{}: realized_mag at {:.0} Hz = {:.1} dB (expected ~0)",
                                ft_name, band_name, f, result.realized_mag[i]
                            );
                        }
                    }
                }

                // Export WAV
                let wav_path = tmp_dir.join(format!("TEST_{}_{}.wav", ft_name, band_name));
                export_wav_f64(&result.impulse, 48000.0, &wav_path).unwrap();
                assert!(wav_path.exists(), "{}/{}: WAV not created", ft_name, band_name);
                let meta = std::fs::metadata(&wav_path).unwrap();
                let expected_size = 44 + 65536 * 8; // header + 64-bit samples
                assert_eq!(
                    meta.len(),
                    expected_size as u64,
                    "{}/{}: WAV size mismatch",
                    ft_name, band_name
                );

                println!(
                    "  {} | norm_db={:+.2} dB | wav={:.0} KB",
                    band_name,
                    result.norm_db,
                    meta.len() as f64 / 1024.0
                );
                total += 1;
            }
        }

        // Cleanup
        let _ = std::fs::remove_dir_all(&tmp_dir);
        println!("\n=== All {} FIR filters generated and exported OK ===", total);
    }

    /// b141.2: mixed per-filter phase (HP min + LP linear) must produce a FIR
    /// whose realised magnitude AND phase match the model the UI displays.
    /// Before b141.2 this band routed to the IIR cascade and collapsed both
    /// filters to min-phase; the cepstral `use_model_phase` path now honours the
    /// per-filter model phase verbatim.
    #[test]
    fn mixed_phase_fir_honours_hp_min_lp_linear() {
        use crate::target::{evaluate, FilterConfig, FilterType, TargetCurve};

        let n = 512;
        let freq: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                (20.0_f64.ln() + t * (20000.0_f64.ln() - 20.0_f64.ln())).exp()
            })
            .collect();

        let hp = FilterConfig {
            filter_type: FilterType::LinkwitzRiley, order: 4, freq_hz: 200.0,
            shape: None, linear_phase: false, q: None, subsonic_protect: None,
        };
        let lp = FilterConfig {
            filter_type: FilterType::LinkwitzRiley, order: 4, freq_hz: 2000.0,
            shape: None, linear_phase: true, q: None, subsonic_protect: None,
        };
        let tc = TargetCurve {
            reference_level_db: 0.0, tilt_db_per_octave: 0.0, tilt_ref_freq: 1000.0,
            high_pass: Some(hp), low_pass: Some(lp), low_shelf: None, high_shelf: None,
        };
        // Model: magnitude = full bandpass; phase = HP min-phase only (the
        // linear-phase LP contributes zero excess phase — exactly what the UI
        // shows on the export tab).
        let model = evaluate(&tc, &freq);

        let config = FirConfig {
            taps: 32768, sample_rate: 48000.0, max_boost_db: 24.0, noise_floor_db: -150.0,
            window: WindowType::Blackman, phase_mode: PhaseMode::MixedPhase,
            iterations: 0, freq_weighting: false, narrowband_limit: false,
            nb_smoothing_oct: 0.333, nb_max_excess_db: 6.0,
            gaussian_min_phase_filters: vec![],
            linear_phase_main: false, subsonic_cutoff_hz: None,
        };

        let result = generate_model_fir(&freq, &model.magnitude, &[], &model.phase, &config)
            .expect("mixed-phase FIR");

        // The model phase must be non-trivial (HP LR4 min-phase) or the test
        // would pass vacuously against an all-zero phase.
        assert!(
            model.phase.iter().any(|p| p.abs() > 10.0),
            "HP min-phase should give a non-trivial model phase",
        );
        if let Some(pi) = freq.iter().position(|&f| f >= 400.0) {
            eprintln!(
                "sample @ {:.0} Hz: model φ={:.1}° realised φ={:.1}° | model mag={:.2} realised={:.2} dB",
                freq[pi], model.phase[pi], result.realized_phase[pi],
                model.magnitude[pi], result.realized_mag[pi],
            );
        }

        // Passband 300–1500 Hz: realised must track the model.
        let mut max_dmag = 0.0_f64;
        let mut max_dphase = 0.0_f64;
        for (i, &f) in freq.iter().enumerate() {
            if !(300.0..=1500.0).contains(&f) { continue; }
            let dmag = (result.realized_mag[i] - model.magnitude[i]).abs();
            // Wrap the phase difference to (-180, 180] — realised phase is
            // unwrapped (can be multi-turn) while the model is wrapped; they are
            // equal modulo 360° when the FIR honours the model.
            let raw = result.realized_phase[i] - model.phase[i];
            let dph = (raw - 360.0 * (raw / 360.0).round()).abs();
            if dmag > max_dmag { max_dmag = dmag; }
            if dph > max_dphase { max_dphase = dph; }
        }
        eprintln!("mixed-phase passband: max Δmag={:.2} dB, max Δphase={:.1}°", max_dmag, max_dphase);
        assert!(max_dmag < 1.5, "realized magnitude diverges from model: {:.2} dB", max_dmag);
        assert!(max_dphase < 15.0, "realized phase diverges from model: {:.1}°", max_dphase);
    }

    // NOTE: original phase-match test for the Gaussian MixedPhase path was
    // removed because peak-centering perturbs phase; the b141.2 model-phase
    // path above is the supported mixed-phase case.
    #[test]
    fn test_fir_matches_model_magnitude() {
        // Test that FIR realized_mag matches the target Model magnitude
        // for Gaussian filters across all 4 phase combinations.
        // Key insight: changing lin-phase vs min-phase should ONLY affect PHASE,
        // not MAGNITUDE. If FIR magnitude changes with phase mode, there's a bug.

        let n = 256;
        let freq: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                (20.0_f64.ln() + t * (20000.0_f64.ln() - 20.0_f64.ln())).exp()
            })
            .collect();

        let fc_hp = 200.0;
        let fc_lp = 2000.0;
        let shape = 2.0;
        let ln2 = 2.0_f64.ln();

        // Build Gaussian bandpass target magnitude
        let target_mag: Vec<f64> = freq.iter().map(|&f| {
            let hp_ratio = if fc_hp > 0.0 { f / fc_hp } else { 0.0 };
            let hp_lin = 1.0 - (-ln2 * hp_ratio.powf(2.0 * shape)).exp();
            let lp_ratio = if fc_lp > 0.0 { f / fc_lp } else { 0.0 };
            let lp_lin = (-ln2 * lp_ratio.powf(2.0 * shape)).exp();
            let lin = hp_lin * lp_lin;
            if lin > 1e-20 { 20.0 * lin.log10() } else { -400.0 }
        }).collect();

        let phase_zero = vec![0.0; n];

        let base_config = FirConfig {
            taps: 16384,
            sample_rate: 48000.0,
            max_boost_db: 24.0,
            noise_floor_db: -150.0,
            window: WindowType::Blackman,
            phase_mode: PhaseMode::LinearPhase,
            iterations: 0,
            freq_weighting: false,
            narrowband_limit: false,
            nb_smoothing_oct: 0.333,
            nb_max_excess_db: 6.0,
            gaussian_min_phase_filters: vec![],
        linear_phase_main: false,
        subsonic_cutoff_hz: None,
        };

        // Passband indices: 350-1200 Hz
        let passband_indices: Vec<usize> = freq.iter().enumerate()
            .filter(|(_, &f)| f >= 350.0 && f <= 1200.0)
            .map(|(i, _)| i)
            .collect();
        assert!(!passband_indices.is_empty(), "No passband frequencies found");

        // Compute target mag in passband (for reference)
        let target_passband: Vec<f64> = passband_indices.iter()
            .map(|&i| target_mag[i])
            .collect();

        // Define the 4 phase combinations
        struct PhaseCase {
            name: &'static str,
            phase_mode: PhaseMode,
            min_phase_filters: Vec<GaussianFilterInfo>,
        }

        let cases = vec![
            PhaseCase {
                name: "HP=lin, LP=lin",
                phase_mode: PhaseMode::LinearPhase,
                min_phase_filters: vec![],
            },
            PhaseCase {
                name: "HP=min, LP=lin",
                phase_mode: PhaseMode::MixedPhase,
                min_phase_filters: vec![
                    GaussianFilterInfo { freq_hz: fc_hp, shape, is_lowpass: false },
                ],
            },
            PhaseCase {
                name: "HP=lin, LP=min",
                phase_mode: PhaseMode::MixedPhase,
                min_phase_filters: vec![
                    GaussianFilterInfo { freq_hz: fc_lp, shape, is_lowpass: true },
                ],
            },
            PhaseCase {
                name: "HP=min, LP=min",
                phase_mode: PhaseMode::MixedPhase,
                min_phase_filters: vec![
                    GaussianFilterInfo { freq_hz: fc_hp, shape, is_lowpass: false },
                    GaussianFilterInfo { freq_hz: fc_lp, shape, is_lowpass: true },
                ],
            },
        ];

        // Store passband realized_mag for each case (for cross-comparison)
        let mut all_passband_mags: Vec<(String, Vec<f64>)> = Vec::new();

        for case in &cases {
            let config = FirConfig {
                phase_mode: case.phase_mode.clone(),
                gaussian_min_phase_filters: case.min_phase_filters.clone(),
                ..base_config.clone()
            };

            let result = generate_model_fir(&freq, &target_mag, &[], &phase_zero, &config).unwrap();

            // Compare realized_mag vs target_mag in passband
            let realized_passband: Vec<f64> = passband_indices.iter()
                .map(|&i| result.realized_mag[i])
                .collect();

            let errors: Vec<f64> = realized_passband.iter()
                .zip(target_passband.iter())
                .map(|(&r, &t)| (r - t).abs())
                .collect();

            let max_err = errors.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let rms_err = (errors.iter().map(|e| e * e).sum::<f64>() / errors.len() as f64).sqrt();

            // Find frequency of max error
            let max_err_idx = errors.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            let max_err_freq = freq[passband_indices[max_err_idx]];

            println!(
                "[{}] max_err={:.3} dB @ {:.0} Hz, rms_err={:.3} dB",
                case.name, max_err, max_err_freq, rms_err
            );

            // Print a few sample points for diagnostics
            for &idx in &[0, passband_indices.len() / 4, passband_indices.len() / 2, 3 * passband_indices.len() / 4, passband_indices.len() - 1] {
                let fi = passband_indices[idx];
                println!(
                    "  f={:.0} Hz: target={:.2} dB, realized={:.2} dB, err={:.3} dB",
                    freq[fi], target_mag[fi], result.realized_mag[fi], errors[idx]
                );
            }

            assert!(
                max_err < 1.0,
                "[{}] max passband error {:.3} dB >= 1.0 dB @ {:.0} Hz",
                case.name, max_err, max_err_freq
            );
            assert!(
                rms_err < 0.5,
                "[{}] RMS passband error {:.3} dB >= 0.5 dB",
                case.name, rms_err
            );

            all_passband_mags.push((case.name.to_string(), realized_passband));
        }

        // Cross-compare all cases: magnitude should be nearly identical
        // (changing phase mode should NOT change magnitude)
        println!("\n--- Cross-comparison of passband magnitudes ---");
        let ref_name = &all_passband_mags[0].0;
        let ref_mag = &all_passband_mags[0].1;

        for (name, mag) in &all_passband_mags[1..] {
            let cross_errors: Vec<f64> = mag.iter()
                .zip(ref_mag.iter())
                .map(|(&a, &b)| (a - b).abs())
                .collect();
            let cross_max = cross_errors.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let cross_rms = (cross_errors.iter().map(|e| e * e).sum::<f64>() / cross_errors.len() as f64).sqrt();

            let cross_max_idx = cross_errors.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            let cross_max_freq = freq[passband_indices[cross_max_idx]];

            println!(
                "[{} vs {}] max_diff={:.3} dB @ {:.0} Hz, rms_diff={:.3} dB",
                ref_name, name, cross_max, cross_max_freq, cross_rms
            );

            assert!(
                cross_max < 1.0,
                "[{} vs {}] cross-case max diff {:.3} dB >= 1.0 dB @ {:.0} Hz",
                ref_name, name, cross_max, cross_max_freq
            );
            assert!(
                cross_rms < 0.5,
                "[{} vs {}] cross-case RMS diff {:.3} dB >= 0.5 dB",
                ref_name, name, cross_rms
            );
        }
    }

    /// E2E test: FIR realized_mag must match target_mag in passband for ALL phase modes.
    /// This catches the half-window-on-non-causal-impulse bug (88 dB error).
    #[test]
    fn test_fir_magnitude_matches_target_all_phase_modes() {
        let n = 512;
        let freq: Vec<f64> = (0..n).map(|i| {
            let t = i as f64 / (n - 1) as f64;
            (20.0_f64.ln() + t * (20000.0_f64.ln() - 20.0_f64.ln())).exp()
        }).collect();
        let fc_hp = 200.0;
        let fc_lp = 2000.0;
        let shape = 2.0;
        let pb_lo = 350.0;
        let pb_hi = 1200.0;

        // Compute bandpass target magnitude
        let ln2 = 2.0_f64.ln();
        let target_mag: Vec<f64> = freq.iter().map(|&f| {
            let r_lp: f64 = f / fc_lp;
            let r_hp: f64 = f / fc_hp;
            let lp = (-ln2 * r_lp.powf(2.0 * shape)).exp();
            let hp = 1.0 - (-ln2 * r_hp.powf(2.0 * shape)).exp();
            let bp = hp * lp;
            if bp > 1e-20 { 20.0 * bp.log10() } else { -400.0 }
        }).collect();

        let cases: Vec<(&str, PhaseMode, Vec<GaussianFilterInfo>)> = vec![
            ("LinearPhase", PhaseMode::LinearPhase, vec![]),
            ("MinimumPhase", PhaseMode::MinimumPhase, vec![]),
            ("MixedPhase HP-only", PhaseMode::MixedPhase, vec![
                GaussianFilterInfo { freq_hz: fc_hp, shape, is_lowpass: false },
            ]),
            ("MixedPhase LP-only", PhaseMode::MixedPhase, vec![
                GaussianFilterInfo { freq_hz: fc_lp, shape, is_lowpass: true },
            ]),
            ("MixedPhase HP+LP", PhaseMode::MixedPhase, vec![
                GaussianFilterInfo { freq_hz: fc_hp, shape, is_lowpass: false },
                GaussianFilterInfo { freq_hz: fc_lp, shape, is_lowpass: true },
            ]),
        ];

        println!("\n=== FIR Magnitude Match Test (E2E) ===");
        for (name, mode, gauss_filters) in &cases {
            let config = FirConfig {
                taps: 65536,
                sample_rate: 48000.0,
                max_boost_db: 24.0,
                noise_floor_db: -150.0,
                window: WindowType::Blackman,
                phase_mode: mode.clone(),
                iterations: 0,
                freq_weighting: false,
                narrowband_limit: false,
                nb_smoothing_oct: 0.333,
                nb_max_excess_db: 6.0,
                gaussian_min_phase_filters: gauss_filters.clone(),
                linear_phase_main: false,
                subsonic_cutoff_hz: None,
            };

            let zero_phase = vec![0.0; n];
            let result = generate_model_fir(&freq, &target_mag, &[], &zero_phase, &config)
                .unwrap_or_else(|e| panic!("{}: generate_model_fir failed: {}", name, e));

            // Normalize target same as FIR engine does
            let norm_target: Vec<f64> = target_mag.iter().map(|&v| v - result.norm_db).collect();

            // Compute passband error
            let mut max_err = 0.0_f64;
            let mut max_err_freq = 0.0;
            let mut sum_sq = 0.0;
            let mut count = 0;
            for i in 0..freq.len() {
                if freq[i] >= pb_lo && freq[i] <= pb_hi {
                    let err = (result.realized_mag[i] - norm_target[i]).abs();
                    if err > max_err {
                        max_err = err;
                        max_err_freq = freq[i];
                    }
                    sum_sq += err * err;
                    count += 1;
                }
            }
            let rms_err = if count > 0 { (sum_sq / count as f64).sqrt() } else { 0.0 };

            println!(
                "{}: maxErr={:.2}dB@{:.0}Hz, RMS={:.2}dB, norm_db={:.2}",
                name, max_err, max_err_freq, rms_err, result.norm_db
            );

            assert!(
                max_err < 3.0,
                "{}: magnitude error {:.2} dB @ {:.0} Hz exceeds 3 dB threshold. \
                 This likely means half-window was applied to non-causal impulse.",
                name, max_err, max_err_freq
            );
            assert!(
                rms_err < 1.0,
                "{}: RMS magnitude error {:.2} dB exceeds 1 dB threshold.",
                name, rms_err
            );
        }
    }

    // -----------------------------------------------------------------------
    // b139.0 golden snapshot for generate_model_fir.
    // -----------------------------------------------------------------------

    /// Stable FNV-1a 64-bit over the rounded impulse — no extra crate needed.
    fn b139_impulse_hash(impulse: &[f64]) -> String {
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;
        let mut h: u64 = FNV_OFFSET;
        for &v in impulse {
            // Round to 6 decimals first so hash is stable against float noise.
            let r = (v * 1_000_000.0).round() as i64;
            for byte in r.to_le_bytes().iter() {
                h ^= *byte as u64;
                h = h.wrapping_mul(FNV_PRIME);
            }
        }
        format!("{:016x}", h)
    }

    #[test]
    fn generate_fir_b139_golden_lr4_baseline_impulse_hash() {
        // Reference config: LR4 HP=80 Hz, no PEQ, sr=48000, taps=8192,
        // Blackman window, MinimumPhase mode (LR4 HP without linear_phase).
        let n = 512;
        let freq: Vec<f64> = (0..n).map(|i| 5.0 * (40000f64 / 5.0).powf(i as f64 / (n - 1) as f64)).collect();
        // Build LR4 HP=80 magnitude in dB on this grid (8th-order Butterworth-squared).
        let target_mag: Vec<f64> = freq.iter().map(|&f| {
            // |H_LR4_HP|^2 = (f/fc)^16 / (1 + (f/fc)^16)
            let r = (f / 80.0).powi(8);
            let mag_lin = r / (1.0 + r);
            10.0 * (mag_lin.max(1e-20)).log10()
        }).collect();

        let config = FirConfig {
            taps: 8192,
            sample_rate: 48000.0,
            max_boost_db: 18.0,
            noise_floor_db: -150.0,
            window: WindowType::Blackman,
            phase_mode: PhaseMode::MinimumPhase,
            iterations: 0,
            freq_weighting: false,
            narrowband_limit: false,
            nb_smoothing_oct: 0.333,
            nb_max_excess_db: 6.0,
            gaussian_min_phase_filters: vec![],
        linear_phase_main: false,
        subsonic_cutoff_hz: None,
        };

        let result = generate_model_fir(&freq, &target_mag, &[], &vec![0.0; n], &config)
            .expect("generate_model_fir should succeed");
        assert_eq!(result.impulse.len(), 8192);

        let hash = b139_impulse_hash(&result.impulse);
        // Captured from b138.4 reference run; any change in the FIR pipeline
        // that touches this config flips the hash → investigate before
        // accepting.
        let expected = "4574e5da87ade187"; // b141.14: unified WAV peak (N/2 shift)
        assert_eq!(hash, expected,
            "FIR impulse hash drift — capture new value from this test failure if intentional");
    }

    // -----------------------------------------------------------------------
    // b139.3 lock-in tests: Gaussian-and-subsonic FIR phase paths. Each test
    // captures the b138.4 hash so stage-4/5 migrations can detect drift.
    // -----------------------------------------------------------------------

    fn b139_3_gaussian_hp_mag(freq: &[f64], fc: f64, m: f64, with_subsonic: bool) -> Vec<f64> {
        let ln2 = 2.0_f64.ln();
        let f_sub = fc / 8.0;
        freq.iter().map(|&f| {
            if f <= 0.0 { return -400.0; }
            // Gaussian HP = 1 - LP(f/fc)
            let ratio = (f / fc).powf(2.0 * m);
            let lp_lin = (-ln2 * ratio).exp();
            let hp_lin = 1.0 - lp_lin;
            let mut db = if hp_lin > 1e-20 { 20.0 * hp_lin.log10() } else { -400.0 };
            if with_subsonic {
                let sub_ratio = (f_sub / f).powi(16);
                db += -10.0 * (1.0 + sub_ratio).log10();
            }
            db
        }).collect()
    }

    fn b139_3_fir_config(phase_mode: PhaseMode) -> FirConfig {
        FirConfig {
            taps: 8192, sample_rate: 48000.0,
            max_boost_db: 18.0, noise_floor_db: -150.0,
            window: WindowType::Blackman, phase_mode,
            iterations: 0, freq_weighting: false, narrowband_limit: false,
            nb_smoothing_oct: 0.333, nb_max_excess_db: 6.0,
            gaussian_min_phase_filters: vec![],
        linear_phase_main: false,
        subsonic_cutoff_hz: None,
        }
    }

    /// Gaussian HP=632, linear_phase=true, subsonic ON → caller demotes to
    /// MinimumPhase (b138.4 isLin), Rust Hilbert reconstructs subsonic phase
    /// from full magnitude. Lock the resulting hash.
    #[test]
    fn generate_fir_b139_3_gaussian_lin_subsonic_min_phase() {
        let n = 512;
        let freq: Vec<f64> = (0..n).map(|i| 5.0 * (40000f64 / 5.0).powf(i as f64 / (n - 1) as f64)).collect();
        let target_mag = b139_3_gaussian_hp_mag(&freq, 632.0, 1.0, true);
        let result = generate_model_fir(&freq, &target_mag, &[], &vec![0.0; n], &b139_3_fir_config(PhaseMode::MinimumPhase))
            .expect("generate_model_fir should succeed");
        let hash = b139_impulse_hash(&result.impulse);
        let expected = "dd841e3f05cb81a9"; // b141.14: unified WAV peak (N/2 shift)
        assert_eq!(hash, expected,
            "Gaussian linear + subsonic FIR hash drift — capture new value if intentional");
    }

    /// Gaussian HP=632, linear_phase=false, subsonic ON → MinimumPhase mode,
    /// Hilbert from (Gaussian × subsonic) magnitude. Same Rust input as
    /// the linear+subsonic case (caller demoted it), so same hash.
    #[test]
    fn generate_fir_b139_3_gaussian_min_subsonic_min_phase() {
        let n = 512;
        let freq: Vec<f64> = (0..n).map(|i| 5.0 * (40000f64 / 5.0).powf(i as f64 / (n - 1) as f64)).collect();
        let target_mag = b139_3_gaussian_hp_mag(&freq, 632.0, 1.0, true);
        let result = generate_model_fir(&freq, &target_mag, &[], &vec![0.0; n], &b139_3_fir_config(PhaseMode::MinimumPhase))
            .expect("generate_model_fir should succeed");
        let hash = b139_impulse_hash(&result.impulse);
        let expected = "dd841e3f05cb81a9"; // b141.14: unified WAV peak (N/2 shift)
        assert_eq!(hash, expected,
            "Gaussian min-phase + subsonic FIR hash drift — capture new value if intentional");
    }

    /// LR4 HP=80 with one peaking PEQ band at 1kHz Q=4 +6dB. PEQ phase
    /// contribution comes from Rust Hilbert over the PEQ-only magnitude
    /// (already in generate_model_fir today). Lock the resulting hash.
    #[test]
    fn generate_fir_b139_3_lr4_with_peq_peak() {
        let n = 512;
        let freq: Vec<f64> = (0..n).map(|i| 5.0 * (40000f64 / 5.0).powf(i as f64 / (n - 1) as f64)).collect();
        let target_mag: Vec<f64> = freq.iter().map(|&f| {
            let r = (f / 80.0).powi(8);
            let mag_lin = r / (1.0 + r);
            10.0 * (mag_lin.max(1e-20)).log10()
        }).collect();
        // PEQ magnitude (peaking biquad analytic): A=10^(gain/40), w0=2π*fc/fs,
        // alpha=sin(w0)/(2Q). For sr=48k, fc=1k, Q=4, gain=6dB.
        let sr = 48000.0;
        let peq_mag: Vec<f64> = freq.iter().map(|&f| {
            let a = 10f64.powf(6.0 / 40.0);
            let w0 = 2.0 * std::f64::consts::PI * 1000.0 / sr;
            let cos_w0 = w0.cos();
            let sin_w0 = w0.sin();
            let alpha = sin_w0 / (2.0 * 4.0);
            let b0 = 1.0 + alpha * a;
            let b1 = -2.0 * cos_w0;
            let b2 = 1.0 - alpha * a;
            let a0 = 1.0 + alpha / a;
            let a1 = -2.0 * cos_w0;
            let a2 = 1.0 - alpha / a;
            // |H(e^{jw})|^2
            let w = 2.0 * std::f64::consts::PI * f / sr;
            let cos_w = w.cos();
            let cos_2w = (2.0 * w).cos();
            let num = b0 * b0 + b1 * b1 + b2 * b2 + 2.0 * (b0 * b1 + b1 * b2) * cos_w + 2.0 * b0 * b2 * cos_2w;
            let den = a0 * a0 + a1 * a1 + a2 * a2 + 2.0 * (a0 * a1 + a1 * a2) * cos_w + 2.0 * a0 * a2 * cos_2w;
            10.0 * (num / den).max(1e-20).log10()
        }).collect();
        let result = generate_model_fir(&freq, &target_mag, &peq_mag, &vec![0.0; n], &b139_3_fir_config(PhaseMode::MinimumPhase))
            .expect("generate_model_fir should succeed");
        let hash = b139_impulse_hash(&result.impulse);
        let expected = "f5d8b961a20d053a"; // b141.14: unified WAV peak (N/2 shift)
        assert_eq!(hash, expected,
            "LR4 + PEQ FIR hash drift — capture new value if intentional");
    }

    // -----------------------------------------------------------------------
    // b139.3.1 identity / subsonic regression tests. These are physical
    // correctness tests, not pinned hashes — so they survive future
    // refactors as long as the math stays right.
    // -----------------------------------------------------------------------

    fn b139_3_1_log_grid(n: usize, fmin: f64, fmax: f64) -> Vec<f64> {
        (0..n).map(|i| fmin * (fmax / fmin).powf(i as f64 / (n - 1) as f64)).collect()
    }

    /// Find the absolute-peak sample index of an impulse response.
    fn b139_3_1_peak_idx(impulse: &[f64]) -> usize {
        let mut best = 0usize;
        let mut best_v = 0.0_f64;
        for (i, &v) in impulse.iter().enumerate() {
            if v.abs() > best_v.abs() {
                best_v = v;
                best = i;
            }
        }
        best
    }

    fn b139_3_1_off_peak_energy(impulse: &[f64], peak_idx: usize) -> f64 {
        let mut s = 0.0_f64;
        for (i, &v) in impulse.iter().enumerate() {
            if i != peak_idx { s += v * v; }
        }
        s
    }

    /// Flat target + zero phase + no PEQ + LinearPhase mode → FIR must be
    /// (windowed) impulse: one large peak with ≪ peak² energy elsewhere.
    #[test]
    fn fir_identity_for_flat_input_no_filters() {
        let n = 512;
        let freq = b139_3_1_log_grid(n, 5.0, 40000.0);
        let target_mag = vec![0.0; n];
        let target_phase = vec![0.0; n];
        let peq_mag: Vec<f64> = vec![];

        let cfg = b139_3_fir_config(PhaseMode::LinearPhase);
        let result = generate_model_fir(&freq, &target_mag, &peq_mag, &target_phase, &cfg)
            .expect("generate_model_fir should succeed");

        let pi = b139_3_1_peak_idx(&result.impulse);
        let pv = result.impulse[pi];
        let off = b139_3_1_off_peak_energy(&result.impulse, pi);
        // Window scales the peak (Blackman has gain ~0.42 at center, but the
        // FIR is already DFT-amplitude-normalized + window-energy-corrected).
        // The realised peak should be O(1).
        assert!(pv.abs() > 0.5 && pv.abs() < 1.5,
            "Identity LinearPhase FIR peak should be ~1.0, got {pv} at idx {pi}");
        assert!(off < 0.1 * pv * pv,
            "Identity FIR should have ≪10% energy off peak, got off-peak energy {off}, peak² {}",
            pv * pv);
    }

    /// Same baseline but MinimumPhase mode — peak is at the start (causal),
    /// off-peak energy still small.
    #[test]
    fn fir_identity_with_min_phase_mode() {
        let n = 512;
        let freq = b139_3_1_log_grid(n, 5.0, 40000.0);
        let target_mag = vec![0.0; n];
        let target_phase = vec![0.0; n];
        let peq_mag: Vec<f64> = vec![];

        let cfg = b139_3_fir_config(PhaseMode::MinimumPhase);
        let result = generate_model_fir(&freq, &target_mag, &peq_mag, &target_phase, &cfg)
            .expect("generate_model_fir should succeed");

        let pi = b139_3_1_peak_idx(&result.impulse);
        let pv = result.impulse[pi];
        let off = b139_3_1_off_peak_energy(&result.impulse, pi);
        assert!(pv.abs() > 0.5 && pv.abs() < 1.5,
            "Identity MinimumPhase FIR peak should be ~1.0, got {pv}");
        assert!(off < 0.1 * pv * pv,
            "Identity MinimumPhase FIR should have ≪10% energy off peak, got {off}");
        // b141.14: unified WAV peak convention — the causal min-phase impulse
        // ships with an adaptive shift to N/2 (parity with linear-phase and
        // IIR-path WAVs). For an identity FIR the tail decays instantly, so
        // the shift is exactly N/2.
        assert!(pi >= cfg.taps / 2 && pi <= cfg.taps / 2 + 32,
            "MinimumPhase peak should be near N/2 (unified WAV convention), got idx {pi} of {} taps", cfg.taps);
    }

    /// b139.3.2/3: real-world divergence reproducer. Kirill's logs showed
    /// iterative_refine errors GROWING (0.151 → 12.091 → 13.486 dB) when
    /// linear-phase Gaussian + subsonic gets demoted to MinimumPhase mode.
    /// Pure single-pass tests never see this because they don't run iter=2+.
    /// b139.3.3 uses the production grid generator and the production
    /// FirConfig; if it still PASSes, divergence depends on something else
    /// the harness doesn't cover yet.
    #[test]
    fn iterative_refine_converges_with_min_phase_subsonic() {
        use crate::target::{evaluate, FilterConfig, FilterType, TargetCurve};
        let n = 512;
        // Production-shape grid: same as evaluate_target_standalone with
        // the explicit 5–40k range fir-export historically passed.
        let freq = crate::dsp::generate_log_freq_grid(n, 5.0, 40000.0);
        let target_curve = TargetCurve {
            reference_level_db: 0.0,
            tilt_db_per_octave: 0.0,
            tilt_ref_freq: 1000.0,
            high_pass: Some(FilterConfig {
                filter_type: FilterType::Gaussian,
                order: 4,
                freq_hz: 632.0,
                shape: Some(1.0),
                linear_phase: true,
                q: None,
                subsonic_protect: Some(true),
            }),
            low_pass: None,
            low_shelf: None,
            high_shelf: None,
        };
        let resp = evaluate(&target_curve, &freq);
        let target_mag = resp.magnitude;
        let target_phase = vec![0.0; n];
        let peq_mag: Vec<f64> = vec![];

        // Production-shape config: 65k taps, iterations=3, freq_weighting on,
        // narrowband_limit on — matches what fir-export.ts sends.
        let cfg = FirConfig {
            taps: 65536, sample_rate: 48000.0,
            max_boost_db: 24.0, noise_floor_db: -150.0,
            window: WindowType::Blackman,
            phase_mode: PhaseMode::MinimumPhase,
            iterations: 3, freq_weighting: true,
            narrowband_limit: true,
            nb_smoothing_oct: 0.333,
            nb_max_excess_db: 6.0,
            gaussian_min_phase_filters: vec![],
        linear_phase_main: false,
        subsonic_cutoff_hz: None,
        };
        super::helpers::iter_stats_reset();
        let result = generate_model_fir(&freq, &target_mag, &peq_mag, &target_phase, &cfg)
            .expect("generate_model_fir should succeed");
        let history = super::helpers::iter_stats_take();
        eprintln!("iterative_refine history ({} iters):", history.len());
        for s in &history {
            eprintln!("  iter={} max_err={:.3} dB rms={:.3} dB", s.iter, s.max_err, s.rms_err);
        }
        // Sanity: errors must not blow up between iterations.
        let mut prev = f64::INFINITY;
        for s in &history {
            if s.iter > 1 {
                assert!(s.max_err <= prev * 1.5 + 1.0,
                    "DIVERGENCE at iter {}: max_err {:.3} dB > 1.5 × previous {:.3} dB",
                    s.iter, s.max_err, prev);
            }
            prev = s.max_err;
        }

        // Check passband (1 kHz – 10 kHz) — well above subsonic, well above
        // Gaussian rolloff. Realised magnitude must track target within 1 dB
        // after iterative_refine converges.
        let mut max_err = 0.0_f64;
        let mut max_err_freq = 0.0;
        for (i, &f) in freq.iter().enumerate() {
            if f < 1000.0 || f > 10000.0 { continue; }
            let realised = result.realized_mag[i];
            let expected = target_mag[i] - result.norm_db;
            let err = (realised - expected).abs();
            if err > max_err { max_err = err; max_err_freq = f; }
        }
        eprintln!("iterative_refine_converges: passband max_err = {:.3} dB at {:.1} Hz",
            max_err, max_err_freq);
        assert!(max_err < 1.0,
            "iterative_refine diverged: passband max_err = {:.3} dB at {:.1} Hz \
             (target -10..0 dB drift). This reproduces Kirill's b139.3 bug.",
            max_err, max_err_freq);
    }

    /// Linear-phase Gaussian HP=632 + subsonic ON, run through generate_model_fir
    /// in MinimumPhase mode (matches b138.4 isLin demotion). Single-pass
    /// (iterations=0) — passband should track target.
    #[test]
    fn fir_linear_gaussian_with_subsonic_keeps_passband_intact() {
        let n = 512;
        let freq = b139_3_1_log_grid(n, 5.0, 40000.0);
        let target_mag = b139_3_gaussian_hp_mag(&freq, 632.0, 1.0, true);
        let target_phase = vec![0.0; n];
        let peq_mag: Vec<f64> = vec![];

        let cfg = b139_3_fir_config(PhaseMode::MinimumPhase);
        let result = generate_model_fir(&freq, &target_mag, &peq_mag, &target_phase, &cfg)
            .expect("generate_model_fir should succeed");

        // Realised magnitude is computed by Rust on the input freq grid.
        // Compare to target_mag in the passband (1 kHz – 10 kHz) — must
        // align modulo norm_db offset and clipping.
        let mut max_err = 0.0_f64;
        for (i, &f) in freq.iter().enumerate() {
            if f < 1000.0 || f > 10000.0 { continue; }
            let realised = result.realized_mag[i];
            let expected = target_mag[i] - result.norm_db;
            max_err = max_err.max((realised - expected).abs());
        }
        assert!(max_err < 1.0,
            "Realised magnitude in passband (1k–10k) drifts from target by up to {max_err:.3} dB");
    }

    // -----------------------------------------------------------------------
    // b139.4a — Composite phase mode (linear-phase main + min-phase subsonic).
    // -----------------------------------------------------------------------

    fn b139_4a_composite_config(linear_main: bool, subsonic_cutoff: Option<f64>) -> FirConfig {
        FirConfig {
            taps: 8192, sample_rate: 48000.0,
            max_boost_db: 24.0, noise_floor_db: -150.0,
            window: WindowType::Blackman,
            phase_mode: PhaseMode::Composite,
            iterations: 3, freq_weighting: true, narrowband_limit: false,
            nb_smoothing_oct: 0.333, nb_max_excess_db: 6.0,
            gaussian_min_phase_filters: vec![],
            linear_phase_main: linear_main,
            subsonic_cutoff_hz: subsonic_cutoff,
        }
    }

    /// Find the realized_phase value at the freq closest to `target_hz`.
    fn b139_4a_phase_at(freq: &[f64], realized_phase: &[f64], target_hz: f64) -> f64 {
        let idx = freq.iter().enumerate()
            .min_by(|(_, a), (_, b)| (**a - target_hz).abs().partial_cmp(&(**b - target_hz).abs()).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        realized_phase[idx]
    }

    /// Maximum |phase| over a frequency window.
    fn b139_4a_max_abs_phase(freq: &[f64], realized_phase: &[f64], lo: f64, hi: f64) -> f64 {
        let mut m = 0.0_f64;
        for (i, &f) in freq.iter().enumerate() {
            if f < lo || f > hi { continue; }
            if realized_phase[i].abs() > m { m = realized_phase[i].abs(); }
        }
        m
    }

    /// Composite, linear main, no subsonic → reduces to LinearPhase semantics:
    /// passband phase ≈ 0 across the audible band.
    #[test]
    fn fir_composite_lin_main_no_subsonic() {
        let n = 512;
        let freq = b139_3_1_log_grid(n, 5.0, 40000.0);
        let target_mag = b139_3_gaussian_hp_mag(&freq, 632.0, 1.0, false);
        let cfg = b139_4a_composite_config(true, None);
        let result = generate_model_fir(&freq, &target_mag, &[], &vec![0.0; n], &cfg)
            .expect("generate_model_fir should succeed");
        let max_phase = b139_4a_max_abs_phase(&freq, &result.realized_phase, 1000.0, 10000.0);
        assert!(max_phase < 1.0,
            "Composite lin main, no subsonic: passband phase should be ~0°, got max |phase| = {:.3}°",
            max_phase);
    }

    /// Composite, linear main, subsonic ON → main filter is linear-phase
    /// (flat phase across passband), subsonic carries the only min-phase
    /// rotation. Cumulative phase at passband includes the full -720° BW8
    /// asymptote, so we check passband *flatness* (slope), not absolute.
    #[test]
    fn fir_composite_lin_main_with_subsonic() {
        let n = 512;
        let freq = b139_3_1_log_grid(n, 5.0, 40000.0);
        let target_mag = b139_3_gaussian_hp_mag(&freq, 632.0, 1.0, true);
        let cfg = b139_4a_composite_config(true, Some(79.0));
        let result = generate_model_fir(&freq, &target_mag, &[], &vec![0.0; n], &cfg)
            .expect("generate_model_fir should succeed");

        // Passband flatness: phase variation 1k–10k must be small. The
        // subsonic Hilbert has asymptoted to -720° well before 1 kHz, so the
        // residual slope across the passband should be tiny (< 5°).
        let p_lo = b139_4a_phase_at(&freq, &result.realized_phase, 1000.0);
        let p_hi = b139_4a_phase_at(&freq, &result.realized_phase, 10000.0);
        let pass_var = (p_hi - p_lo).abs();
        // BW8 HP (fc=79) hasn't fully reached its -720° asymptote at 1 kHz;
        // residual slope from 1k→10k is ~20° of real physics, not windowing.
        assert!(pass_var < 30.0,
            "Composite lin main + subsonic: passband phase should be flat, got Δphase(1k→10k) = {:.3}° (p_lo={:.1}°, p_hi={:.1}°)",
            pass_var, p_lo, p_hi);

        // Infrasound: subsonic min-phase rotation must be present.
        let infra_max = b139_4a_max_abs_phase(&freq, &result.realized_phase, 5.0, 40.0);
        assert!(infra_max > 100.0,
            "Composite lin main + subsonic: infrasound should show min-phase rotation, got max |phase| = {:.3}°",
            infra_max);
    }

    /// Composite, min-phase main, no subsonic → reduces to MinimumPhase
    /// semantics: phase is the natural Gaussian min-phase rotation.
    #[test]
    fn fir_composite_min_main_no_subsonic() {
        let n = 512;
        let freq = b139_3_1_log_grid(n, 5.0, 40000.0);
        let target_mag = b139_3_gaussian_hp_mag(&freq, 632.0, 1.0, false);
        let cfg = b139_4a_composite_config(false, None);
        let result = generate_model_fir(&freq, &target_mag, &[], &vec![0.0; n], &cfg)
            .expect("generate_model_fir should succeed");
        // Around HP corner phase rotation must be measurable but bounded.
        let near_hp = b139_4a_phase_at(&freq, &result.realized_phase, 632.0);
        assert!(near_hp.abs() > 1.0,
            "Composite min main, no subsonic: phase near HP corner should rotate, got {:.3}°",
            near_hp);
    }

    /// Composite, min-phase main, subsonic ON → both Gaussian min-phase
    /// (around 632 Hz corner) AND subsonic min-phase (5–40 Hz) contribute.
    #[test]
    fn fir_composite_min_main_with_subsonic() {
        let n = 512;
        let freq = b139_3_1_log_grid(n, 5.0, 40000.0);
        let target_mag = b139_3_gaussian_hp_mag(&freq, 632.0, 1.0, true);
        let cfg = b139_4a_composite_config(false, Some(79.0));
        let result = generate_model_fir(&freq, &target_mag, &[], &vec![0.0; n], &cfg)
            .expect("generate_model_fir should succeed");
        let infra_max = b139_4a_max_abs_phase(&freq, &result.realized_phase, 5.0, 40.0);
        assert!(infra_max > 100.0,
            "Composite min main + subsonic: infrasound should rotate, got max |phase| = {:.3}°",
            infra_max);
        // Magnitude in passband must still match target after iterative_refine.
        let mut max_err = 0.0_f64;
        for (i, &f) in freq.iter().enumerate() {
            if f < 1000.0 || f > 10000.0 { continue; }
            let realised = result.realized_mag[i];
            let expected = target_mag[i] - result.norm_db;
            max_err = max_err.max((realised - expected).abs());
        }
        assert!(max_err < 0.5,
            "Composite min main + subsonic: passband magnitude drift {:.3} dB > 0.5 dB", max_err);
    }

    /// b139.5.3 — FIR design must run on a grid that extends below 20 Hz so
    /// the HP rolloff is realisable. Pre-fix the JS layer was passing the
    /// measurement grid (20 Hz – 20 kHz) and the Gaussian HP rolloff at 7 Hz
    /// was simply absent from the FIR. With the standalone grid 5 Hz – 40 kHz
    /// the realised magnitude at 7 Hz must be ≥ 30 dB below the passband.
    #[test]
    fn fir_wide_grid_realises_hp_rolloff_below_20hz() {
        let n = 512;
        let freq = b139_3_1_log_grid(n, 5.0, 40000.0);
        let target_mag = b139_3_gaussian_hp_mag(&freq, 1000.0, 1.0, false);
        let cfg = b139_4a_composite_config(true, None);
        let result = generate_model_fir(&freq, &target_mag, &[], &vec![0.0; n], &cfg)
            .expect("generate_model_fir should succeed");

        // Passband peak (~1 kHz – 10 kHz) should be at 0 dB after norm.
        let mut pb_max = f64::NEG_INFINITY;
        for (i, &f) in freq.iter().enumerate() {
            if f >= 1000.0 && f <= 10000.0 && result.realized_mag[i] > pb_max {
                pb_max = result.realized_mag[i];
            }
        }
        // 7 Hz is ~7 octaves below fc=1000 with a Gaussian shape — must be
        // strongly attenuated relative to the passband peak.
        let infra_at_7hz = {
            let mut idx = 0;
            let mut best = f64::INFINITY;
            for (i, &f) in freq.iter().enumerate() {
                let d = (f - 7.0).abs();
                if d < best { best = d; idx = i; }
            }
            result.realized_mag[idx]
        };
        let attenuation = pb_max - infra_at_7hz;
        assert!(attenuation > 30.0,
            "Wide-grid FIR must roll off HP below 20 Hz: passband peak {:.2} dB, 7 Hz {:.2} dB, attenuation {:.2} dB (need > 30)",
            pb_max, infra_at_7hz, attenuation);
    }


# Промт для Code: b140.0 — E2E test harness для экспорта

ТЗ целиком: `docs/TZ-b140-total-rebuild.md`.
Текущий билд: 0.1.0-b139.5.3 → bump до 0.1.0-b140.0.

## Главный принцип

Прежде чем что-либо менять в production — построить тесты которые
**bit-exact** проверяют итоговый экспорт против target curve.
Никаких изменений в production коде на этом этапе. Только тестовая
инфраструктура.

## Что нужно сделать

### 1. Synthetic fixtures (Rust)

`src-tauri/tests/fixtures.rs` (новый):

```rust
use phaseforge::io::Measurement;
use phaseforge::target::{TargetCurve, FilterConfig, FilterType};

/// Plain flat measurement: 0 dB, 0° phase, log freq grid.
pub fn flat_measurement(f_min: f64, f_max: f64, n: usize) -> Measurement {
    let freq: Vec<f64> = (0..n).map(|i| {
        f_min * (f_max / f_min).powf(i as f64 / (n - 1) as f64)
    }).collect();
    Measurement {
        name: "flat".into(),
        source_path: None,
        sample_rate: Some(48000.0),
        freq: freq.clone(),
        magnitude: vec![0.0; n],
        phase: Some(vec![0.0; n]),
        metadata: Default::default(),
    }
}

/// 8 канонических конфигураций для acceptance матрицы:
/// linear/min × subsonic on/off × peq none/3 bands
pub struct ExportConfig {
    pub name: &'static str,
    pub target: TargetCurve,
    pub linear_phase_main: bool,
    pub peq_bands: Vec<phaseforge::peq::PeqBand>,
}

pub fn acceptance_configs() -> Vec<ExportConfig> {
    // 8 configs: см. ТЗ b140.0 секция Tests
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
            peq_bands: sample_peq_bands(),  // 3 узких полосы для тестирования phase
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
        tilt_db_per_oct: 0.0,
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
        // shelves все zero
    }
}

fn sample_peq_bands() -> Vec<phaseforge::peq::PeqBand> {
    // 3 узких полосы с известными частотами и Q
    // на 200 Hz, 1 kHz, 5 kHz, gain ±3 dB, Q=2-4
    vec![/* ... */]
}
```

### 2. E2E export test (Rust integration)

`src-tauri/tests/e2e_export.rs` (новый):

```rust
mod fixtures;

use fixtures::*;

/// Воспроизводит full export pipeline которая используется в UI:
/// 1. evaluate_target_standalone (FIR grid 5–min(40k, sr/2*0.95))
/// 2. compute_peq_complex (если есть PEQ)
/// 3. generate_model_fir (Composite mode)
/// Результат: impulse response.
fn run_export_pipeline(
    cfg: &ExportConfig,
    sample_rate: f64,
) -> Vec<f64> {
    use phaseforge::target;
    use phaseforge::peq::compute_peq_complex;
    use phaseforge::fir::{generate_model_fir, FirConfig, PhaseMode, WindowType};

    let f_max = (40000.0_f64).min(sample_rate / 2.0 * 0.95);
    let n = 512;
    let freq = phaseforge::dsp::generate_log_freq_grid(n, 5.0, f_max);
    let target_response = target::evaluate(&cfg.target, &freq);

    let (peq_mag, peq_phase) = if !cfg.peq_bands.is_empty() {
        compute_peq_complex(&freq, &cfg.peq_bands, sample_rate)
    } else {
        (vec![0.0; n], vec![0.0; n])
    };

    let model_phase: Vec<f64> = target_response.phase.iter().zip(peq_phase.iter())
        .map(|(t, p)| t + p).collect();

    let subsonic_cutoff = if let Some(hp) = &cfg.target.high_pass {
        if hp.filter_type == FilterType::Gaussian
            && hp.subsonic_protect == Some(true)
            && hp.freq_hz > 40.0 {
            Some(hp.freq_hz / 8.0)
        } else { None }
    } else { None };

    let fir_cfg = FirConfig {
        taps: 65536,
        sample_rate,
        max_boost_db: 24.0,
        noise_floor_db: -150.0,
        window: WindowType::Blackman,
        phase_mode: PhaseMode::Composite,
        linear_phase_main: cfg.linear_phase_main,
        subsonic_cutoff_hz: subsonic_cutoff,
        iterations: 3,
        freq_weighting: true,
        narrowband_limit: true,
        nb_smoothing_oct: 0.333,
        nb_max_excess_db: 6.0,
    };

    generate_model_fir(&freq, &target_response.magnitude, &peq_mag, &model_phase, &fir_cfg).impulse
}

/// Ожидаемые magnitude и phase на FIR grid для сравнения.
fn expected_response(cfg: &ExportConfig, freq: &[f64]) -> (Vec<f64>, Vec<f64>) {
    // Target magnitude через evaluate
    let resp = phaseforge::target::evaluate(&cfg.target, freq);
    let mut mag = resp.magnitude;
    let mut phase = resp.phase;

    // Добавить PEQ contribution
    if !cfg.peq_bands.is_empty() {
        let (pm, pp) = phaseforge::peq::compute_peq_complex(freq, &cfg.peq_bands, 48000.0);
        for i in 0..mag.len() {
            mag[i] += pm[i];
            phase[i] += pp[i];
        }
    }

    // Phase reconstruction для linear_phase=false:
    // полная min-phase Hilbert (Gaussian + PEQ + subsonic).
    if !cfg.linear_phase_main {
        // ...
    } else {
        // linear_phase=true: только PEQ + subsonic min-phase
        // (main phase = 0)
        // ...
    }

    (mag, phase)
}

/// FFT impulse → magnitude/phase на FIR grid.
fn realized_response(impulse: &[f64], freq: &[f64], sample_rate: f64) -> (Vec<f64>, Vec<f64>) {
    use rustfft::{FftPlanner, num_complex::Complex};
    let n_fft = impulse.len();
    let mut buffer: Vec<Complex<f64>> = impulse.iter().map(|&x| Complex::new(x, 0.0)).collect();
    FftPlanner::<f64>::new().plan_fft_forward(n_fft).process(&mut buffer);

    let bin_hz = sample_rate / n_fft as f64;
    let mut mag = vec![0.0; freq.len()];
    let mut phase = vec![0.0; freq.len()];
    for (i, &f) in freq.iter().enumerate() {
        let bin = (f / bin_hz).round() as usize;
        if bin < n_fft / 2 {
            mag[i] = if buffer[bin].norm() > 1e-20 { 20.0 * buffer[bin].norm().log10() } else { -400.0 };
            phase[i] = buffer[bin].arg().to_degrees();
        }
    }
    (mag, phase)
}

#[test]
fn e2e_acceptance_matrix() {
    let configs = acceptance_configs();
    let sample_rate = 48000.0;
    let n = 512;
    let f_max = (40000.0_f64).min(sample_rate / 2.0 * 0.95);
    let freq = phaseforge::dsp::generate_log_freq_grid(n, 5.0, f_max);

    let mut report = String::new();
    let mut failures = 0;

    for cfg in &configs {
        let impulse = run_export_pipeline(cfg, sample_rate);
        let (real_mag, real_phase) = realized_response(&impulse, &freq, sample_rate);
        let (exp_mag, exp_phase) = expected_response(cfg, &freq);

        // Passband region: 1k–10k Hz (исключаем edge cases на rolloff)
        let passband: Vec<usize> = (0..freq.len())
            .filter(|&i| freq[i] >= 1000.0 && freq[i] <= 10000.0)
            .collect();

        let max_mag_err = passband.iter()
            .map(|&i| (real_mag[i] - exp_mag[i]).abs())
            .fold(0.0_f64, f64::max);
        let max_phase_err = passband.iter()
            .map(|&i| {
                let dp = real_phase[i] - exp_phase[i];
                let dp = ((dp + 180.0).rem_euclid(360.0)) - 180.0;  // wrap
                dp.abs()
            })
            .fold(0.0_f64, f64::max);

        let pass = max_mag_err < 0.5 && max_phase_err < 5.0;
        if !pass { failures += 1; }

        report.push_str(&format!(
            "{}: mag_err={:.3} dB, phase_err={:.3}° → {}\n",
            cfg.name, max_mag_err, max_phase_err,
            if pass { "PASS" } else { "FAIL" },
        ));
    }

    eprintln!("=== E2E Acceptance Matrix ===\n{}", report);

    // На текущем коде ожидается что часть FAIL — это baseline для b140.1.
    // Здесь мы НЕ assert!(failures == 0) — фиксируем reality.
    if failures > 0 {
        eprintln!("Baseline: {} of {} configs FAIL on current code (b139.5.3).", failures, configs.len());
        eprintln!("b140.1 should bring this to 0/{}.", configs.len());
    }
}
```

Замечания:
- Тест **выводит** через `eprintln!` reality каждой конфигурации.
- В b140.0 НЕ делаем `assert!(failures == 0)` — может быть baseline
  где часть FAIL. Просто фиксируем где FAIL и где PASS.
- В b140.1 после исправления Composite — все 8 должны PASS.

### 3. Snapshot регрессия для текущих PASS

Те конфигурации которые на b139.5.3 PASS — зафиксировать как
golden snapshot (импульс или его SHA-256 hash). Любое будущее
изменение которое сломает их — тест поймает.

### 4. Bump

- `src-tauri/tauri.conf.json` → `0.1.140` (numeric для MSI).
- `src-tauri/src/lib.rs` → b140.0.
- `src/lib/version.ts` → b140.0.
- skill `build-version`.

## Acceptance

1. `tests/fixtures.rs` и `tests/e2e_export.rs` созданы.
2. `cargo test --test e2e_export` запускается, выводит отчёт по 8
   конфигурациям (PASS/FAIL для каждой).
3. Golden hash для PASS-конфигураций зафиксирован.
4. Все existing 169+ cargo тестов остаются PASS.
5. vitest 143+ остаются PASS.
6. **Никаких изменений в production коде.**

## Что НЕ делать

- Не править Composite mode в Rust (это b140.1).
- Не трогать SUM (b140.2).
- Не править frontend (это всё инфраструктура для будущих этапов).

## Что прислать обратно

Отчёт E2E матрицы из `cargo test --test e2e_export -- --nocapture`:

```
linear_no_subsonic_no_peq: mag_err=0.000 dB, phase_err=0.000° → PASS/FAIL
linear_subsonic_no_peq:    mag_err=..., phase_err=... → PASS/FAIL
... (8 строк)

Baseline: X of 8 configs FAIL on b139.5.3.
```

Это и есть scope для b140.1.

## Правила

- Один коммит: `test: e2e export pipeline harness with 8-config matrix (b140.0)` + Co-Authored-By.
- 7-vector review.
- Без нарратива.

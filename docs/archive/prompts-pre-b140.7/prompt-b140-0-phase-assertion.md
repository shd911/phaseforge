# Промт для Code: b140.0 — расширение E2E теста на phase

**Тип:** доработка существующего теста. БЕЗ bump версии (всё ещё b140.0).
Если тест fail — это и есть baseline для b140.1, без bumping.

## Контекст

Текущий e2e тест (b140.0 commit 21b09c9) проверяет magnitude и rolloff,
но phase помечена как «informational» и не входит в acceptance.
Это дыра — изначальный bug-report пользователя был именно про phase
(«экспорт игнорирует фазу, ровная линия»). Magnitude после b139.5.3
работал, но phase могла быть сломана независимо.

Нужно расширить тест на assertive phase comparison и узнать реальное
состояние.

## Что нужно сделать

### 1. Expected phase для каждой конфигурации

В `src-tauri/tests/fixtures.rs` или `e2e_export.rs` добавить функцию
вычисления **ожидаемой** phase:

```rust
/// Expected phase response для каждой конфигурации.
/// linear_phase_main определяет получает ли main filter min-phase
/// reconstruction или остаётся 0.
/// PEQ phase и subsonic phase — всегда min-phase (Hilbert от их
/// magnitude отдельно).
fn expected_phase(cfg: &ExportConfig, freq: &[f64]) -> Vec<f64> {
    let n = freq.len();
    let n_fft = 65536;  // используется в FIR
    let mut phase = vec![0.0; n];

    // Main filter phase
    if !cfg.linear_phase_main {
        // Вычислить main_mag (target без subsonic) и взять Hilbert
        let main_mag = main_filter_magnitude(&cfg.target, freq);
        let main_phase = phaseforge::dsp::minimum_phase_from_magnitude(&main_mag, n_fft);
        // Интерполировать на freq grid если нужно
        for i in 0..n { phase[i] += main_phase[i]; }
    }

    // PEQ phase — всегда min-phase (по физике биквадов)
    if !cfg.peq_bands.is_empty() {
        let (_, peq_phase) = phaseforge::peq::compute_peq_complex(freq, &cfg.peq_bands, 48000.0);
        for i in 0..n { phase[i] += peq_phase[i]; }
    }

    // Subsonic phase — всегда min-phase
    if let Some(hp) = &cfg.target.high_pass {
        if hp.subsonic_protect == Some(true) && hp.freq_hz > 40.0 {
            let f_sub = hp.freq_hz / 8.0;
            let subsonic_mag: Vec<f64> = freq.iter().map(|&f| {
                if f <= 0.0 { return -400.0; }
                let r = (f_sub / f).powi(16);
                let lin = (1.0 / (1.0 + r)).sqrt();
                if lin > 1e-20 { 20.0 * lin.log10() } else { -400.0 }
            }).collect();
            let subsonic_phase = phaseforge::dsp::minimum_phase_from_magnitude(&subsonic_mag, n_fft);
            for i in 0..n { phase[i] += subsonic_phase[i]; }
        }
    }

    phase
}

/// Magnitude основного фильтра без PEQ и без subsonic.
fn main_filter_magnitude(target: &TargetCurve, freq: &[f64]) -> Vec<f64> {
    // Tilt + shelves + HP + LP без subsonic application.
    // Если есть hp с subsonic_protect — временно его выключить для расчёта main.
    let mut tc = target.clone();
    if let Some(hp) = &mut tc.high_pass {
        hp.subsonic_protect = Some(false);
    }
    phaseforge::target::evaluate(&tc, freq).magnitude
}
```

### 2. Realized phase с компенсацией linear delay

Для линейно-фазового импульса peak находится в центре (N/2),
что даёт линейный ramp в phase = `-2π × f × N/2 / sr`. Это надо
вычесть до сравнения с expected.

```rust
fn realized_phase_compensated(
    impulse: &[f64],
    freq: &[f64],
    sample_rate: f64,
    linear_phase_main: bool,
) -> Vec<f64> {
    let n_fft = impulse.len();
    let raw_phase = realized_response(impulse, freq, sample_rate).1;

    if linear_phase_main {
        // Subtract linear delay = N/2 samples.
        let delay_sec = (n_fft as f64 / 2.0) / sample_rate;
        raw_phase.iter().zip(freq.iter()).map(|(p, &f)| {
            let linear_delay_phase = -360.0 * f * delay_sec;
            wrap_deg(p - linear_delay_phase)
        }).collect()
    } else {
        // Min-phase impulse peak at 0 — no linear delay to subtract.
        raw_phase
    }
}

fn wrap_deg(p: f64) -> f64 {
    ((p + 180.0).rem_euclid(360.0)) - 180.0
}
```

### 3. Phase assertion в acceptance

Заменить informational phase_var на assertive comparison:

```rust
let expected_ph = expected_phase(cfg, &freq);
let realized_ph = realized_phase_compensated(&impulse, &freq, sample_rate, cfg.linear_phase_main);

let max_phase_err = passband.iter()
    .map(|&i| {
        let dp = wrap_deg(realized_ph[i] - expected_ph[i]);
        dp.abs()
    })
    .fold(0.0_f64, f64::max);

let pass = max_mag_err < 0.5 && max_phase_err < 5.0;
```

Допуск 5° — типичный для FIR с 65k taps. Если реальная ошибка
больше — это ловит баг.

### 4. Обновить отчёт

```
config                              mag_err   phase_err  rolloff   verdict
-----------------------------------------------------------------------
linear_no_subsonic_no_peq          0.002 dB   X.XX°    82.37 dB   PASS/FAIL
linear_subsonic_no_peq             0.002 dB   X.XX°   150.14 dB   PASS/FAIL
... (8 строк)
```

Если хотя бы одна строка FAIL — это baseline для b140.1.

### 5. Если все 8 PASS

Если магнитуда И фаза проходят все 8 конфигураций — это значит
экспорт корректно работает. Тогда баг который наблюдал пользователь
(«ровная фаза в WAV») — где-то в **другом месте**:
- Возможно UI рендер импортированного WAV.
- Возможно Import обратно теряет phase.
- Нужна отдельная диагностика конкретного workflow пользователя.

### 6. БЕЗ bump версии

Если тест PASS — bump до b140.0.1 опционально, или просто amend
существующий b140.0 commit.

Если FAIL — оставить b140.0 как baseline, b140.1 будет фиксить.

## Что прислать обратно

Полный отчёт E2E:

```
config                              mag_err   phase_err  rolloff   verdict
linear_no_subsonic_no_peq          ...
linear_subsonic_no_peq             ...
min_no_subsonic_no_peq             ...
min_subsonic_no_peq                ...
linear_no_subsonic_with_peq        ...
linear_subsonic_with_peq           ...
min_no_subsonic_with_peq           ...
min_subsonic_with_peq              ...

Phase acceptance: X of 8 PASS, Y FAIL.
```

## Правила

- Без bump (тест-инфраструктура).
- Если все PASS — amend commit `test: e2e ... (b140.0)` с phase
  assertion.
- Если FAIL — отдельный коммит `test: phase assertion in e2e (baseline for b140.1)` + Co-Authored-By, версия не меняется.
- 7-vector review.
- Без нарратива.

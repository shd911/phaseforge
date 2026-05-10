# Промт для Code: b139.3.3 — точное воспроизведение divergence

**Тип:** расширение cargo теста до точного воспроизведения. Без bump
если тест в итоге PASS; bump до b139.3.3 если FAIL и нужен фикс.

## Контекст

В b139.3.2 cargo тест `iterative_refine_converges_with_min_phase_subsonic`
прошёл. Но Кирилл реально наблюдал divergence в logs:

```
phase_mode=MinimumPhase, has_peq=false, max_phase_abs=0.00°
iter=1, max_err=0.151 dB
iter=2, max_err=12.091 dB
iter=3, max_err=13.486 dB
final realized_max=0.00 dB
```

Значит cargo тест **не точно** воспроизводит реальный workflow.
Различие есть в каком-то параметре.

## Извлечённые параметры из реальных логов и UI

```
HP: Gaussian, freq_hz=632.0, shape=1.0, linear_phase=true, subsonic_protect=true
LP: None
Target: Preset=Flat, level=0, tilt=0
Sample rate: 48000 Hz
Taps: 65536
Window: Blackman
phase_mode: MinimumPhase (после демотации из-за subsonic)
iterations: 3
freq_weighting: true
narrowband_limit: true (default UI)
nb_smoothing_oct: 0.333 (default)
nb_max_excess_db: 6.0 (default)
max_boost_db: 24.0 (default)
noise_floor_db: -150.0 (default)

Freq grid: evaluate_target_standalone default → 20.0 to 20000.0 Hz, n=512 точек

target_phase передаётся в Rust как vec![0.0; 512] (max_phase_abs=0.00°)
```

## Что нужно сделать

### 1. Обновить `iterative_refine_converges_with_min_phase_subsonic`

Использовать **точные** параметры выше. Особое внимание:

```rust
// Freq grid через ту же функцию что используется в evaluate_target_standalone
let freq: Vec<f64> = crate::dsp::generate_log_freq_grid(512, 20.0, 20000.0);
// ИЛИ проверить какую функцию использует evaluate_target_standalone и
// использовать ту же.

// Target через target::evaluate с конкретным TargetCurve
let target = TargetCurve {
    reference_level_db: 0.0,
    tilt_db_per_oct: 0.0,
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
    // shelves — все zeros
};
let response = target::evaluate(&target, &freq);
let target_mag = response.magnitude;
let target_phase = vec![0.0; 512];  // Rust в MinimumPhase сам Hilbert делает

let cfg = FirConfig {
    taps: 65536,
    sample_rate: 48000.0,
    max_boost_db: 24.0,
    noise_floor_db: -150.0,
    window: WindowType::Blackman,
    phase_mode: PhaseMode::MinimumPhase,
    iterations: 3,
    freq_weighting: true,
    narrowband_limit: true,    // ← возможно отличается от cargo b139.3.2
    nb_smoothing_oct: 0.333,
    nb_max_excess_db: 6.0,
};

let peq_mag = vec![0.0; 512];
```

### 2. Извлечь per-iteration errors

Если `iterative_refine` (или какая функция вызывается из
`generate_model_fir`) пишет per-iter errors через `tracing::info!`,
добавить `tracing-subscriber` capture в тест чтобы перехватить эти
строки. Иначе — добавить test-only API:

```rust
#[cfg(test)]
pub fn generate_model_fir_with_history(
    /* args */
) -> (FirResult, Vec<IterStats>);

#[cfg(test)]
pub struct IterStats {
    pub iter: usize,
    pub max_err_db: f64,
    pub rms_err_db: f64,
}
```

В тесте:

```rust
let (result, history) = generate_model_fir_with_history(&freq, &target_mag, &peq_mag, &target_phase, &cfg);

eprintln!("iterative_refine history:");
for stats in &history {
    eprintln!("  iter={} max_err={:.3} dB rms={:.3} dB", stats.iter, stats.max_err_db, stats.rms_err_db);
}

// Проверка сходимости: errors не должны расти.
let mut prev = f64::INFINITY;
for stats in &history {
    if stats.iter == 1 {
        // первая итерация устанавливает baseline
        prev = stats.max_err_db;
    } else {
        assert!(stats.max_err_db <= prev * 1.5,
            "divergence at iter {}: max_err {:.3} > 1.5 × previous {:.3}",
            stats.iter, stats.max_err_db, prev);
        prev = stats.max_err_db;
    }
}

assert!(history.last().unwrap().max_err_db < 1.0,
    "iterative_refine не сошёлся: финальный max_err {:.3} dB > 1 dB",
    history.last().unwrap().max_err_db);
```

### 3. Если тест **PASS** — расширить покрытие

Если даже с точными параметрами тест PASS, попробовать:

a) **`narrowband_limit=true` vs `narrowband_limit=false`** — два отдельных тест-кейса.

b) **realised_mag сравнение через FFT** (как в b139.3.2) — может баг
   виден через realized response, не через iter errors.

c) **freq grid через `dsp::generate_log_freq_grid`** vs ручной log_grid —
   функция может иметь точный startend specifier который влияет.

d) **Параметр `freq_weighting`** — true vs false.

Каждый вариант — отдельный test case. Один из них должен поймать
divergence.

### 4. Если тест **FAIL** — отчитаться

Output сохранить и приложить:

```
Reproduced! iterative_refine history:
  iter=1 max_err=0.151 dB rms=0.002 dB
  iter=2 max_err=12.091 dB rms=0.261 dB
  iter=3 max_err=13.486 dB rms=0.315 dB
```

Это локализация. Затем — отдельный промт b139.3.4 с фиксом.

### 5. Этот промт НЕ фиксит баг

Цель — repro. Фикс будет следующим этапом после получения failing
test.

## Acceptance

1. Cargo тест расширен с точными параметрами реального workflow.
2. Доступ к per-iter errors есть (через tracing capture или test API).
3. Прогон `cargo test` — отчитаться:
   - **FAIL с numeric diff** соответствующим Кирилле логам → repro
     получен → дальше идёт b139.3.4 фикс.
   - **PASS** → запустить дополнительные варианты (4 sub-кейса выше)
     и отчитаться какой из них наконец FAIL'нул, или ни один не
     поймал.
4. Существующие 158 cargo тестов остаются зелёными.
5. vitest 136 остаются зелёными.

## Что НЕ делать

- Не фиксить `iterative_refine` без failing repro теста.
- Не предлагать гипотезы — собирать данные.
- Не делать bump если тест PASS (нет существенных изменений).

## Что прислать обратно

```
cargo test (после расширения):
  iterative_refine_converges_with_min_phase_subsonic: PASS / FAIL
    if FAIL — eprintln output:
      iter=1 max_err=... rms=...
      iter=2 max_err=... rms=...
      iter=3 max_err=... rms=...
    if PASS — список subcases которые проверены, какой из них поймал/не поймал

  все existing: PASS count
```

## Правила (CLAUDE.md)

- Если тест PASS на всех вариантах — отчитаться, **не коммитить**
  (нет смысла комитить тест который ничего не ловит). Кирилл должен
  предоставить дополнительный workflow который ловит.
- Если тест FAIL — коммит:
  `test: cargo reproduces iterative_refine divergence (b139.3.3)`.
- 7-vector review (если коммит).
- Без нарратива прогресса.

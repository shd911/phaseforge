# Промт для Code: b139.3.2 — divergence iterative_refine с subsonic

**Тип:** диагностический cargo тест + удаление console.log из b139.3.1.
Bump до 0.1.0-b139.3.2.

## Контекст и evidence

В реальном workflow Кирилла Rust логи показали divergence:

```
phase_mode=MinimumPhase, has_peq=false, max_phase_abs=0.00°
iterative_refine: iter=1, max_err=0.151 dB, rms_err=0.002 dB
iterative_refine: iter=2, max_err=12.091 dB, rms_err=0.261 dB   ← divergence
iterative_refine: iter=3, max_err=13.486 dB, rms_err=0.315 dB
```

То же место без subsonic сходится за 1 итерацию (max_err=0.036 dB).

Гипотеза: `iterative_refine` (в `src-tauri/src/fir/helpers.rs` или
аналог) не пересчитывает target_phase после adjustment magnitude.
В MinimumPhase mode это приводит к нарастающему phase mismatch на
каждой итерации.

Автотесты b139.3.1 проверили single-pass output (`generate_model_fir`
один раз). Iterative path не покрыт.

## Что нужно сделать

### 1. Удалить диагностический console.log из b139.3.1

В `src/lib/band-evaluator.ts` найти и удалить
`console.log("[BandEval:FIR] ...")` который добавлен временно в b139.3.1.

### 2. Cargo тест на сходимость iterative_refine

В `src-tauri/src/fir/mod.rs` (или helpers.rs где живёт iterative_refine)
добавить:

```rust
#[test]
fn iterative_refine_converges_with_min_phase_target() {
    // Воспроизводим Кирилла сценарий: Gaussian HP=632, linear_phase=true,
    // subsonic ON → демотируется в MinimumPhase mode внутри generate_model_fir.
    // Без PEQ.
    let n_target = 512;
    let freq: Vec<f64> = log_grid(n_target, 5.0, 40000.0);

    // Target magnitude: Gaussian HP=632 + subsonic Butterworth-8 на 79 Hz
    let target_mag: Vec<f64> = freq.iter().map(|&f| {
        let gauss = if f >= 632.0 { 0.0 } else {
            // Gaussian HP magnitude
            let lp_lin = (-LN_2 * (f / 632.0_f64).powi(2)).exp();
            let hp_lin = 1.0 - lp_lin;
            if hp_lin > 1e-20 { 20.0 * hp_lin.log10() } else { -400.0 }
        };
        let f_sub = 79.0;
        let ratio = (f_sub / f).powi(16);
        let sub_lin = (1.0 / (1.0 + ratio)).sqrt();
        let sub_db = if sub_lin > 1e-20 { 20.0 * sub_lin.log10() } else { -400.0 };
        gauss + sub_db
    }).collect();

    // Target phase: zeros (Rust сам делает Hilbert в MinimumPhase mode).
    let target_phase: Vec<f64> = vec![0.0; n_target];
    let peq_mag: Vec<f64> = vec![0.0; n_target];

    let cfg = FirConfig {
        taps: 65536, sample_rate: 48000.0,
        max_boost_db: 24.0, noise_floor_db: -150.0,
        window: WindowType::Blackman,
        phase_mode: PhaseMode::MinimumPhase,
        iterations: 3, freq_weighting: true,
        narrowband_limit: false,
        nb_smoothing_oct: 0.333,
        nb_max_excess_db: 6.0,
    };
    let result = generate_model_fir(&freq, &target_mag, &peq_mag, &target_phase, &cfg);

    // Проверка сходимости: error ДОЛЖЕН монотонно НЕ расти.
    // Если есть API чтобы получить per-iter errors — использовать его.
    // Иначе — реализовать вспомогательную версию которая возвращает
    // historty errors через Vec<f64>.

    // Минимальная проверка через result: realized response должна быть
    // близка к target.
    let realized_mag = compute_realized_mag(&result.impulse, &freq, cfg.sample_rate);
    let max_err: f64 = realized_mag.iter().zip(target_mag.iter())
        .map(|(r, t)| (r - t).abs())
        .fold(0.0_f64, f64::max);

    eprintln!("iterative_refine_converges: max_err = {:.3} dB", max_err);
    assert!(max_err < 1.0,
        "iterative_refine не сошёлся: max_err {:.3} dB > 1 dB. \
         Это и есть divergence bug который Кирилл наблюдал в b139.3.",
        max_err);
}

fn compute_realized_mag(impulse: &[f64], freq: &[f64], sr: f64) -> Vec<f64> {
    // FFT impulse → magnitude → interpolate на freq grid → 20*log10
    // Если есть утилита — использовать. Иначе минимально:
    use rustfft::{FftPlanner, num_complex::Complex};
    let n_fft = impulse.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut buffer: Vec<Complex<f64>> = impulse.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft.process(&mut buffer);

    // Linear FFT freq → log freq mapping
    let bin_hz = sr / n_fft as f64;
    freq.iter().map(|&f| {
        let bin = (f / bin_hz).round() as usize;
        if bin >= n_fft / 2 { return -400.0; }
        let mag_lin = buffer[bin].norm();
        if mag_lin > 1e-20 { 20.0 * mag_lin.log10() } else { -400.0 }
    }).collect()
}
```

Если функция `compute_realized_mag` уже есть в проекте — использовать её,
не дублировать. Если `rustfft` уже подключён в Cargo.toml — использовать.

### 3. Вывести per-iteration errors через test helper

Если в `iterative_refine` есть способ получить history errors (Vec<f64>
с max_err каждой iter) — использовать. Если нет — добавить **test-only**
версию `iterative_refine_with_history` которая возвращает errors каждой
итерации, и использовать её в тесте.

```rust
#[cfg(test)]
pub fn iterative_refine_with_history(/* params */) -> (Vec<f64>, Vec<f64>) {
    // returns (final_impulse, max_err_per_iter)
}
```

Это даёт diagnostic output: какая именно итерация divergent, насколько.

В тесте:

```rust
let (impulse, errors) = iterative_refine_with_history(...);
eprintln!("iter errors: {:?}", errors);
// errors[0] — после iter=1
// errors[1] — после iter=2
// и т.д.
assert!(errors.iter().enumerate().all(|(i, &e)| {
    if i == 0 { true } else { e <= errors[0] * 1.5 } // не должны расти
}), "divergence: errors per iter = {:?}", errors);
```

### 4. Этот этап НЕ фиксит баг

Цель — **зафиксировать divergence через cargo тест**. После этого
тест либо pass (всё нормально, баг был временный) либо fail (баг
воспроизводится → следующий этап b139.3.3 фиксит конкретно
`iterative_refine`).

### 5. Bump версии

- `src-tauri/tauri.conf.json` — version + productName/title до b139.3.2.
- `src-tauri/src/lib.rs` — startup-лог.
- После билда — skill `build-version`.

## Acceptance

1. Console.log `[BandEval:FIR]` удалён из `band-evaluator.ts`.
2. Cargo тест `iterative_refine_converges_with_min_phase_target`
   добавлен.
3. После прогона `cargo test` отчитаться:
   - **Если PASS** — divergence не воспроизводится в этих условиях.
     Тогда нужно расширить тест ближе к Кирилле сценарию (возможно
     subsonic_protect=Some(true) явно через target curve, не
     manually-crafted target_mag).
   - **Если FAIL** — приложить вывод `eprintln!` с per-iter errors.
     Это и есть локализация бага.
4. Все 154 + 3 предыдущих cargo тестов остаются зелёными.
5. Vitest 136 тестов остаются зелёными.

## Что прислать обратно

```
cargo test:
  iterative_refine_converges_with_min_phase_target: PASS / FAIL
  if FAIL — eprintln output:
    iter errors: [..., ..., ...]
    final max_err: ... dB
  (existing 157 tests): PASS count

Если PASS — расширили тест и всё ещё PASS, или нужны другие условия?
Если FAIL — какие конкретно числа.
```

## Что НЕ делать

- Не фиксить `iterative_refine` в этом этапе. Только тест.
- Не менять frontend.
- Не делать full UI прогон до получения cargo результата.

## Тестировать на `.dmg`

Не обязательно. Этап тестовой инфраструктуры. Если хочется
подтвердить что console.log убран — собрать `.dmg b139.3.2` и
проверить что DevTools console чистая при импорте.

## Правила (CLAUDE.md)

- Один коммит: `test: regression test for iterative_refine divergence with subsonic (b139.3.2)`
  + Co-Authored-By.
- 7-vector review.
- Без нарратива прогресса.

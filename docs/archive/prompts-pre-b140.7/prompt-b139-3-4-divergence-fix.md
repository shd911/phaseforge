# Промт для Code: b139.3.4 — фикс iterative_refine divergence

**Тип:** Rust фикс. Bump до 0.1.0-b139.3.4.

## Контекст

В b139.3.3 cargo тест зафиксировал failing repro:

```
iter=1 max_err=0.151 dB rms=0.002 dB
iter=2 max_err=12.091 dB rms=0.261 dB    ← divergence
iter=3 max_err=13.486 dB rms=0.315 dB
```

Гипотеза (от b139.3.3 recap, физически правильная):

> iterative_refine в MinimumPhase mode пересобирает спектр из
> refined_db с фиксированной phase_rad (assemble_complex_spectrum
> line ~140), которая больше не соответствует corrected magnitude.
> На следующей FFT round-trip это даёт расходящийся error.

Phase должна быть пересчитана через Hilbert каждый раз когда
magnitude меняется. Иначе complex spectrum (mag, phase) пара
становится inconsistent → IFFT даёт impulse не реализующий ни одно
ни другое.

## Что нужно сделать

### 1. Audit `src-tauri/src/fir/helpers.rs:iterative_refine`

Прочитать функцию целиком. Понять:
- Где формируется `phase_rad` (вероятно через `minimum_phase_from_magnitude`)
- Где `assemble_complex_spectrum(refined_db, phase_rad)` вызывается (~line 140)
- Когда `refined_db` обновляется на каждой итерации
- Используется ли тот же `phase_rad` повторно на каждой iter

Также проверить путь LinearPhase — там phase обычно zeros или
linear, не зависит от magnitude → не должна пересчитываться. Только
MinimumPhase нуждается в пересчёте.

### 2. Применить фикс

Внутри iterative_refine loop в MinimumPhase mode:

```rust
// псевдокод
for iter in 0..config.iterations {
    let refined_db = target_db - error_db;

    // ФИКС: пересчитать phase для текущего refined_db в MinimumPhase mode
    let phase_rad = match config.phase_mode {
        PhaseMode::MinimumPhase => minimum_phase_from_magnitude(&refined_db, n_fft),
        PhaseMode::LinearPhase => zero_phase(n_fft),  // или существующая логика
        PhaseMode::Hybrid => /* существующая логика */,
    };

    let spectrum = assemble_complex_spectrum(&refined_db, &phase_rad);
    let impulse = ifft(&spectrum);
    let realized = compute_realized_response(&impulse);
    let error_db = realized - target_db;

    // capture stats для тестов через ITER_STATS
    push_iter_stat(iter, max_err, rms_err);

    if max_err < CONVERGENCE_THRESHOLD {
        break;
    }
}
```

Точная реализация зависит от существующего кода — следовать ему,
просто переместить `phase_rad` calculation **внутрь loop** для
MinimumPhase mode (если сейчас оно снаружи loop). Для LinearPhase —
оставить как было (расчёт один раз снаружи).

### 3. Проверить тесты

Цель: failing repro test из b139.3.3 должен **PASS** после фикса.

```
iterative_refine_converges_with_min_phase_subsonic: PASS
```

Errors должны монотонно убывать или остаться маленькими:

```
iter=1 max_err=0.151 dB
iter=2 max_err=<= 0.151 dB
iter=3 max_err=<= 0.151 dB (или converged)
```

### 4. Регрессия

Все 158 cargo тестов должны остаться **PASS**. Особенно:

- `generate_fir_b139_golden_lr4_baseline_impulse_hash` — hash
  `3a56a4dab45f0fb1` не должен измениться (LR4 baseline без subsonic
  не задействует MinimumPhase Hilbert recompute).
- `fir_identity_for_flat_input_no_filters` — flat input не должен
  сломаться.
- `fir_identity_with_min_phase_mode` — это test использует
  MinimumPhase mode, но с flat target. Hilbert от flat = zeros, что
  даёт identity FIR. Должен оставаться identity.
- `generate_fir_b139_3_*` — три b139.3 теста.
- `fir_linear_gaussian_with_subsonic_keeps_passband_intact` — должен
  стать ещё точнее после фикса.

Если **golden hash изменился** — это сильный знак что фикс затронул
не только divergence path. Diagnostic, не пропускать.

vitest 136 тестов — без изменений (Rust фикс не затрагивает frontend).

### 5. Bump версии

- `src-tauri/tauri.conf.json` — version + productName/title до b139.3.4.
- `src-tauri/src/lib.rs` — startup-лог.
- После билда — skill `build-version`.

## Acceptance

1. Прочитан `iterative_refine` в `helpers.rs`, понята точка
   фиксированной phase_rad.
2. Phase recompute добавлен в loop для MinimumPhase mode.
3. `iterative_refine_converges_with_min_phase_subsonic` теперь PASS.
4. Все existing 157+ cargo тестов PASS, golden hash не изменился.
5. vitest 136 PASS.
6. **Manual sanity на `.dmg b139.3.4`:** импорт flat measurement,
   default target + Gaussian HP=632 + subsonic ON, Export FIR. В Rust
   логах:
   ```
   iter=1 max_err=<малое значение>
   iter=2 max_err=<<= iter 1 или converged>
   iter=3 max_err=<<= iter 2 или converged>
   ```
   Никаких 12 dB скачков.

## Что НЕ делать

- Не менять API `iterative_refine` — только internal logic.
- Не менять LinearPhase path.
- Не менять Hybrid path (если есть).
- Не трогать frontend.
- Не трогать `evaluate_target_standalone` или другие команды.
- Не пытаться решить отдельный вопрос «hybrid mode для linear
  Gaussian + min-phase subsonic» — это другая задача, после фикса
  divergence.

## Что прислать обратно

```
cargo test:
  iterative_refine_converges_with_min_phase_subsonic: PASS
    iter history: iter=1 ..., iter=2 ..., iter=3 ...
  generate_fir_b139_golden_lr4_baseline_impulse_hash: PASS (hash 3a56a4dab45f0fb1)
  все остальные 156+ тестов: PASS

vitest: 136 PASS

Manual .dmg b139.3.4 sanity: <iter errors из логов>
```

## Правила (CLAUDE.md)

- Один коммит: `fix: recompute min-phase per iteration in iterative_refine (b139.3.4)`
  + Co-Authored-By.
- 7-vector review.
- Без нарратива прогресса.
- `cargo tauri build` для финальной сборки.

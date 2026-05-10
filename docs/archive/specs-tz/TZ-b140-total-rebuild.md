# ТЗ: Тотальный ребилд b140 — полная унификация pipeline

**Цель:** одна точка истины для всех расчётов отображения и экспорта.
Добавление любой новой функции = одна правка в одном месте. Никаких
разрывов между тем что показывает SPL и что попадает в экспортируемый
WAV.

**Триггер:** каскад b139.x показал что инкрементальная унификация не
работает — каждое незавершённое звено становится источником регрессий.
Composite режим в Rust теряет PEQ phase. SUM до сих пор на параллельных
расчётах. Время закрыть всё.

---

## Карта целевого pipeline

```
                    ┌──────────────────┐
                    │  band-evaluator  │  ← единственная точка
                    │  (frontend)      │     истины
                    └─────────┬────────┘
                              │
        ┌─────────────────────┼──────────────────────┐
        │                     │                      │
        ▼                     ▼                      ▼
   ┌──────────┐         ┌──────────┐          ┌──────────┐
   │   SPL    │         │  IR/Step │          │  Export  │
   │   /GD    │         │  / SUM   │          │  WAV     │
   └──────────┘         └──────────┘          └──────────┘
   читает result        читает result          читает result.fir
   .targetPhase         .ir.{meas,             — bit-exact
   и т.д.               target,corrected}      реализация target
```

Все вкладки и экспорт получают данные из единого результата
`evaluateBandFull` (одна полоса) или `evaluateSum` (несколько полос).
Никаких inline `invoke` в `FrequencyPlot.tsx`.

---

## Что неправильно сейчас (после b139.5.3)

1. **Composite режим в Rust** реконструирует фазу самостоятельно,
   разделяя total magnitude на base + subsonic через вычитание.
   Это работает для Gaussian + subsonic, но **теряет PEQ phase** когда
   `linear_phase_main=true` (PEQ оказывается в base, а base = 0 для
   linear).

2. **SUM aggregation** (~1000 строк inline в FrequencyPlot) делает
   свои `invoke evaluate_target / compute_peq_complex / compute_cross_section`
   — параллельно с unified pipeline.

3. **`filterEquals` в peq-optimize** не сравнивает `linear_phase` и
   `subsonic_protect` — оранжевый банер «PEQ устарел» (b136) не
   срабатывает на эти изменения.

4. **Нет E2E тестов фактического экспорта.** Все тесты — на промежуточных
   шагах. Bit-exact проверки итогового FIR не было. Поэтому регрессии
   ловились только пользователем.

---

## Целевая архитектура

### Frontend: `band-evaluator.ts`

`evaluateBandFull(band, options)` возвращает всё необходимое.
Расчёты делаются на **двух** freq grid:
- **Display grid** = freq измерения (для SPL/IR/Step/GD).
- **Export grid** = standalone 5–min(40k, sr/2·0.95) Гц, 512 точек
  (для FIR).

Result содержит данные с обоих grid в соответствующих полях.

### Frontend: `evaluateSum`

Aggregates `evaluateBandFull` каждой полосы. Применяет polarity и
alignment delays per-band. Возвращает суммарные mag/phase/IR.
Используется в SUM view.

### Rust: разделённый Composite режим

`generate_model_fir` принимает **три** магнитуды отдельно:
```rust
target_mag       — main filter (Gaussian/LR/Butterworth + LP + tilt + shelves)
peq_mag          — PEQ correction (всегда min-phase по физике)
subsonic_mag     — Butterworth-8 защитный (всегда min-phase)
```

Phase reconstruction:
```rust
main_phase     = if linear_phase_main { 0 } else { Hilbert(target_mag) }
peq_phase      = Hilbert(peq_mag)
subsonic_phase = Hilbert(subsonic_mag)
total_phase    = main_phase + peq_phase + subsonic_phase
total_mag      = target_mag + peq_mag + subsonic_mag
```

Это разрывает связь «PEQ pхase теряется при линейном Gaussian».

### Rust: единая точка истины для phase reconstruction

Текущая `composite_phase_inner` остаётся, но получает **разделённые**
магнитуды от frontend. Нет вычитания в Rust.

---

## Тестовая стратегия (приоритет — автоматизация)

### Уровень 1: E2E экспорт (новый, главный)

Cargo integration test который воспроизводит **полный export
pipeline**:

```rust
#[test]
fn e2e_flat_input_yields_target_curve() {
    // 1. Synthetic flat measurement (0 dB всё, phase=0)
    let measurement = make_flat_measurement(20.0, 20000.0, 484);

    // 2. Target: Gaussian HP=632, M=1, linear_phase=true, subsonic ON
    let target = make_target_with_gaussian_hp(632.0, true, true);

    // 3. Прогнать через тот же код что использует UI Export:
    //    evaluate_target_standalone (FIR grid) → compute_peq_complex →
    //    generate_model_fir (Composite mode)
    let impulse = run_export_pipeline(&measurement, &target);

    // 4. Bit-exact сравнение реализованного FIR с target:
    //    FFT(impulse) → magnitude и phase
    //    На FIR grid сравнить с target_mag и target_phase
    let realized = fft_response(&impulse, sample_rate);
    assert_curves_match(&realized, &target_response, tol_db=0.5, tol_phase=2.0);
}
```

8 такие тесты для матрицы конфигураций:
- linear/min × subsonic on/off × PEQ none/3 полосы

Каждый — самодостаточный, никакого UI, никаких ручных проверок.

### Уровень 2: Snapshot регрессия

Vitest snapshot тесты на `evaluateBandFull` и `evaluateSum`. Снапшоты
зафиксированы — любое изменение output поднимает diff.

### Уровень 3: Golden hash

SHA-256 от FFT magnitude итогового FIR для baseline (LR4 HP=80 без
PEQ без subsonic). Не должен меняться между этапами refactor.

### Уровень 4: Subagent верификация после каждого этапа

После каждого этапа b140.X — subagent запускает:
```
cargo test
npm test
```
И отчитывается о PASS/FAIL по всем 4 уровням. Если хоть один уровень
fail — этап не считается завершённым.

---

## План этапов

### b140.0: E2E test harness (предварительно)

**Цель:** прежде чем что-либо менять — построить инфраструктуру
которая ловит регрессии автоматически.

- Synthetic fixtures: flat measurement, 6 target конфигураций.
- Rust integration test `tests/e2e_export.rs` который воспроизводит
  full export pipeline (evaluate → peq → fir → IFFT → impulse).
- Bit-exact comparison helpers (FFT, magnitude, phase).
- 8 acceptance configurations (4 phase modes × subsonic on/off).
  На текущем коде они **могут не все pass** — это OK, фиксируем
  baseline где они FAIL и где PASS.

**Acceptance:**
- Tests добавлены, прогон даёт ясный отчёт «X pass / Y fail».
- Сохранён golden output для случаев где сейчас PASS.
- Никаких изменений в production коде.

### b140.1: Разделённый Composite в Rust

**Цель:** убрать вычитание subsonic из target в Rust. Принимать
3 магнитуды раздельно.

- Расширить `FirConfig` или сигнатуру `generate_model_fir`:
  принимать `peq_mag: Vec<f64>` и `subsonic_mag: Option<Vec<f64>>`
  отдельно от `target_mag`.
- `composite_phase_inner` получает 3 части, не делает вычитание.
- Frontend `band-evaluator.ts` передаёт 3 части (он уже знает их
  все).
- E2E test pipeline после фикса: 8/8 acceptance configurations PASS.

**Acceptance:**
- 8 E2E тестов всё PASS (включая ранее FAIL для PEQ + linear).
- Golden hash baseline (LR4) не изменился.
- Все 169+ cargo / 143+ vitest тестов PASS.

### b140.2: SUM полностью на единую сущность

**Цель:** удаление 1000 строк inline aggregation в `renderSumMode`.

- `evaluateSum` использует `evaluateBandFull` для каждой полосы.
- Aggregating logic (polarity, alignment_delay phase rotation,
  coherent sum) — в одной функции.
- `renderSumMode` читает result.
- IR для SUM — поле `result.ir`.
- Удалить inline `invoke evaluate_target / compute_peq_complex /
  compute_cross_section / compute_impulse` из `renderSumMode`.

**Acceptance:**
- E2E тест на SUM (3-полосный синтетический проект) → правильное
  суммирование.
- Snapshot golden на per-band components в SUM не разошёлся.

### b140.3: filterEquals + cleanup

**Цель:** закрыть оставшиеся точки обхода и удалить мёртвый код.

- `filterEquals` в peq-optimize.ts включает `linear_phase` и
  `subsonic_protect`.
- Удалить inline PEQ live-update в `FrequencyPlot.tsx:1547` если
  можно (или явно документировать как локальное UX-ускорение).
- Удалить `evaluate_target_standalone` если больше не используется.
- Финальный аудит: grep на `invoke.*evaluate_target` и
  `invoke.*compute_peq_complex` в `src/components/` → должно быть
  пусто.

**Acceptance:**
- Аудит чистый.
- regression-checklist 5 manual UI пунктов на финальном `.dmg`.

---

## Subagent оркестрация

После каждого этапа b140.X — оркестрировать subagent на верификацию:

```
Subagent: «Запусти cargo test и npm test в /Users/olegryzhikov/phaseforge.
Прогони e2e_export integration test. Отчитайся:
1. cargo test: N PASS / M FAIL — список failing.
2. vitest: N PASS / M FAIL.
3. e2e_export: 8 acceptance — какие PASS / FAIL с цифрами расхождения.
4. golden hash baseline: совпадает / drift.»
```

Только если все 4 пункта зелёные — этап считается завершённым.
При FAIL — diagnostic, не слепые правки.

---

## Что НЕ трогаем

- `peq-optimize.ts` — независимый pipeline (это не display, оптимизатор).
- `auto-align.ts` — независимый.
- `analyze_measurement` (b135) — отдельная функция.
- Rust apply_filter в target/mod.rs — math остаётся.

---

## Размер и сроки

- b140.0: 1 итерация. Тестовая инфраструктура.
- b140.1: 1 итерация. Главная архитектурная правка.
- b140.2: 1 итерация. SUM миграция.
- b140.3: 1 итерация. Cleanup.

После b140 — каскад регрессий невозможен в этой зоне. Любое будущее
расширение (новый тип фильтра, новое поле, новый режим экспорта) =
правка в одном месте.

---

## Главный принцип b140

**E2E тест с bit-exact проверкой экспорта — единственный надёжный
arbiter.** Все промежуточные snapshot и unit тесты — вспомогательные.
Если итоговый FIR не реализует target curve в пределах допуска —
любые внутренние тесты PASS неинтересны.

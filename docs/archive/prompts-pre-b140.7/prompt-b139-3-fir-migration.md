# Промт для Code: b139.3 — миграция FIR export на BandEvaluator

ТЗ целиком: `docs/TZ-unified-evaluation.md` (Этап 3).
Текущий билд: 0.1.0-b139.2 → bump до 0.1.0-b139.3.

## Контекст

В b139.2 SPL view одной полосы переключён на `evaluateBandFull`.
Этап 3 — переключить **FIR export pipeline и Export tab phase
preview**.

Это закрывает **два бага одновременно**:
1. FIR не учитывает PEQ phase contribution (передаётся только peqMag,
   без peqPhase) — скрытый баг существовал давно.
2. FIR при linear-phase Gaussian + subsonic не отражает subsonic
   phase rotation (regression checklist пункт 3 для Export view).

В b139.1 promt был known issue: `generate_model_fir` в MinimumPhase
mode может игнорировать переданный `model_phase` и делать собственный
Hilbert. Этот вопрос решается в этом этапе.

## Pre-flight audit (обязательно)

### 1. Прочитать `src-tauri/src/fir/mod.rs` — функции generate_*

Понять:
- Какие PhaseMode вариантов существуют (`LinearPhase`, `MinimumPhase`,
  возможно `Hybrid`).
- Как каждый mode использует `model_phase` параметр:
  - LinearPhase: использует напрямую как phase response?
  - MinimumPhase: игнорирует и делает свой Hilbert от target_mag?
  - Hybrid: что-то промежуточное.
- Есть ли возможность передать **готовую** min-phase которую Rust
  не пересчитает.

Зафиксировать findings явно. Без этих данных не двигаться дальше.

### 2. Найти callsites

```
grep -rn "generate_model_fir\|generateBandImpulse\|exportBandWav" src/
grep -n "generate_fir" src-tauri/src/
```

Классифицировать:
- `fir-export.ts:generateBandImpulse` — мигрируется.
- `fir-export.ts:exportBandWav` — wrapper, может остаться или
  обновиться вместе.
- Inline phase preview в `FrequencyPlot.tsx:~1712` — мигрируется.
- Любые другие — описать.

## Решение проблемы Rust phase

Один из трёх подходов в зависимости от audit:

**A. LinearPhase mode уже принимает явную `model_phase`** —
переключиться на него + передать combinedTargetPhase. Простейший
вариант, без изменений Rust.

**B. MinimumPhase mode принимает `model_phase` если не all-zeros** —
использовать его. Тоже без Rust изменений.

**C. Ни один mode не принимает explicit phase** — добавить новый
вариант `PhaseMode::Provided` в Rust enum который использует
`model_phase` как есть, без преобразований. Минимально-инвазивная
правка.

Выбор делается **на основе кода**, не предположения. Если решение
требует Rust изменений (вариант C) — добавить unit-тест в
`src-tauri/src/fir/mod.rs` на новый вариант.

## Что нужно сделать

### 1. Frontend: миграция `fir-export.ts:generateBandImpulse`

Заменить inline pipeline на `evaluateBandFull` с `includeFir: true`:

```typescript
async function generateBandImpulse(b: BandState): Promise<number[]> {
  const result = await evaluateBandFull({
    band: b,
    includeFir: true,
  });
  if (!result.fir) {
    throw new Error("FIR generation failed");
  }
  return result.fir.impulse;
}
```

Внутри `evaluateBandFull` (band-evaluator.ts) — обеспечить что
`includeFir: true` ветка передаёт полную combined phase в
`generate_model_fir` через выбранный подход (A/B/C).

### 2. Frontend: миграция Export tab inline phase preview

В `FrequencyPlot.tsx` (~line 1712) — точка где Export tab показывает
preview phase. Заменить inline вызов `evaluate_target` +
`addGaussianMinPhase` на чтение из `createBandEvalResource(activeBand)`
с `includeFir: true`.

Использовать тот же resource что для SPL view (если уже есть в
компоненте) — не создавать второй.

### 3. Что НЕ трогать

- SUM view (FrequencyPlot ~3590) — остаётся на legacy до Этапа 4.
- IR/Step plot для отдельной полосы — Этап 4.
- `src/lib/band-evaluation.ts` — старый код остаётся.
- `src/stores/peq-optimize.ts` — независимый pipeline.
- `src/lib/auto-align.ts` — независимый.
- `evaluate_target_standalone` — пока не удалять.

### 4. Тесты

**A. Cargo: golden FIR hash из b139.0 НЕ должен измениться.**

`generate_fir_b139_golden_lr4_baseline_impulse_hash` использует LR4
baseline без PEQ, без Gaussian, без subsonic. После Этапа 3 hash
должен совпадать (потому что для этих условий новое поведение не
активируется — PEQ phase нет, Gaussian phase нет).

Если hash меняется — это регрессия, остановиться, diagnostic.

**B. Vitest: snapshot тесты b139.0 и b139.1 НЕ должны измениться.**

Они тестируют helpers и evaluator, не FIR. Должны остаться зелёные.

**C. Новые cargo тесты: subsonic в FIR.**

Добавить в `src-tauri/src/fir/mod.rs` тесты:

```rust
#[test]
fn generate_fir_b139_3_gaussian_lin_subsonic_min_phase() {
    // Gaussian HP=632, linear_phase=true, subsonic ON, без PEQ
    // Generate FIR через выбранный подход (A/B/C).
    // Проверить: spectrum hash отличается от того же без subsonic.
    // Проверить: phase в зоне 5-50 Hz нелинейная (≠ 0).
}

#[test]
fn generate_fir_b139_3_gaussian_min_subsonic_min_phase() {
    // Gaussian HP=632, linear_phase=false, subsonic ON, без PEQ
    // Phase в зоне Gaussian — non-linear.
    // Phase в зоне subsonic — дополнительный rotation.
}

#[test]
fn generate_fir_b139_3_peq_phase_in_fir() {
    // LR4 HP=80 + PEQ полоса с высоким Q
    // Раньше FIR игнорировал peq phase contribution.
    // После b139.3 phase в зоне PEQ резонанса должна отражать min-phase
    // PEQ rotation.
    // Проверить: hash отличается от того же без PEQ.
}
```

### 5. Bump версии

- `src-tauri/tauri.conf.json` — version + productName/title до b139.3.
- `src-tauri/src/lib.rs` — startup-лог.
- После билда — skill `build-version`.

## Acceptance

1. **Golden FIR hash из b139.0** не изменился (LR4 baseline без PEQ
   без subsonic).
2. FIR с **Gaussian linear_phase=true + subsonic ON** — phase response
   отражает min-phase Butterworth в инфразвуковой зоне (новое
   поведение, фикс).
3. FIR с **Gaussian linear_phase=false + subsonic ON** — phase в обеих
   зонах (Gaussian + subsonic) корректна.
4. FIR с PEQ полосами — phase учитывает PEQ contribution (новое
   поведение, фикс скрытого бага).
5. FIR без Gaussian/subsonic/PEQ — bytewise идентичен предыдущему
   билду (regression baseline).
6. Export tab phase preview корректно отражает phase для всех 4
   Gaussian × subsonic комбинаций.
7. **regression-checklist 10 пунктов на `.dmg b139.3`** проходит.
   Особое внимание пункт 10 — экспорт WAV должен корректно работать.

## Регрессионная проверка

- b131-b139.2 функционал работает.
- vitest всё зелёное.
- cargo всё зелёное (включая golden hash и новые subsonic тесты).
- Реактивность Export tab при изменении target.
- Стандартный workflow: импорт → optimize → export — даёт
  работающий .wav.

## Что делать при провале acceptance

Если **golden hash из b139.0 изменился** — это сильный сигнал что
LR4 baseline pipeline случайно затронут. Diagnostic:
- Какой именно метод применил подход A/B/C?
- Не использует ли LR4 какой-то новый branch который раньше не
  активировался?

Если phase в FIR с subsonic не крутится после миграции — diagnostic
с логами в Rust generate_model_fir и frontend evaluateBandFull
includeFir branch. НЕ слепые правки.

## Учёт уроков b138-b139 каскада

1. **Audit before write.** Прочитать Rust `generate_model_fir` целиком
   до решения подхода A/B/C. Не предполагать.
2. **Diagnostic-first при провале.** Hash mismatch — стоп.
3. **Минимально-инвазивные изменения в Rust.** Если требуется новый
   PhaseMode — добавить, не модифицируя существующие режимы.
4. **Версия в заголовке** = b139.3.
5. **Один этап = один коммит.**

## Тестировать на `.dmg`

После сборки запустить
`src-tauri/target/release/bundle/dmg/PhaseForge_0.1.139-3_aarch64.dmg`,
проверить версию = b139.3, прогнать `docs/regression-checklist.md`.

Дополнительный manual тест:
- Создать полосу с Gaussian HP=632, linear_phase=TRUE, subsonic ON.
- Export FIR. Импортировать .wav в любой DAW или сторонний
  визуализатор IR (или прямо в PhaseForge через Import) — проверить
  что phase response отражает subsonic rotation.

## Правила (CLAUDE.md)

- Один коммит: `refactor: FIR export uses BandEvaluator (b139.3)`
  + Co-Authored-By.
- 7-vector review.
- Без нарратива прогресса.
- Cargo unit-тесты для нового FIR поведения обязательны.
- `cargo tauri build` для финальной сборки.

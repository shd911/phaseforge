# Промт для Code: b140.2.0 — автотесты для evaluateSum (без UI изменений)

ТЗ контекст: `docs/TZ-b140-total-rebuild.md` (этап SUM миграции).
Текущий билд: 0.1.0-b140.1 → bump до 0.1.0-b140.2.0.

## Контекст

В `src/lib/band-evaluator.ts:502-586` существует функция `evaluateSum`,
но она **dead code** — никем не вызывается. Параллельная inline
логика (~590 строк) живёт в `FrequencyPlot.tsx:renderSumMode` и
~400 строк в SUM ветке `renderTimeTab`.

Стратегия: построить параллельный pipeline через `evaluateSum`, в UI
будет переключатель Legacy/New (b140.2.1). Перед интеграцией — этот
этап покрывает `evaluateSum` автотестами на synthetic 2-3-полосных
проектах. Никаких изменений в production коде на этом этапе.

## Что нужно сделать

### 1. Synthetic SUM fixtures

`src-tauri/tests/sum_fixtures.rs` (новый) или расширение
`src-tauri/tests/fixtures.rs`. Несколько эталонных проектов:

```rust
// Проект из 2 полос с разной полосой пропускания
pub fn fixture_two_band_lr_lp() -> Vec<BandConfig> {
    vec![
        BandConfig { hp: None, lp: Some(lr4(500.0)), inverted: false, alignment_delay: 0.0 },
        BandConfig { hp: Some(lr4(500.0)), lp: None, inverted: false, alignment_delay: 0.0 },
    ]
}

// Проект с инверсией полярности
pub fn fixture_two_band_with_polarity() -> ... {
    // band 2 inverted=true
}

// Проект с alignment delay
pub fn fixture_two_band_with_delay() -> ... {
    // band 2 alignment_delay = 0.5 ms
}

// Проект из 3 полос (3-way)
pub fn fixture_three_band() -> ...

// Проект где не у всех полос есть phase (incoherent fallback)
pub fn fixture_mixed_phase_availability() -> ...
```

### 2. Vitest unit тесты для `evaluateSum`

Файл: `src/lib/__tests__/evaluate-sum.test.ts` (новый).

Mock invoke (как в `band-evaluator.test.ts`) для деterminистичных
тестов без Tauri runtime. Проверки:

```typescript
describe("evaluateSum — coherent magnitude", () => {
  it("two bands with identical flat response sum to +6 dB", async () => {
    // Две полосы с magnitude=0 dB и phase=0 на всём диапазоне
    const bands = makeBandsWithFlatResponse(2);
    const result = await evaluateSum(bands, {});

    // Coherent sum: 1 + 1 = 2 amplitude → +6 dB
    expect(result.sumCorrectedMag![passbandIdx]).toBeCloseTo(6.02, 1);
  });

  it("two bands with opposite polarity sum to -∞ dB (cancellation)", async () => {
    const bands = makeBandsWithFlatResponse(2);
    bands[1].inverted = true;
    const result = await evaluateSum(bands, {});

    // 1 + (-1) = 0 → very low
    expect(result.sumCorrectedMag![passbandIdx]).toBeLessThan(-60);
  });
});

describe("evaluateSum — alignment delay rotation", () => {
  it("delay shifts phase: 0.5 ms at 1 kHz → -180°", async () => {
    const bands = makeBandsWithFlatResponse(1);
    bands[0].alignmentDelay = 0.0005;  // 0.5 ms
    const result = await evaluateSum(bands, {});

    const idx1k = result.freq.findIndex(f => f >= 1000);
    expect(result.sumTargetPhase![idx1k]).toBeCloseTo(-180, 0);
  });
});

describe("evaluateSum — incoherent fallback", () => {
  it("when one band lacks phase, falls back to power sum", async () => {
    const bands = makeBandsWithFlatResponse(2);
    bands[1].measurement!.phase = null;
    const result = await evaluateSum(bands, {});

    // Power sum: 0 dB + 0 dB = +3 dB (10·log10(2))
    expect(result.sumCorrectedMag![passbandIdx]).toBeCloseTo(3.01, 1);
  });
});

describe("evaluateSum — snapshot regression", () => {
  // 5 фикстур × snapshot mag/phase
  // Защита от случайных изменений алгоритма.
});
```

### 3. Е2Е cargo тесты для SUM

`src-tauri/tests/e2e_sum.rs` (новый):

Проверка numerical equivalence между legacy aggregation (как в
`renderSumMode`) и `evaluateSum` для одинаковых fixture проектов.
Если расхождение > 0.1 dB / 1° в полосе пропускания — флаг для
ручной проверки (может быть улучшение, может быть регрессия).

```rust
#[test]
fn e2e_sum_two_band_coherent_magnitude() {
    let bands = fixture_two_band_lr_lp();
    let result = run_evaluate_sum(&bands);

    // Acceptance: SUM в полосе перекрытия (около crossover 500 Hz)
    // должен быть в пределах допуска от ожидаемого LR4 sum.
    let idx_xo = result.freq.iter().position(|&f| f >= 500.0).unwrap();
    assert!((result.sum_target_mag[idx_xo] - 0.0).abs() < 0.5,
        "LR4 crossover SUM mag at 500 Hz: expected 0 dB, got {:.2}",
        result.sum_target_mag[idx_xo]);
}
```

### 4. Активация evaluateSum (только тесты)

Этот этап **не вызывает** `evaluateSum` из production кода. UI
по-прежнему использует inline pipeline. Но `evaluateSum` теперь
покрыта тестами и можно безопасно интегрировать в b140.2.1.

### 5. Bump

- `src-tauri/tauri.conf.json` → `0.1.140`.
- `src-tauri/src/lib.rs` startup → b140.2.0.
- `src/lib/version.ts` → b140.2.0.

## Acceptance

1. 5+ vitest unit тестов на `evaluateSum` (coherent, polarity, delay, incoherent, snapshot).
2. 3+ cargo e2e тестов для SUM aggregation на synthetic проектах.
3. Snapshot baseline зафиксирован.
4. Все existing 170+ cargo тестов PASS.
5. Vitest 143+ + новые тесты PASS.
6. Никаких изменений в `FrequencyPlot.tsx` или другом production коде.
7. `evaluateSum` остаётся dead-code в production, но теперь полностью
   покрыта автотестами.

## Что НЕ делать

- Не интегрировать `evaluateSum` в renderSumMode — это b140.2.1.
- Не удалять inline aggregation — пока работает legacy.
- Не делать UI переключатель — это b140.2.1.

## Что прислать обратно

```
vitest: N PASS / M FAIL — список новых evaluate-sum тестов
cargo: N PASS / M FAIL — e2e_sum тесты
existing: PASS count

Snapshot baseline для evaluateSum зафиксирован: yes/no.
```

## Правила

- Один коммит: `test: cover evaluateSum with unit + e2e tests (b140.2.0)` + Co-Authored-By.
- 7-vector review.
- Без нарратива.
- При FAIL — diagnostic, не слепые правки.

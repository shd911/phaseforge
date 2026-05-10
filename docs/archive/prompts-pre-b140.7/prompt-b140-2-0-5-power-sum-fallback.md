# Промт для Code: b140.2.0.5 — power-sum fallback в evaluateSum

Текущий билд: 0.1.0-b140.2.0 → bump до 0.1.0-b140.2.0.5.

## Контекст

В b140.2.0 покрыли `evaluateSum` тестами. Найдена разница с legacy
inline aggregation в `FrequencyPlot.tsx:renderSumMode`:

- **Legacy:** при отсутствии phase у части полос — falls back на
  power-sum (некогерентная сумма): `mag_dB = 10·log10(Σ 10^(mag_i/10))`.
  Полярность игнорируется.
- **New `evaluateSum`:** полосы без phase **drop-аются** из corrected
  суммы целиком.

Для паритета поведения при переключении Legacy/New SUM (b140.2.1) —
добавить fallback в `evaluateSum`.

## Что нужно сделать

### 1. Логика fallback в `evaluateSum`

В `src/lib/band-evaluator.ts:evaluateSum` определить mixed-phase
condition:

```typescript
const enabledBands = perBand.filter(/* условие что в SUM */);
const allHavePhase = enabledBands.every(b =>
  b.measurementPhase !== null  // или соответствующее поле
);

if (allHavePhase) {
  // Существующая coherent-sum логика
} else {
  // Power-sum fallback:
  // amp_lin[j] = sqrt(Σ 10^(mag_i[j] / 10))
  // mag_db[j] = 20·log10(amp_lin[j]) = 10·log10(Σ 10^(mag_i[j] / 10))
  // sum_phase = null (без phase в power-sum)
}
```

Применить fallback к `sumCorrectedMag/Phase`.

`sumTargetMag` и `sumTargetPhase` — target curve есть всегда с phase
(строится через evaluate_target + reconstructTargetPhase),
fallback не нужен.

### 2. Возвращаемый результат

`SumEvalResult` должен содержать признак какая ветка использовалась
(для отладки и UI feedback):

```typescript
interface SumEvalResult {
  // ... existing fields
  coherent: boolean;  // true если использовалась когерентная сумма
}
```

### 3. Vitest тесты на fallback

В `src/lib/__tests__/evaluate-sum.test.ts`:

```typescript
describe("evaluateSum — power-sum fallback", () => {
  it("two flat bands without phase sum to +3 dB (incoherent)", async () => {
    const bands = makeBandsWithFlatResponse(2);
    bands[0].measurement!.phase = null;
    bands[1].measurement!.phase = null;
    const result = await evaluateSum(bands, {});

    // Power sum: 10·log10(2) = 3.01 dB
    expect(result.sumCorrectedMag![passbandIdx]).toBeCloseTo(3.01, 1);
    expect(result.sumCorrectedPhase).toBeNull();
    expect(result.coherent).toBe(false);
  });

  it("mixed phase availability triggers power-sum fallback", async () => {
    const bands = makeBandsWithFlatResponse(2);
    bands[1].measurement!.phase = null;  // одна без phase
    const result = await evaluateSum(bands, {});

    // Mixed: power sum
    expect(result.sumCorrectedMag![passbandIdx]).toBeCloseTo(3.01, 1);
    expect(result.coherent).toBe(false);
  });

  it("polarity ignored in power-sum (no cancellation)", async () => {
    const bands = makeBandsWithFlatResponse(2);
    bands[0].measurement!.phase = null;
    bands[1].measurement!.phase = null;
    bands[1].inverted = true;  // в coherent дало бы cancellation
    const result = await evaluateSum(bands, {});

    // Power sum: polarity не влияет
    expect(result.sumCorrectedMag![passbandIdx]).toBeCloseTo(3.01, 1);
  });

  it("all bands with phase keeps coherent sum (no fallback)", async () => {
    const bands = makeBandsWithFlatResponse(2);
    // both have phase
    const result = await evaluateSum(bands, {});

    expect(result.sumCorrectedMag![passbandIdx]).toBeCloseTo(6.02, 1);
    expect(result.coherent).toBe(true);
  });
});
```

### 4. Cargo e2e обновление

В `src-tauri/tests/e2e_sum.rs` добавить тест incoherent:

```rust
#[test]
fn e2e_sum_incoherent_fallback() {
    let mut bands = fixture_two_band_lr_lp();
    bands[0].measurement.phase = None;
    let result = run_evaluate_sum(&bands);

    assert!(!result.coherent, "expected incoherent fallback");
    // Дополнительно: проверить mag = 10·log10(Σ 10^(m_i/10))
}
```

### 5. Документация в коде

Краткий comment в `evaluateSum`:

```typescript
// b140.2.0.5: при отсутствии phase у любой полосы — fallback на
// power-sum (incoherent). Это паритет с legacy renderSumMode.
// `result.coherent` отражает какая ветка использовалась.
```

### 6. Bump

- `src-tauri/tauri.conf.json` → `0.1.140`.
- `src-tauri/src/lib.rs` startup → b140.2.0.5.
- `src/lib/version.ts` → b140.2.0.5.

## Acceptance

1. Все 14 ранее vitest тестов на evaluateSum остаются PASS.
2. Новые 4 теста на power-sum fallback — PASS.
3. Все 6 e2e_sum cargo тестов остаются PASS, плюс новый incoherent e2e.
4. Все existing 176+ cargo / 157+ vitest тестов остаются PASS.
5. `evaluateSum` теперь имеет паритет с legacy aggregation для
   mixed-phase случая.
6. Никаких изменений в production коде вне band-evaluator.ts.

## Что НЕ делать

- Не интегрировать `evaluateSum` в renderSumMode — это b140.2.1.
- Не удалять inline aggregation.
- Не делать UI переключатель.
- Не менять target SUM (target всегда coherent, изменения только
  для corrected sum).

## Что прислать обратно

```
vitest: 161 PASS (157 + 4)
cargo: 177 PASS (176 + 1 e2e)
existing: PASS

Fallback verified: incoherent path returns coherent=false и
правильную mag через power sum.
```

## Правила

- Один коммит: `feat: power-sum fallback in evaluateSum for legacy parity (b140.2.0.5)` + Co-Authored-By.
- 7-vector review.
- Без нарратива.

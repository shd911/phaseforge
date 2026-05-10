# Промт для Code: b140.2.1.3 — globalRef + corrOffset в evaluateSum

Текущий билд: 0.1.0-b140.2.1.1 → bump до 0.1.0-b140.2.1.3.

## Контекст

Diff test b140.2.1.2 локализовал баг: `evaluateSum` использует
per-band `autoRef` для каждой полосы (свой passband-avg), а Legacy
использует **globalRef** — максимальный passband-avg среди всех
полос, и поднимает каждую target к этому уровню. Для multi-way систем
target tweeter тонет на 35 dB ниже target woofer в New, а в Legacy
все выровнены.

Также Legacy после aggregation применяет **corrOffset** — выравнивает
sumCorrectedMag к sumTargetMag в passband 200–2000 Hz. New этого не
делает.

## Что нужно сделать

### 1. `BandEvalRequest` — поддержка refLevelOverride

В `src/lib/band-evaluator.ts`:

```typescript
export interface BandEvalRequest {
  band: BandState;
  freq?: number[];
  fir?: FirRequestConfig;
  includeIr?: boolean;
  /** Если задан — использовать как target reference level вместо
   *  autoRef. evaluateSum передаёт globalRef для multi-band паритета. */
  refLevelOverride?: number;
}
```

В `evaluateBandFull` логика выбора reference level:

```typescript
const refLevel = req.refLevelOverride !== undefined
  ? req.refLevelOverride
  : (measurement
    ? autoRefLevel(measurement.freq, measurement.magnitude, ...)
    : (targetCurve.reference_level_db ?? 0));
```

### 2. `evaluateSum` — два прохода

**Проход 1:** для каждой полосы вычислить passband-avg (200–2000 Hz)
от measurement. Это дёшево — read-only вычисление по существующей
measurement magnitude, без полного evaluateBandFull.

**Глобально:** `globalRef = Math.max(...passbandAvgs)`.

**Проход 2:** `evaluateBandFull({band: b, refLevelOverride: globalRef})`
для каждой полосы. Все targets теперь выровнены к одному уровню.

```typescript
export async function evaluateSum(bands: BandState[], options?: SumEvalOptions): Promise<SumEvalResult> {
  // 1. Compute global reference: max passband-avg across bands
  const passbandAvgs = bands.map(b => {
    if (!b.measurement) return -Infinity;
    return passbandAvgDb(b.measurement.freq, b.measurement.magnitude, 200, 2000);
  }).filter(v => isFinite(v));
  const globalRef = passbandAvgs.length > 0 ? Math.max(...passbandAvgs) : 0;

  // 2. Per-band evaluation с глобальным ref
  const perBand = await Promise.all(bands.map(b =>
    evaluateBandFull({ band: b, refLevelOverride: globalRef })
  ));

  // ... existing aggregation logic
}

function passbandAvgDb(freq: number[], mag: number[], fLo: number, fHi: number): number {
  let sum = 0, n = 0;
  for (let i = 0; i < freq.length; i++) {
    if (freq[i] >= fLo && freq[i] <= fHi) { sum += mag[i]; n++; }
  }
  return n > 0 ? sum / n : 0;
}
```

### 3. `evaluateSum` — corrOffset после aggregation

После coherent/incoherent sum sumCorrectedMag и sumTargetMag — выровнять:

```typescript
if (sumCorrectedMag && sumTargetMag) {
  const passbandIdx0 = freq.findIndex(f => f >= 200);
  const passbandIdx1 = freq.findIndex(f => f > 2000);
  let targetSum = 0, corrSum = 0, count = 0;
  for (let i = passbandIdx0; i < passbandIdx1; i++) {
    if (isFinite(sumCorrectedMag[i]) && isFinite(sumTargetMag[i])) {
      targetSum += sumTargetMag[i];
      corrSum += sumCorrectedMag[i];
      count++;
    }
  }
  if (count > 0) {
    const corrOffset = (targetSum - corrSum) / count;
    if (Math.abs(corrOffset) > 0.01) {
      sumCorrectedMag = sumCorrectedMag.map(v => v + corrOffset);
    }
  }
}
```

### 4. Vitest тесты

Существующие 14+ vitest на evaluateSum должны проходить. Для них
globalRef = их единственный autoRef → результат тот же.

Новый тест:

```typescript
it("multi-band system: target uses globalRef, not per-band autoRef", async () => {
  // Band 1: passband -10 dB, Band 2: passband -40 dB
  // globalRef = -10 (max). Both targets shift to -10 baseline.
  // New (без фикса): Band 1 target at -10, Band 2 target at -40 — 30 dB разницы.
  // С фиксом: оба target вокруг -10.
  const bands = makeMultiBandWithDifferentLevels([-10, -40]);
  const result = await evaluateSum(bands, {});

  // Sum target в полосе пропускания где обе полосы перекрываются —
  // ожидаем ~ -10 + 6 dB (coherent sum двух drivers at same level).
  const idx = passbandIdx;
  expect(result.sumTargetMag![idx]).toBeCloseTo(-4, 0);  // -10 + 6
});
```

### 5. Cargo diff test после фикса

Запустить b140.2.1.2 diff test (`cargo test --ignored
diff_legacy_vs_new_5wayNew`). После фикса:

```
Σ target mag: max diff < 0.5 dB
Σ corrected mag: max diff < 0.5 dB
Σ measurement: уже в паритете (был 0.002 dB)
```

Если всё ещё >0.5 dB на каких-то частотах — **stop**, дополнительная
diagnostic, не слепые правки.

### 6. Bump

- `src-tauri/tauri.conf.json` → `0.1.140`.
- `src-tauri/src/lib.rs` startup → b140.2.1.3.
- `src/lib/version.ts` → b140.2.1.3.

## Acceptance

1. Diff test после фикса: max diff < 0.5 dB на всех Σ кривых.
2. Visual паритет с Legacy в New режиме на проекте 5wayNew.
3. Existing 164+ vitest + 178+ cargo тестов PASS.
4. Новый vitest тест на multi-band globalRef PASS.

## Что НЕ делать

- Не трогать legacy renderSumMode.
- Не менять SUM IR (это b140.2.2).
- Не трогать peq-optimize / auto-align.

## Тестировать на `.dmg`

После сборки запустить b140.2.1.3, открыть 5wayNew, переключить
Legacy ↔ New. Кривые должны совпадать визуально (mag и phase в
пределах 0.5 dB / 5°).

## Правила

- Один коммит: `fix: globalRef + corrOffset for SUM target/corrected parity (b140.2.1.3)` + Co-Authored-By.
- 7-vector review.
- Без нарратива.

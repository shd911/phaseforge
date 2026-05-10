# Промт для Code: b140.2.1.5 — per-band кривые на common grid

Текущий билд: 0.1.0-b140.2.1.4 → bump до 0.1.0-b140.2.1.5.

## Контекст

В b140.2.1.1 в `renderSumModeNew` per-band entries пушатся с native
freq grid (длина массива измерения), но uPlot x-ось — common grid от
evaluateSum (другая длина). uPlot растягивает короткий массив на
весь grid → каждая полоса визуально занимает не свой диапазон.

Подтверждено пользователем: на 5wayNew все ачх кроме баса отображены
не на своих частотах.

Фикс: per-band данные тоже использовать **resampled** (с fence из
b140.2.1.4) вместо native.

## Что нужно сделать

### 1. evaluateSum — экспортировать resampled per-band данные

В `src/lib/band-evaluator.ts`:

```typescript
interface SumEvalResult {
  // ... existing
  /** Per-band data, ресемпленная на common grid (с fence для bins
   *  вне native freq диапазона полосы). Используется UI для отрисовки
   *  per-band curves когда ось общей сетки. */
  perBandResampled: Array<{
    measurementMag: number[] | null;
    measurementPhase: number[] | null;
    targetMag: number[] | null;
    targetPhase: number[] | null;
    correctedMag: number[] | null;
    correctedPhase: number[] | null;
  }>;
}
```

В функции evaluateSum уже есть resampling for aggregation. Сохранить
эти resampled данные в `perBandResampled[i]` параллельно.

Один additional resample для measurement (если он не делался для
Σ measurement) — но он будет делаться после b140.2.1.1, так что
всё уже там.

### 2. renderSumModeNew — использовать resampled

В `src/components/FrequencyPlot.tsx`:

```typescript
// Per-band measurement
const resampled = result.perBandResampled[i];
if (resampled.measurementMag) {
  uSeries.push({...});
  uData.push(resampled.measurementMag);  // ← теперь длина = freq.length
  legend.push({...});
}
```

То же для target / corrected per-band.

### 3. Vitest тест на длины массивов

```typescript
it("perBandResampled have same length as common grid", async () => {
  const bands = makeBandsWithMixedRanges([[20, 22000], [1000, 22000]]);
  const result = await evaluateSum(bands, {});

  for (const r of result.perBandResampled) {
    if (r.measurementMag) {
      expect(r.measurementMag.length).toBe(result.freq.length);
    }
    if (r.targetMag) {
      expect(r.targetMag.length).toBe(result.freq.length);
    }
  }
});

it("supertweeter perBandResampled silent below native range", async () => {
  const bands = makeBandsWithMixedRanges([[20, 22000], [1000, 22000]]);
  const result = await evaluateSum(bands, {});

  const idx100 = result.freq.findIndex(f => f >= 100);
  const supertweeter = result.perBandResampled[1];
  // Below 1000 Hz — должно быть -200 dB (fence)
  expect(supertweeter.measurementMag![idx100]).toBeLessThan(-150);
});
```

### 4. Bump

- `src-tauri/tauri.conf.json` → `0.1.140`.
- `src-tauri/src/lib.rs` startup → b140.2.1.5.
- `src/lib/version.ts` → b140.2.1.5.

## Acceptance

1. На 5wayNew в New SUM все per-band ачх показаны на своих native
   частотах (Band 5 supertweeter — только выше 1220 Hz).
2. Visual паритет с Legacy на per-band кривых.
3. Σ Meas/Target/Corrected без изменений (уже работают).
4. Existing 169+ vitest + 178+ cargo PASS.
5. Новые тесты на длины массивов и fence per-band PASS.

## Что НЕ делать

- Не трогать Σ aggregation logic — она уже корректна.
- Не менять resampleOntoGrid сам — fence в нём уже работает.

## Тестировать на `.dmg`

Открыть 5wayNew → SUM → New → включить per-band measurement каждой
полосы. Все 5 кривых должны отображаться на своих freq диапазонах,
не растягиваться на всю ось.

## Правила

- Один коммит: `fix: per-band SUM curves use resampled data on common grid (b140.2.1.5)` + Co-Authored-By.
- 7-vector review.
- Без нарратива.

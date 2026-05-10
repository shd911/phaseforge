# Промт для Code: b140.2.1.1 — UI паритет New SUM с Legacy

Текущий билд: 0.1.0-b140.2.1 → bump до 0.1.0-b140.2.1.1.

## Контекст

Скриншот показал что в New SUM:
- В колонке Σ legend grid есть только **CORR+XO** чекбокс
- Чекбоксы Σ **TARGETS** и Σ **MEAS** отсутствуют
- Per-band колонки (Band 1–5) полностью пустые

Корень: UI legend grid строится из per-band entries. New рисует
только агрегаты Σ — grid не имеет per-band заголовков и пропускает
строки. Также `evaluateSum` не возвращает Σ Measurement aggregate
(только target и corrected).

## Что нужно сделать

### 1. Расширение `evaluateSum` — Σ measurement

В `src/lib/band-evaluator.ts`:

```typescript
interface SumEvalResult {
  // ...existing
  sumMeasurementMag: number[] | null;
  sumMeasurementPhase: number[] | null;
}
```

Логика — аналогично corrected но из per-band `measurementMag` /
`measurementPhase`:
- Resample на common grid.
- Coherent sum если все полосы имеют phase, polarity и delay
  применяются.
- Incoherent fallback если у любой полосы нет phase (как для corrected).
- Если ни одна полоса не имеет measurement — `null`.

### 2. Vitest тесты на Σ measurement

В `src/lib/__tests__/evaluate-sum.test.ts`:

```typescript
describe("evaluateSum — Σ measurement", () => {
  it("two flat measurements with phase sum to +6 dB coherent");
  it("two flat measurements without phase fall back to power-sum +3 dB");
  it("polarity inversion in coherent measurement causes cancellation");
  it("no measurements returns null");
});
```

### 3. renderSumModeNew — per-band entries для UI grid

В `src/components/FrequencyPlot.tsx:renderSumModeNew`:

После Σ entries добавить **per-band** entries для measurement,
target, corrected. Они работают как в legacy — отдельные кривые
которые можно скрыть/показать через legend.

```typescript
for (let i = 0; i < result.perBand.length; i++) {
  const r = result.perBand[i];
  const band = bands[i];
  const cf = bandColorFamily(band.color);

  // Per-band measurement
  if (r.measurementMag) {
    uSeries.push({
      label: `${band.name} dB`, stroke: cf.meas, width: 1.5, scale: "mag",
    });
    uData.push(r.measurementMag);
    legend.push({
      label: "Measurement", color: cf.meas, dash: false, visible: false,
      seriesIdx: sIdx, category: "measurement",
      bandIndex: i,  // если LegendEntry поддерживает
    });
    sIdx++;
    if (showPhase && r.measurementPhase) {
      uSeries.push({
        label: `${band.name} °`, stroke: cf.measPhase, width: 1, dash: [6,3], scale: "phase",
      });
      uData.push(wrapPhase(r.measurementPhase));
      legend.push({
        label: "Meas °", color: cf.measPhase, dash: true, visible: false,
        seriesIdx: sIdx, category: "measurement",
      });
      sIdx++;
    }
  }

  // Per-band target и corrected — аналогично, читая из r.targetMag/Phase
  // и r.correctedMag/Phase
}
```

Visibility всех per-band — `false` по умолчанию (UI grid отобразит
чекбоксы пустыми, пользователь сам включит то что хочет видеть).

Проверить совместимость с `LegendEntry` интерфейсом — возможно
нужно добавить поле `bandIndex` для grid layout если оно используется.

### 4. Σ Measurement в UI

После расширения evaluateSum — добавить Σ Measurement в
renderSumModeNew:

```typescript
if (result.sumMeasurementMag) {
  uSeries.push({
    label: result.coherentMeasurement ? "Σ meas (New)" : "Σ meas (New, incoh)",
    stroke: SUM_MEAS_COLOR, width: 2.5, scale: "mag",
  });
  uData.push(result.sumMeasurementMag);
  legend.push({ /* ... category: "measurement" */ });
  sIdx++;
  if (showPhase && result.sumMeasurementPhase) { /* phase */ }
}
```

`coherentMeasurement` — отдельный флаг от `coherent` (который про
corrected). Возможно объединить как `coherentSums: { measurement, corrected }`.

### 5. Bump

- `src-tauri/tauri.conf.json` → `0.1.140`.
- `src-tauri/src/lib.rs` startup → b140.2.1.1.
- `src/lib/version.ts` → b140.2.1.1.

## Acceptance

1. На скриншоте Кирилла теперь должны быть:
   - В колонке Σ: чекбоксы TARGETS, MEAS, CORR+XO (все три).
   - В per-band колонках: чекбоксы для каждой полосы по соответствующим
     категориям.
2. По умолчанию визуально — те же кривые что в Legacy (Σ Target /
   Σ Corrected / Σ Measurement). Per-band — скрыты до клика.
3. Vitest 160+ + 4 новых = PASS.
4. Cargo тесты PASS.
5. Switch Legacy ↔ New сохраняет visible state по category (как
   написано в существующей логике sumVisMap).

## Что НЕ делать

- Не трогать legacy renderSumMode.
- Не реорганизовывать UI legend компонент (только наполнение его
  данными).
- Не менять SUM IR/Step (это b140.2.2).

## Тестировать на `.dmg`

После сборки на проекте 5wayNew (как на скриншоте Кирилла):
- Switch на New → в legend все три чекбокса в Σ + per-band
  чекбоксы появились.
- Кликнуть Σ MEAS → Σ measurement кривая появляется.
- Кликнуть Band 1 MEAS → per-band measurement кривая появляется.
- Visual diff с Legacy на тех же visibility — должен быть
  минимальным (numerical equivalence из b140.2.0 уже подтверждена).

## Правила

- Один коммит: `feat: per-band entries + Σ measurement in New SUM (b140.2.1.1)` + Co-Authored-By.
- 7-vector review.
- Без нарратива.

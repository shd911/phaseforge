# Промт для Code: b140.3.0.1 — per-band target curves в legend grid

Текущий билд: 0.1.0-b140.3.0 → bump до 0.1.0-b140.3.0.1.

## Самооценка эффективности

| Критерий | Оценка | Комментарий |
|---|---|---|
| Размер | ✅ малый | Только UI / per-band entries в legend |
| Pre-flight audit | ✅ | Уже знаем что legend grid нужны per-band entries |
| Гипотезы без данных | ✅ | UI grouping ожидает per-band rows, без них Σ checkbox не отображается |

## Контекст

В b140.3.0 убрали все per-band кривые из renderSumModeNew, оставили
только Σ Target. UI legend grid не отображает Σ checkbox без per-band
строк (та же проблема что в b140.2.1.1).

Этот промт добавляет per-band target curves обратно. Только target,
без measurement / corrected (они вернутся в b140.3.1+).

## Что нужно сделать

### 1. evaluateSum — экспортировать per-band target

В `src/lib/band-evaluator.ts`:

```typescript
export interface SumEvalResult {
  freq: number[];
  sumTargetMag: number[] | null;
  sumTargetPhase: number[] | null;
  /** Per-band target данные на common grid. Для UI отрисовки кривых
   *  отдельных полос (visible:false по умолчанию). */
  perBandTarget: Array<{
    mag: number[];
    phase: number[];
  } | null>;
}
```

В функции — после per-band evaluate_target сохранять resampled data
(уже на common grid, не нужен дополнительный resample т.к. evaluate_target
вызывается с common freq):

```typescript
const perBandTarget: Array<{mag: number[]; phase: number[]} | null> = [];

for (const band of bands) {
  if (!band.targetEnabled) {
    perBandTarget.push(null);
    perBandTargetData.push(null);
    continue;
  }
  const response = await invoke("evaluate_target", { target: band.target, freq });
  const phase = await reconstructTargetPhase(...);
  perBandTarget.push({ mag: response.magnitude, phase });
  perBandTargetData.push({ mag: response.magnitude, phase, sign, delay });
}
```

### 2. renderSumModeNew — per-band target в legend

После Σ Target series — добавить per-band:

```typescript
// Per-band target curves (visible:false по умолчанию)
for (let i = 0; i < bands.length; i++) {
  const pb = result.perBandTarget[i];
  if (!pb) continue;
  const cf = bandColorFamily(bands[i].color);

  uSeries.push({
    label: `${bands[i].name} tgt`,
    stroke: cf.target, width: 1.5, dash: [8, 4], scale: "mag",
  });
  uData.push(pb.mag);
  legend.push({
    label: "Target", color: cf.target, dash: false,
    visible: false, seriesIdx: sIdx, category: "target",
    bandIndex: i,  // если LegendEntry поддерживает
  });
  sIdx++;

  if (showPhase) {
    uSeries.push({
      label: `${bands[i].name} tgt °`,
      stroke: cf.targetPhase, width: 1, dash: [4, 4], scale: "phase",
    });
    uData.push(wrapPhase(pb.phase));
    legend.push({
      label: "Target °", color: cf.targetPhase, dash: true,
      visible: false, seriesIdx: sIdx, category: "target",
    });
    sIdx++;
  }
}
```

Структура legend grid: per-band строки + Σ строка для category "target".
UI отображает чекбоксы во всех ячейках.

### 3. Vitest

Добавить тест:

```typescript
it("evaluateSum returns perBandTarget for enabled bands", async () => {
  const bands = makeTwoEnabledBands();
  const result = await evaluateSum(bands, {});
  expect(result.perBandTarget).toHaveLength(2);
  expect(result.perBandTarget[0]).not.toBeNull();
  expect(result.perBandTarget[0]!.mag).toHaveLength(result.freq.length);
});

it("perBandTarget is null for disabled bands", async () => {
  const bands = makeTwoBandsOneDisabled();
  const result = await evaluateSum(bands, {});
  expect(result.perBandTarget[1]).toBeNull();
});
```

### 4. Bump

- `src-tauri/tauri.conf.json` → `0.1.140`.
- `src-tauri/src/lib.rs` startup → b140.3.0.1.
- `src/lib/version.ts` → b140.3.0.1.

## Acceptance

1. На 5wayNew в New SUM legend grid имеет чекбоксы:
   - Σ TARGETS (видна по умолчанию).
   - Per-band Target (по умолчанию скрыта, можно включить кликом).
2. При клике на per-band target — соответствующая кривая полосы
   рисуется на графике.
3. Σ Target — coherent sum target curves каждой полосы (как в b140.3.0).
4. existing 64+ vitest PASS.

## Что НЕ делать

- Не добавлять measurement / corrected — это b140.3.1+.
- Не пытаться normalize / shift уровни — оставляем absolute SPL.

## End-of-prompt автозапуск dev

```
pkill -f "PhaseForge" 2>/dev/null || true
lsof -ti:1420 | xargs kill -9 2>/dev/null || true
cd /Users/olegryzhikov/phaseforge && nohup cargo tauri dev > /tmp/phaseforge-dev.log 2>&1 &
```

## Правила

- Один коммит: `feat: per-band target curves in New SUM legend (b140.3.0.1)` + Co-Authored-By.
- Без нарратива.

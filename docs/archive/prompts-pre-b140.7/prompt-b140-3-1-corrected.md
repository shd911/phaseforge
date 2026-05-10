# Промт для Code: b140.3.1 — добавить CORR кривые в SUM

Текущий билд: 0.1.0-b140.3.0.1 → bump до 0.1.0-b140.3.1.

## Самооценка эффективности

| Критерий | Оценка | Комментарий |
|---|---|---|
| Размер | ✅ малый | Один категория кривых добавляется |
| Pre-flight audit | ✅ | evaluateBandFull уже считает correctedMag/Phase для каждой полосы |
| Гипотезы без данных | ✅ | Coherent sum corrected — стандартный алгоритм |
| Учёт уроков | ✅ | Никакого legacy mimicry: чистая аггрегация без normalize/shift магии |

## Контекст

В b140.3.0 / b140.3.0.1 в New SUM есть только Σ Target и per-band
target. Этот шаг добавляет CORR (corrected = measurement + PEQ +
cross-section).

Используем `evaluateBandFull` для каждой полосы — он уже корректно
вычисляет correctedMag/Phase с реконструкцией Gaussian/subsonic phase.
Просто берём результат и складываем coherent sum.

## Что нужно сделать

### 1. evaluateSum — добавить corrected aggregation

В `src/lib/band-evaluator.ts`:

```typescript
export interface SumEvalResult {
  freq: number[];
  sumTargetMag: number[] | null;
  sumTargetPhase: number[] | null;
  perBandTarget: Array<{mag, phase} | null>;

  // НОВОЕ
  sumCorrectedMag: number[] | null;
  sumCorrectedPhase: number[] | null;
  perBandCorrected: Array<{mag, phase} | null>;
  /** Признак какой ветки в corrected sum: coherent (с phase) или power-sum. */
  correctedCoherent: boolean;
}

export async function evaluateSum(bands, options): Promise<SumEvalResult> {
  const freq = options?.freq ?? buildCommonGrid(bands);

  // ... existing target evaluation ...

  // 4. Per-band corrected через evaluateBandFull
  // (он уже даёт correctedMag/Phase с правильной phase reconstruction)
  const perBandResults = await Promise.all(
    bands.map(b => evaluateBandFull({ band: b }))
  );

  const perBandCorrected: Array<{mag, phase} | null> = [];
  const correctedDataForSum: Array<{mag, phase, sign, delay} | null> = [];
  let anyMissingPhase = false;

  for (let i = 0; i < bands.length; i++) {
    const r = perBandResults[i];
    if (!r.correctedMag) {
      perBandCorrected.push(null);
      correctedDataForSum.push(null);
      continue;
    }
    // Resample correctedMag и correctedPhase на общую freq grid
    const resampled = await resampleOntoGrid(
      r.freq, r.correctedMag, r.correctedPhase ?? null, freq
    );
    if (!resampled.mag) {
      perBandCorrected.push(null);
      correctedDataForSum.push(null);
      continue;
    }
    perBandCorrected.push({
      mag: resampled.mag,
      phase: resampled.phase ?? new Array(freq.length).fill(0),
    });

    if (!resampled.phase) anyMissingPhase = true;
    correctedDataForSum.push({
      mag: resampled.mag,
      phase: resampled.phase ?? new Array(freq.length).fill(0),
      sign: bands[i].inverted ? -1 : 1,
      delay: bands[i].alignmentDelay ?? 0,
    });
  }

  // 5. Sum: coherent если все имеют phase, иначе power-sum
  let sumCorrectedMag: number[] | null = null;
  let sumCorrectedPhase: number[] | null = null;
  let correctedCoherent = true;

  if (correctedDataForSum.some(d => d !== null)) {
    if (!anyMissingPhase) {
      const sum = coherentSum(freq, correctedDataForSum);
      sumCorrectedMag = sum?.mag ?? null;
      sumCorrectedPhase = sum?.phase ?? null;
    } else {
      // Power-sum (без polarity, без phase)
      sumCorrectedMag = powerSum(
        correctedDataForSum.filter(d => d !== null).map(d => d!.mag)
      );
      sumCorrectedPhase = null;
      correctedCoherent = false;
    }
  }

  return {
    freq,
    sumTargetMag, sumTargetPhase, perBandTarget,
    sumCorrectedMag, sumCorrectedPhase, perBandCorrected,
    correctedCoherent,
  };
}
```

`resampleOntoGrid` остаётся **без extension** (просто resample c
fence -200 dB вне native — как сейчас в коде после прошлых правок).
Никакой target extension магии.

### 2. renderSumModeNew — добавить corrected curves

После target секции:

```typescript
// Σ Corrected
if (showMag && result.sumCorrectedMag) {
  const label = result.correctedCoherent ? "Σ corr" : "Σ corr (incoh)";
  uSeries.push({
    label, stroke: SUM_CORRECTED_COLOR, width: 3, scale: "mag",
  });
  uData.push(result.sumCorrectedMag);
  legend.push({
    label, color: SUM_CORRECTED_COLOR, dash: false,
    visible: true, seriesIdx: sIdx, category: "corrected",
  });
  sIdx++;
  if (showPhase && result.correctedCoherent && result.sumCorrectedPhase) {
    uSeries.push({
      label: "Σ corr °", stroke: SUM_CORRECTED_COLOR,
      width: 1.5, dash: [4, 4], scale: "phase",
    });
    uData.push(wrapPhase(result.sumCorrectedPhase));
    legend.push({
      label: "Σ corr °", color: SUM_CORRECTED_COLOR, dash: true,
      visible: true, seriesIdx: sIdx, category: "corrected",
    });
    sIdx++;
  }
}

// Per-band corrected (visible:false)
for (let i = 0; i < bands.length; i++) {
  const pb = result.perBandCorrected[i];
  if (!pb) continue;
  const cf = bandColorFamily(bands[i].color);
  uSeries.push({
    label: `${bands[i].name} corr`,
    stroke: cf.corrected, width: 1.5, scale: "mag",
  });
  uData.push(pb.mag);
  legend.push({
    label: "Corrected", color: cf.corrected, dash: false,
    visible: false, seriesIdx: sIdx, category: "corrected",
  });
  sIdx++;
  // phase per-band — пока не добавлять в legend, не критично
}
```

### 3. Vitest

```typescript
describe("evaluateSum — corrected", () => {
  it("two bands with measurement → coherent corrected sum");
  it("one band without phase → power-sum fallback, correctedCoherent=false");
  it("all bands without measurement → sumCorrectedMag=null");
  it("polarity inversion in corrected → cancellation");
});
```

### 4. Bump

- `src-tauri/tauri.conf.json` → `0.1.140`.
- `src-tauri/src/lib.rs` startup → b140.3.1.
- `src/lib/version.ts` → b140.3.1.

## Acceptance

1. На 5wayNew в New SUM кроме Σ Target теперь видна **Σ Corrected**.
2. Per-band Corrected кривые в legend (visible:false по умолчанию).
3. При клике на per-band corrected — кривая полосы рисуется.
4. Уровни — какие получаются из чистой coherent суммы measurement+PEQ+XO.
5. Existing 64+ vitest PASS + 4 новых corrected.

## Что НЕ делать

- Не добавлять Σ Measurement (это b140.3.2).
- Не добавлять SUM IR/Step.
- Не нормализовать Σ Corrected к Σ Target (никакого corrOffset).
- Не extend corrected вне native range.

## End-of-prompt автозапуск dev

```
osascript -e 'tell application "PhaseForge" to quit' 2>/dev/null || true
pkill -9 -f -i "phaseforge" 2>/dev/null || true
pkill -9 -f "tauri dev" 2>/dev/null || true
pkill -9 -f "tauri-driver" 2>/dev/null || true
sleep 1
lsof -ti:1420 | xargs kill -9 2>/dev/null || true
lsof -ti:5173 | xargs kill -9 2>/dev/null || true
sleep 1
cd /Users/olegryzhikov/phaseforge && nohup cargo tauri dev > /tmp/phaseforge-dev.log 2>&1 &
```

Сообщить что запущен.

## Правила

- Один коммит: `feat: corrected curves in New SUM (b140.3.1)` + Co-Authored-By.
- Без нарратива.

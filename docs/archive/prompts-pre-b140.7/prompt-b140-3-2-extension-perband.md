# Промт для Code: b140.3.2 — extension логика в evaluateBandFull, visible на band view

Текущий билд: 0.1.0-b140.3.1.6 → bump до 0.1.0-b140.3.2.

## Самооценка эффективности

| Критерий | Оценка | Комментарий |
|---|---|---|
| Размер | ⚠️ средний | Расширение BandEvalResult + UI rendering + чистка evaluateSum |
| Pre-flight audit | ✅ | Логика extension уже отлажена в b140.3.1.5 |
| Гипотезы без данных | ✅ | DRY refactor, нет новой логики |
| Учёт уроков | ✅ | Single point of truth — extension только в одном месте |

## Контекст

Сейчас extension через target+Hilbert делается в `resampleOntoGrid`
внутри `evaluateSum`. Это значит SUM показывает extended данные, но
band view — только native. Пользователь не видит насколько realtimes
работает измерение vs где synthesis.

Перенос extension в `evaluateBandFull`:
- Single point of truth — одна функция-источник.
- Band view рисует extended faded цветом — UX cue про synthetic data.
- evaluateSum упрощается — просто складывает уже extended per-band.

## Что нужно сделать

### 1. Расширить `BandEvalResult`

В `src/lib/band-evaluator.ts`:

```typescript
export interface BandEvalResult {
  freq: number[];
  measurementMag: number[] | null;       // native, без extension
  measurementPhase: number[] | null;
  /** Границы реальных measurement данных. null если measurement нет. */
  nativeRange: [number, number] | null;

  // НОВОЕ: extended кривые (target+Hilbert вне native range)
  extendedMeasurementMag: number[] | null;
  extendedMeasurementPhase: number[] | null;

  // ... existing target / corrected fields
  extendedCorrectedMag: number[] | null;
  extendedCorrectedPhase: number[] | null;

  refLevel, peqMag, peqPhase, ... // как сейчас
}
```

### 2. Реализация extension в `evaluateBandFull`

После вычисления `measurementMag/Phase` и `targetMag/Phase`:

```typescript
let extendedMeasurementMag: number[] | null = null;
let extendedMeasurementPhase: number[] | null = null;
let nativeRange: [number, number] | null = null;

if (measurement && targetMag) {
  nativeRange = [measurement.freq[0], measurement.freq[measurement.freq.length - 1]];

  // Compute extension через хелпер из b140.3.1.5
  const extended = await computeExtension(
    measurement.freq, measurementMag, measurementPhase,
    freq, targetMag,  // target на ЭТОМ же freq grid
  );
  extendedMeasurementMag = extended.mag;
  extendedMeasurementPhase = extended.phase;
}

// Аналогично для corrected
let extendedCorrectedMag: number[] | null = null;
let extendedCorrectedPhase: number[] | null = null;
if (correctedMag && targetMag) {
  const extended = await computeExtension(
    measurement.freq, correctedMag, correctedPhase,
    freq, targetMag,
  );
  extendedCorrectedMag = extended.mag;
  extendedCorrectedPhase = extended.phase;
}
```

`computeExtension` — вынесенная логика из `resampleOntoGrid`
(b140.3.1.5):
- Magnitude через target shape + boundary offset.
- Phase через Hilbert от extended_mag + boundary offset.

### 3. Render extended кривых на band view (faded)

В `renderBandMode`:

```typescript
// Existing: native measurement (solid color)
if (showMag && result.measurement) {
  uSeries.push({ ..., stroke: cf.meas, width: 1.5 });
  uData.push(result.measurement.magnitude);
  legend.push({ label: "Measurement", ... });
}

// НОВОЕ: extended portion (faded)
if (showMag && result.extendedMeasurementMag && result.nativeRange) {
  const [fLo, fHi] = result.nativeRange;
  // Только bins ВНЕ native range. В native range null чтобы не дублировать
  const extendedOnly = result.freq.map((f, j) =>
    f < fLo || f > fHi ? result.extendedMeasurementMag![j] : null
  );
  uSeries.push({
    label: "Measurement (ext)",
    stroke: hexWithAlpha(cf.meas, 0.35),  // faded
    width: 1.5,
    dash: [3, 3],  // опционально dotted для отличия
    scale: "mag",
  });
  uData.push(extendedOnly);
  legend.push({
    label: "Meas (ext)", color: hexWithAlpha(cf.meas, 0.35),
    dash: true, visible: true, seriesIdx: sIdx, category: "measurement",
  });
}
```

`hexWithAlpha(color, alpha)` — хелпер для применения opacity к stroke.
Если uPlot не поддерживает alpha напрямую — использовать rgba string.

Аналогично для **phase** measurement extended если showPhase.
И для **corrected** extended (после secondary curves в renderBandMode).

### 4. Упростить evaluateSum

Удалить `extensionTargetMag` опцию из `resampleOntoGrid` — extension
теперь только в evaluateBandFull. `resampleOntoGrid` возвращает
fence -200 dB вне range (как было до b140.3.1.5).

В `evaluateSum`:

```typescript
for (let i = 0; i < bands.length; i++) {
  const r = perBandResults[i];
  // Используем EXTENDED данные если есть, иначе native с fence
  const sourceMag = r.extendedCorrectedMag ?? r.correctedMag;
  const sourcePhase = r.extendedCorrectedPhase ?? r.correctedPhase;

  if (!sourceMag) {
    perBandCorrected.push(null);
    correctedDataForSum.push(null);
    continue;
  }
  // sourceMag уже на per-band freq grid (= measurement.freq)
  // Resample на common grid (fence только если нет extension)
  const resampled = await resampleOntoGrid(
    r.freq, sourceMag, sourcePhase ?? null, freq
  );
  // ... per-band normalize, limiter, push
}
```

### 5. Перенос computeExtension

Логика которая сейчас в `resampleOntoGrid` для `extensionTargetMag` —
вынести в отдельную функцию `computeExtension(srcFreq, srcMag, srcPhase, targetFreq, targetMag)`.

Возвращает `{ mag, phase }` где:
- В bins внутри src range — original src values интерполированные на targetFreq.
- В bins вне — target_mag + offset / Hilbert + offset.

Это та же логика что в b140.3.1.5, просто extracted в функцию.

### 6. Vitest

```typescript
describe("evaluateBandFull — extended fields", () => {
  it("extendedMeasurementMag exists when target is set");
  it("extendedMeasurementMag null when no target");
  it("nativeRange reflects measurement freq range");
  it("extendedCorrectedMag includes PEQ + XO + extension");
});

describe("evaluateSum — uses extended from band", () => {
  it("falls back to fence if extended not available");
  it("polosa с extended → smooth Σ без ступенек");
});
```

### 7. Bump

- `src-tauri/tauri.conf.json` → `0.1.140`.
- `src-tauri/src/lib.rs` startup → b140.3.2.
- `src/lib/version.ts` → b140.3.2.

## Acceptance

1. На band view (одна полоса) видны **две** кривые measurement:
   solid в native range, faded вне native (показывает synthesis).
2. На SUM в New режиме — без ступенек на границах native, как
   и было в b140.3.1.5.
3. existing 64+ vitest + новые на extended PASS.
4. resampleOntoGrid упрощена: extensionTargetMag опция убрана.

## Что НЕ делать

- Не менять алгоритм extension (тот же что в b140.3.1.5).
- Не убирать fence из resampleOntoGrid когда extension не передан —
  это safety fallback.
- Не делать extended для bands без measurement — там ничего нет.

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

## Правила

- Один коммит: `refactor: extension logic moved to evaluateBandFull, visible on band view (b140.3.2)` + Co-Authored-By.
- Без нарратива.

# Промт для Code: b140.3.1.1 — per-band corrected ↔ target normalize

Текущий билд: 0.1.0-b140.3.1 → bump до 0.1.0-b140.3.1.1.

## Самооценка эффективности

| Критерий | Оценка | Комментарий |
|---|---|---|
| Размер | ✅ малый | Один offset вычисление per-band |
| Pre-flight audit | ✅ | Понятен alg: avg(t-c) в HP*1.5..LP*0.7 |
| Гипотезы без данных | ✅ | Standard normalization, physical |
| Учёт уроков | ✅ | Не Σ-level shift, per-band only — без legacy mimicry |

## Контекст

В b140.3.1 Σ Corrected получается из coherent sum per-band corrected
(measurement+PEQ+XO). Уровни могут не совпадать с per-band target,
особенно если measurement не калиброван к ожидаемому SPL.

Решение: каждая corrected кривая выравнивается к **target той же
полосы** в её собственном passband перед суммированием. Не Σ-level
shift — per-band.

## Логика

Для каждой полосы:

1. **Per-band passband:**
   - `pbLow = max(20, HP_freq × 1.5)` или `20` если HP=null
   - `pbHigh = min(20000, LP_freq × 0.7)` или `20000` если LP=null
   - Fallback: если `pbLow >= pbHigh` → `[200, 2000]`

2. **Offset:**
   - `offset = avg(target_mag[j] - corrected_mag[j])` для всех `j` где
     `freq[j] ∈ [pbLow, pbHigh]` И обе кривые finite (не -200 fence)
   - Если bins нет — offset = 0

3. **Apply:**
   - Если `|offset| > 0.01 dB` → `corrected_mag[j] += offset` для всех j
   - Phase не трогаем (offset только magnitude)

4. **Apply order:**
   - После resampling corrected на common grid
   - До coherent sum
   - Per-band corrected, который рендерится в UI, тоже offset'нутый
     (consistency)

## Что нужно сделать

### 1. evaluateSum — per-band normalize

В `src/lib/band-evaluator.ts` после resampling per-band corrected на
common grid:

```typescript
for (let i = 0; i < bands.length; i++) {
  const r = perBandResults[i];
  if (!r.correctedMag) {
    perBandCorrected.push(null);
    correctedDataForSum.push(null);
    continue;
  }
  const resampled = await resampleOntoGrid(
    r.freq, r.correctedMag, r.correctedPhase ?? null, freq
  );
  if (!resampled.mag) {
    perBandCorrected.push(null);
    correctedDataForSum.push(null);
    continue;
  }

  // Per-band normalize к target
  let correctedMag = resampled.mag;
  const pbTarget = perBandTarget[i];
  if (pbTarget) {
    const hp = bands[i].target.high_pass;
    const lp = bands[i].target.low_pass;
    const pbLow = hp ? Math.max(20, hp.freq_hz * 1.5) : 20;
    const pbHigh = lp ? Math.min(20000, lp.freq_hz * 0.7) : 20000;
    const eL = pbLow < pbHigh ? pbLow : 200;
    const eH = pbLow < pbHigh ? pbHigh : 2000;

    let dSum = 0, dN = 0;
    for (let j = 0; j < freq.length; j++) {
      if (freq[j] < eL || freq[j] > eH) continue;
      const t = pbTarget.mag[j];
      const c = correctedMag[j];
      if (!isFinite(t) || !isFinite(c) || c < -150) continue;
      dSum += t - c;
      dN++;
    }
    if (dN > 0) {
      const offset = dSum / dN;
      if (Math.abs(offset) > 0.01) {
        correctedMag = correctedMag.map(v => v + offset);
      }
    }
  }

  perBandCorrected.push({
    mag: correctedMag,
    phase: resampled.phase ?? new Array(freq.length).fill(0),
  });
  correctedDataForSum.push({
    mag: correctedMag,
    phase: resampled.phase ?? new Array(freq.length).fill(0),
    sign: bands[i].inverted ? -1 : 1,
    delay: bands[i].alignmentDelay ?? 0,
  });
  if (!resampled.phase) anyMissingPhase = true;
}
```

UI per-band corrected кривые читают из `perBandCorrected[i].mag` —
он уже offset'нутый, consistency.

### 2. Vitest

```typescript
describe("evaluateSum — per-band corrected normalization", () => {
  it("corrected -3 dB target 0 dB in passband → after offset corrected = 0 dB", async () => {
    // Band: target=flat 0 dB, measurement=flat -3 dB, no PEQ
    // → corrected = -3 dB → after normalize → 0 dB
  });

  it("offset uses per-band passband (HP*1.5..LP*0.7)", async () => {
    // Band: HP=200, LP=2000 → passband 300-1400
    // Target=0 outside, target=+5 inside passband
    // Corrected matches target everywhere → offset based on passband bins
  });

  it("inverted passband (HP*1.5 >= LP*0.7) → fallback 200-2000");

  it("no target → no offset applied");

  it("offset < 0.01 dB → not applied");
});
```

### 3. Bump

- `src-tauri/tauri.conf.json` → `0.1.140`.
- `src-tauri/src/lib.rs` startup → b140.3.1.1.
- `src/lib/version.ts` → b140.3.1.1.

## Acceptance

1. На 5wayNew per-band corrected кривые лежат на уровне per-band
   target в собственном passband каждой полосы.
2. Σ Corrected близок к Σ Target в зоне где система играет.
3. existing 64+ vitest + 5 новых normalize PASS.

## Что НЕ делать

- Не делать Σ-level shift (Σ Corrected к Σ Target post-aggregation).
- Не нормализовать target — он остаётся как есть.
- Не нормализовать measurement — это другой шаг.
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

## Правила

- Один коммит: `feat: per-band corrected ↔ target normalize in passband (b140.3.1.1)` + Co-Authored-By.
- Без нарратива.

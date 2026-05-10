# Промт для Code: b140.3.1.2 — width-aware excess limiter

Текущий билд: 0.1.0-b140.3.1.1 → bump до 0.1.0-b140.3.1.2.

## Самооценка эффективности

| Критерий | Оценка | Комментарий |
|---|---|---|
| Размер | ✅ малый | Одна функция-фильтр + вызов в evaluateSum |
| Pre-flight audit | ✅ | Алгоритм описан в обсуждении, не legacy mimicry |
| Гипотезы без данных | ✅ | Точно сформулировано: ширина region → clip |
| Учёт уроков | ✅ | Решает класс проблем (exclusion zones, disabled PEQ, шум) одной логикой |

## Контекст

В b140.3.1.1 per-band normalize выравнивает corrected к target в
passband. Но при наличии необработанных шумов / отключённых PEQ
полос / exclusion zones внутри passband — average искажается, и
весь уровень полосы поднимается на несколько dB.

Решение: width-aware excess limiter. После normalize ограничиваем
широкие превышения corrected над target. Узкие peaks (1/8 окт) —
оставляем (естественные резонансы), широкие (≥ 1/2 окт) — clip к
target + 1 dB. Между — soft transition.

Применяется в зоне `passband ± 1 октава` каждой полосы. Independent
от причины excess (любой источник).

## Логика

```typescript
const EXCESS_THRESHOLD = 1.0;  // dB — порог excess для region
const NARROW_OCT = 1/8;        // ширина ниже которой не clip
const WIDE_OCT = 1/2;          // ширина выше которой hard clip к +1 dB
```

Алгоритм:
1. Для каждой полосы определить limiter zone:
   - `zoneLow = max(20, pbLow / 2)` (passband −1 окт)
   - `zoneHigh = min(20000, pbHigh × 2)` (passband +1 окт)
   - `pbLow / pbHigh` — те же что в b140.3.1.1 (HP×1.5 / LP×0.7).

2. Найти регионы внутри limiter zone где `corrected[j] − target[j] > EXCESS_THRESHOLD`.

3. Для каждого региона:
   - `widthOct = log2(f_end / f_start)`
   - Если `widthOct ≤ NARROW_OCT` → `clipFactor = 0` (не трогать)
   - Если `widthOct ≥ WIDE_OCT` → `clipFactor = 1` (hard clip к target+1 dB)
   - Между → линейная интерполяция: `clipFactor = (widthOct − 1/8) / (1/2 − 1/8)`

4. Применить clip:
   ```
   newExcess[j] = excess[j] × (1 − clipFactor) + EXCESS_THRESHOLD × clipFactor
   corrected[j] = target[j] + newExcess[j]
   ```

5. Phase не трогаем (только magnitude).

## Что нужно сделать

### 1. Хелпер `limitExcessByWidth` в band-evaluator.ts

```typescript
function limitExcessByWidth(
  freq: number[],
  corrected: number[],
  target: number[],
  hpFreqHz: number | null,
  lpFreqHz: number | null,
): number[] {
  const EXCESS_THRESHOLD = 1.0;
  const NARROW_OCT = 1/8;
  const WIDE_OCT = 1/2;

  // Limiter zone: passband ± 1 octave
  const pbLow = hpFreqHz ? hpFreqHz * 1.5 : 20;
  const pbHigh = lpFreqHz ? lpFreqHz * 0.7 : 20000;
  const zoneLow = Math.max(20, pbLow / 2);
  const zoneHigh = Math.min(20000, pbHigh * 2);

  const result = [...corrected];
  let regionStart = -1;

  const finalize = (start: number, end: number) => {
    const f0 = freq[start];
    const f1 = freq[end];
    const widthOct = Math.log2(f1 / f0);
    let clipFactor: number;
    if (widthOct <= NARROW_OCT) clipFactor = 0;
    else if (widthOct >= WIDE_OCT) clipFactor = 1;
    else clipFactor = (widthOct - NARROW_OCT) / (WIDE_OCT - NARROW_OCT);
    if (clipFactor === 0) return;
    for (let k = start; k <= end; k++) {
      const ex = corrected[k] - target[k];
      const newEx = ex * (1 - clipFactor) + EXCESS_THRESHOLD * clipFactor;
      result[k] = target[k] + newEx;
    }
  };

  for (let j = 0; j < freq.length; j++) {
    const inZone = freq[j] >= zoneLow && freq[j] <= zoneHigh;
    const isExcess = inZone
      && isFinite(corrected[j]) && isFinite(target[j])
      && (corrected[j] - target[j]) > EXCESS_THRESHOLD;

    if (isExcess && regionStart < 0) {
      regionStart = j;
    } else if (!isExcess && regionStart >= 0) {
      finalize(regionStart, j - 1);
      regionStart = -1;
    }
  }
  if (regionStart >= 0) finalize(regionStart, freq.length - 1);

  return result;
}
```

### 2. evaluateSum — apply limiter

В per-band loop после `correctedMag` normalize:

```typescript
// Per-band normalize (b140.3.1.1) — уже есть
// ...

// b140.3.1.2: width-aware excess limit
if (pbTarget) {
  correctedMag = limitExcessByWidth(
    freq, correctedMag, pbTarget.mag,
    bands[i].target.high_pass?.freq_hz ?? null,
    bands[i].target.low_pass?.freq_hz ?? null,
  );
}
```

UI per-band corrected кривые (`perBandCorrected[i].mag`) тоже
limited — consistency между UI и Σ.

### 3. Vitest

```typescript
describe("evaluateSum — width-aware excess limiter", () => {
  it("wide excess (1 octave) clips to target + 1 dB", async () => {
    // Band: target = 0 dB on full range, corrected = +5 dB на 1 oct
    // After limit: corrected ≈ +1 dB в этой зоне
  });

  it("narrow peak (1/8 octave) preserved", async () => {
    // Corrected = +5 dB на 1/8 oct
    // After limit: +5 dB остаётся (clipFactor = 0)
  });

  it("medium width (1/4 oct) soft transition", async () => {
    // Corrected = +5 dB на 1/4 oct
    // After limit: примерно +3 dB (50% между +1 и +5)
  });

  it("excess outside limiter zone (> passband + 1 oct) not clipped");

  it("no target → no clipping");
});
```

### 4. Bump

- `src-tauri/tauri.conf.json` → `0.1.140`.
- `src-tauri/src/lib.rs` startup → b140.3.1.2.
- `src/lib/version.ts` → b140.3.1.2.

## Acceptance

1. На 5wayNew bass corrected больше не задирается на 3-4 dB выше
   target в широкой зоне — limited к target + 1 dB.
2. Локальные узкие резонансы (например room mode на 1/8 окт) не
   тронуты.
3. existing 64+ vitest + 5 новых limiter PASS.

## Что НЕ делать

- Не трогать phase (только magnitude clip).
- Не лимитировать вне `passband ± 1 oct` зоны.
- Не лимитировать **дефицит** (corrected ниже target) — только excess.

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

- Один коммит: `feat: width-aware excess limiter for per-band corrected (b140.3.1.2)` + Co-Authored-By.
- Без нарратива.

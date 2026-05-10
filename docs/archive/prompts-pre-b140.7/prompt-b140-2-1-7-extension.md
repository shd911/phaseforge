# Промт для Code: b140.2.1.7 — physically motivated extension вне native range

Текущий билд: 0.1.0-b140.2.1.6 → bump до 0.1.0-b140.2.1.7.

## Контекст

В b140.2.1.4 fence в `resampleOntoGrid` зануляет (-200 dB) bins вне
native freq диапазона полосы. Это убрало phantom constant clamp от
Rust `interp_single`. Однако создаёт ступеньку в Σ measurement —
полоса резко обрывается на границе native range.

Решение: physically motivated extension. Для каждой полосы вне
native range:
1. **Если есть target** для полосы (HP/LP кроссовер задан) — extension
   через target shape (offset так чтобы соответствовал measurement
   на границе native).
2. **Если target нет** или target = 0 dB на этих частотах — fallback
   на log-linear trend по последним N точкам native data.

Это работает для multi-way (через target HP/LP), full-range без
target (через measured trend) и edge cases.

## Что нужно сделать

### 1. `resampleOntoGrid` принимает targetMagAtTarget

Текущая сигнатура:

```typescript
async function resampleOntoGrid(
  srcFreq, srcMag, srcPhase, targetFreq
): Promise<{ mag, phase }>
```

Расширить:

```typescript
async function resampleOntoGrid(
  srcFreq, srcMag, srcPhase, targetFreq,
  options?: {
    /** Target curve magnitude on targetFreq grid. Used for extension
     *  shape when bins are outside native range. */
    targetMagOnTargetGrid?: number[];
    /** Trend extension если target нет/неинформативен. */
    fallbackToTrend?: boolean;
  }
): Promise<{ mag, phase }>
```

### 2. Логика extension

```typescript
const fLo = srcFreq[0];
const fHi = srcFreq[srcFreq.length - 1];

// Native magnitude на границах
const magAtLo = srcMag[0];
const magAtHi = srcMag[srcMag.length - 1];

// Target offset: target shape на native boundary должен совпадать
// с measurement.
let targetOffsetLo = 0, targetOffsetHi = 0;
if (options?.targetMagOnTargetGrid) {
  // Найти target value на native boundary frequencies
  const idxLoOnTarget = targetFreq.findIndex(f => f >= fLo);
  const idxHiOnTarget = targetFreq.findIndex(f => f > fHi) - 1;
  if (idxLoOnTarget >= 0) {
    targetOffsetLo = magAtLo - options.targetMagOnTargetGrid[idxLoOnTarget];
  }
  if (idxHiOnTarget >= 0) {
    targetOffsetHi = magAtHi - options.targetMagOnTargetGrid[idxHiOnTarget];
  }
}

// Trend slope для fallback (последние ~1/4 октавы native data)
function tailSlopeLow() {
  const fHigh = fLo * 1.19;  // ~1/4 octave
  const idxs: number[] = [];
  for (let i = 0; i < srcFreq.length; i++) {
    if (srcFreq[i] > fHigh) break;
    idxs.push(i);
  }
  if (idxs.length < 2) return 0;
  // Linear fit slope в log2(f) space
  const logF = idxs.map(i => Math.log2(srcFreq[i]));
  const m = idxs.map(i => srcMag[i]);
  return linearFitSlope(logF, m);  // dB per octave
}

// Применить extension
const extended = mag.map((v, i) => {
  const f = targetFreq[i];
  if (f >= fLo && f <= fHi) return v;  // в native range — оставить

  if (f < fLo) {
    if (options?.targetMagOnTargetGrid) {
      // Target shape с offset
      return options.targetMagOnTargetGrid[i] + targetOffsetLo;
    }
    if (options?.fallbackToTrend) {
      const slope = tailSlopeLow();
      return magAtLo + slope * (Math.log2(f) - Math.log2(fLo));
    }
    return -200;  // fence как было
  }

  if (f > fHi) {
    // Аналогично для верхней границы
    // ...
  }
});
```

### 3. evaluateSum — передавать target

Для каждой полосы при вызове resampleOntoGrid передавать
`targetMagOnTargetGrid` если band.target существует. evaluateBandFull
уже вычисляет targetMag — нужно его resamplить на common grid и
передавать.

```typescript
// В evaluateSum, после вычисления per-band:
for (let i = 0; i < bands.length; i++) {
  const r = perBand[i];
  let targetOnCommonGrid: number[] | undefined;
  if (r.targetMag) {
    const resampledTarget = await resampleOntoGrid(r.freq, r.targetMag, null, freq);
    targetOnCommonGrid = resampledTarget.mag ?? undefined;
  }

  // Resample measurement с extension через target
  if (r.measurementMag) {
    const resampled = await resampleOntoGrid(r.freq, r.measurementMag, r.measurementPhase, freq, {
      targetMagOnTargetGrid: targetOnCommonGrid,
      fallbackToTrend: true,
    });
    // ...
  }
}
```

То же для corrected.

### 4. Vitest тесты

```typescript
describe("resampleOntoGrid extension", () => {
  it("supertweeter native [1k, 20k] extended through Gaussian HP target down to 20 Hz", async () => {
    // measurement: flat 0 dB на 1k-20k
    // target: Gaussian HP=1k (rolloff ниже 1k)
    // Expected: на 100 Hz extended ≈ -40 dB (target rolloff value)
    const result = await resampleOntoGrid(
      [1000, 20000], [0, 0], null, [100, 1000, 10000],
      { targetMagOnTargetGrid: [-40, 0, 0] }
    );
    expect(result.mag![0]).toBeCloseTo(-40, 0);  // followed target
  });

  it("no target, fallback to trend slope", async () => {
    // measurement: rolloff 12 dB/oct ниже 200 Hz
    // На 100 Hz должно быть extended на slope: -12 dB
    const result = await resampleOntoGrid(
      [200, 250, 300, 22000], [0, 0, 0, 0], null, [100, 200],
      { fallbackToTrend: true }
    );
    // Slope почти 0 (flat), extended ≈ 0 dB
    expect(result.mag![0]).toBeCloseTo(0, 0);
  });

  it("no target, no fallback — fence -200 (preserves b140.2.1.4 behaviour)", async () => {
    const result = await resampleOntoGrid(
      [200, 22000], [0, 0], null, [100, 200],
      {}
    );
    expect(result.mag![0]).toBeLessThan(-150);
  });
});
```

### 5. Diff test после фикса

Запустить b140.2.1.2 diff на 5wayNew. После фикса Σ measurement
должен быть гладким, без ступенек на границах native range полос.

### 6. Bump

- `src-tauri/tauri.conf.json` → `0.1.140`.
- `src-tauri/src/lib.rs` startup → b140.2.1.7.
- `src/lib/version.ts` → b140.2.1.7.

## Acceptance

1. На 5wayNew Σ Measurement в New SUM **гладкий** в зоне границ native
   диапазонов полос (1220 Hz для Band 5, 70 Hz для Band 3 и т.д.).
2. Phantom constant clamp не возвращается — supertweeter не +54 dB
   на 5 Hz.
3. Visual паритет (или лучше) с Legacy.
4. Existing snapshot тесты могут потребовать обновления.
5. 3 новых vitest теста на extension PASS.

## Что НЕ делать

- Не убирать fence b140.2.1.4 — он стал fallback для случаев без
  target и без trend.
- Не extrapolate phase — phase вне native silenced (нет смысла).

## Тестировать на `.dmg`

Открыть 5wayNew → SUM → New → Σ Measurement плавный, без ступенек.

## Правила

- Один коммит: `feat: physically motivated extension via target / trend (b140.2.1.7)` + Co-Authored-By.
- 7-vector review.
- Без нарратива.

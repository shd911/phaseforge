# Промт для Code: b140.2.2 — полная унификация SUM (SPL + IR)

ТЗ: `docs/TZ-b140-total-rebuild.md`. Аудит legacy: всё в этом промте.
Текущий билд: 0.1.0-b140.2.1.7 → bump до 0.1.0-b140.2.2.

---

## Самооценка эффективности промта (для tracking)

| Критерий | Оценка | Комментарий |
|---|---|---|
| Pre-flight audit сделан | ✅ | Полный аудит legacy SUM SPL+IR проведён до промта |
| Гипотезы без данных | ❌ ноль | Все 4 пробела локализованы по строкам кода |
| Acceptance измеримый автотестами | ✅ | Diff test b140.2.1.2 + новые vitest на каждый fix |
| Учёт прошлых уроков (cascade detection) | ✅ | b140 сделан как тотальный refactor, не точечные правки |
| Защита от регрессий | ✅ | Golden snapshots b139.0 + diff test |
| Что НЕ делать явно | ✅ | Раздел в конце |
| Размер | ⚠️ большой | Один промт = 4 fix'а одновременно. Альтернатива — разделить, но это вернёт каскад |

---

## Контекст

Аудит локализовал 4 пробела в New SUM относительно Legacy. Этот
промт закрывает все четыре одним коммитом. После — Legacy SUM
становится unused (toggle оставляем для compat).

## Pre-flight audit (уже сделан, факты из кода)

**Legacy renderSumMode (FrequencyPlot.tsx:3349-3939):**
- `globalRef` (3476-3491): max passband-avg 200-2000 Hz среди ВСЕХ
  ресемплированных measurement.
- `avgRef` (3691-3693): среднее `refLevels` enabled bands, где
  `refLevels[i] = band.target.reference_level_db + globalRef`.
- Σ target coherent (3714): `st[j] = 20*log10(amp) + avgRef` ← в New
  отсутствует.
- Σ corrected coherent (3760): `20*log10(amp)` без avgRef. Per-band
  corrOffset уже выровнял.
- Σ corrected power-sum (3804-3816): после `10*log10(Σ 10^(m/10))`
  применяется offset = avg(targetSum - corrSum) в 200-2000 Hz, если
  > 0.01 dB.
- zoomCenter (3914-3928): после aggregation все mag series сдвигаются
  на `-globalRef` для отображения в dBr (0 dBr = громкая полоса).

**Legacy SUM IR (renderTimeTab, ~2074-2472):**
- Per-band IR: native freq grid, нормализация на 0 dB peak.
- Common grid для SUM: union of ranges, **min 2048 точек**, без
  extension до 20-20k (отличается от SPL grid).
- Coherent sum в freq domain → `compute_impulse` (IFFT).
- alignment_delay применяется как phase rotation `360*f*delay`.
- polarity (`band.inverted`) применяется как sign flip.

**В New evaluateSum (band-evaluator.ts:677-950):**
- Уже есть: globalRef, buildLogGrid, evaluateBandFull, resampleOntoGrid
  (с fence + extension), corrOffset, coherentSum, powerSum,
  perBandResampled.
- Отсутствует: avgRef в Σ target, post-correction power-sum, SUM IR
  generation, zoomCenter normalization.

---

## Что нужно сделать

### Fix 1: avgRef в Σ target

В `coherentSum` принимать `refOffset?: number`:

```typescript
function coherentSum(freq, bandsData, refOffset?: number): {mag, phase} | null {
  // ... existing ...
  for (let j = 0; j < n; j++) {
    const amp = Math.sqrt(re[j]**2 + im[j]**2);
    mag[j] = amp > 0 ? 20*Math.log10(amp) + (refOffset ?? 0) : -200;
    phase[j] = Math.atan2(im[j], re[j]) * 180 / Math.PI;
  }
}
```

В `evaluateSum`:

```typescript
// avgRef = средний reference level всех enabled targets
const refLevels: number[] = [];
for (let i = 0; i < bands.length; i++) {
  if (bands[i].targetEnabled && perBand[i].targetMag) {
    refLevels.push(perBand[i].refLevel);  // refLevel из BandEvalResult
  }
}
const avgRef = refLevels.length > 0
  ? refLevels.reduce((a,b)=>a+b, 0) / refLevels.length
  : 0;

// Σ target использует avgRef
const targetSum = coherentSum(freq, targetData, avgRef);

// Σ corrected и Σ measurement — БЕЗ avgRef (legacy так)
const correctedSum = coherentSum(freq, correctedData);
const measurementSum = coherentSum(freq, measurementData);
```

### Fix 2: Post-correction для power-sum corrected

После `powerSum(correctedMagsForFallback)`:

```typescript
if (sumCorrectedMag && targetSum?.mag) {
  // Adaptive passband: где target имеет данные (target peak - 20 dB)
  let targetPeak = -Infinity;
  for (let j = 0; j < freq.length; j++) {
    if (isFinite(targetSum.mag[j]) && targetSum.mag[j] > targetPeak) {
      targetPeak = targetSum.mag[j];
    }
  }
  const threshold = targetPeak - 20;

  let dSum = 0, dN = 0;
  for (let j = 0; j < freq.length; j++) {
    const t = targetSum.mag[j];
    const c = sumCorrectedMag[j];
    if (!isFinite(t) || !isFinite(c)) continue;
    if (t < threshold) continue;
    if (c < -150) continue;
    dSum += t - c;
    dN++;
  }
  if (dN > 0) {
    const offset = dSum / dN;
    if (Math.abs(offset) > 0.01) {
      sumCorrectedMag = sumCorrectedMag.map(v => v + offset);
    }
  }
}
```

### Fix 3: SUM IR generation в evaluateSum

Расширить `BandEvalRequest`/`SumEvalRequest`:

```typescript
interface SumEvalOptions {
  // ... existing
  includeSumIr?: boolean;  // генерировать импульс/шаг для Σ
}

interface SumEvalResult {
  // ... existing
  ir?: {
    measurement?: { time, impulse, step };
    target?: { time, impulse, step };
    corrected?: { time, impulse, step };
  };
}
```

Реализация в `evaluateSum`:

```typescript
if (options?.includeSumIr) {
  // Отдельный grid для IR (отличается от SPL!):
  // union of ranges, min 2048 points, без 20-20k extension
  const irMin = Math.min(...bands.map(b => b.measurement?.freq[0] ?? Infinity));
  const irMax = Math.max(...bands.map(b => b.measurement?.freq[len-1] ?? -Infinity));
  const irPts = Math.max(2048, ...bands.map(b => b.measurement?.freq.length ?? 0));
  const irFreq = buildLogGrid(irPts, irMin, irMax);

  // Per-band: resample на irFreq, normalize 0 dB peak, apply polarity & delay
  const irBandData: Array<{mag, phase, sign, delay} | null> = [];
  for (let i = 0; i < bands.length; i++) {
    if (!perBand[i].measurementMag) { irBandData.push(null); continue; }
    const resampled = await resampleOntoGrid(
      perBand[i].freq, perBand[i].measurementMag,
      perBand[i].measurementPhase ?? null, irFreq,
      { extensionTargetMag: /* per-band target on irFreq */, fallbackToTrend: true }
    );
    let peak = -Infinity;
    for (const v of resampled.mag!) if (v > peak) peak = v;
    const offset = -peak;
    const normMag = resampled.mag!.map(v => v + offset);
    irBandData.push({
      mag: normMag, phase: resampled.phase ?? new Array(irFreq.length).fill(0),
      sign: bands[i].inverted ? -1 : 1,
      delay: bands[i].alignmentDelay ?? 0,
    });
  }

  // Coherent sum для SUM measurement IR
  const sumIrMeasMag = ...;  // через coherentSum
  const sumIrMeasPhase = ...;
  // compute_impulse → impulse + step
  const r = await invoke("compute_impulse", { freq: irFreq, magnitude: sumIrMeasMag, phase: sumIrMeasPhase, sampleRate: sr });

  // Аналогично для target и corrected
  ir = { measurement: r, target: ..., corrected: ... };
}
```

### Fix 4: refLevel в SumEvalResult + apply в renderSumModeNew

```typescript
interface SumEvalResult {
  // ... existing
  /** Reference level (= globalRef) для display normalization.
   *  UI должен вычитать из всех mag кривых для отображения в dBr. */
  globalRef: number;
}
```

В `renderSumModeNew`:

```typescript
const result = await evaluateSum(bands, { includeSumIr: false });
const zoomCenter = result.globalRef;

// Применить ко ВСЕМ mag сериям перед push в uData
const targetMag = result.sumTargetMag?.map(v => v - zoomCenter);
const correctedMag = result.sumCorrectedMag?.map(v => v - zoomCenter);
const measurementMag = result.sumMeasurementMag?.map(v => v - zoomCenter);
// и аналогично perBandResampled.measurementMag etc.
```

После сдвига 0 dBr = громкая полоса. Display ось теперь работает корректно.

### Fix 5 (бонус): renderTimeTab SUM ветка → evaluateSum

Если включён переключатель New: `renderTimeTab` для SUM использует
`evaluateSum({ includeSumIr: true })` и читает `result.ir.measurement/
target/corrected`. Удаление inline pipeline (~400 строк) — отдельный
коммит после визуальной верификации.

---

## Тесты

### Vitest

Существующие 174+ тесты могут потребовать обновления (snapshots с
новым avgRef behavior). Обновить `vi --update`, документировать в
commit что это правильное поведение.

Новые тесты:

```typescript
describe("evaluateSum — Σ target with avgRef", () => {
  it("two bands passband=20 dB → Σ target = 26 dB (coherent + avgRef)");
  it("globalRef tracks loudest band");
});

describe("evaluateSum — power-sum post-correction", () => {
  it("incoherent Σ corrected shifts to Σ target in adaptive passband");
});

describe("evaluateSum — SUM IR", () => {
  it("includeSumIr returns ir.measurement/target/corrected");
  it("IR grid has min 2048 points");
  it("alignment_delay rotates per-band phase before sum");
  it("polarity flips per-band sign before sum");
});

describe("evaluateSum — globalRef export", () => {
  it("result.globalRef = max passband-avg");
});
```

### Cargo

Существующие 178+ тестов PASS unchanged.

### Diff test b140.2.1.2

После фикса запустить на 5wayNew:
- Σ measurement: max diff < 0.5 dB
- Σ target: max diff < 0.5 dB (благодаря avgRef)
- Σ corrected: max diff < 0.5 dB

Если что-то > 0.5 dB — diagnostic, не правки.

---

## Bump

- `src-tauri/tauri.conf.json` → `0.1.140`.
- `src-tauri/src/lib.rs` startup → b140.2.2.
- `src/lib/version.ts` → b140.2.2.

## Acceptance

1. Diff test < 0.5 dB на всех Σ кривых.
2. Visual паритет с Legacy на 5wayNew (level, ступеньки, display
   range).
3. SUM IR через `evaluateSum.ir` доступен (для b140.2.3 миграции
   renderTimeTab).
4. existing 174+ vitest + 178+ cargo PASS (с обновлёнными snapshots).
5. Новые тесты PASS.

## Что НЕ делать

- Не мигрировать renderTimeTab SUM на evaluateSum в этом коммите —
  это отдельный коммит после визуальной проверки SUM IR через тесты.
- Не убирать переключатель Legacy/New.
- Не удалять legacy renderSumMode.
- Не трогать peq-optimize, auto-align.

## Тестировать на `.dmg`

После сборки на 5wayNew → SUM → переключатель → визуальный паритет.

## Правила

- Один коммит: `feat: avgRef + power-sum offset + SUM IR + zoomCenter (b140.2.2)` + Co-Authored-By.
- 7-vector review.
- Без нарратива.
- При FAIL diff test — diagnostic, не вторая попытка фикса.

---

## Постамбула: эффективность промтов tracking

После этого этапа Кирилл начинает оценку эффективности каждого
промта. Самооценка перед запуском (см. начало) — first iteration.
После результата — самооценка по факту: что сработало, что нет,
сколько итераций потребовалось до passing.

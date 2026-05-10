# Промт для Code: b140.3.5 — SUM IR/Step через единый источник

Текущий билд: 0.1.0-b140.3.4 → bump до 0.1.0-b140.3.5.

## Самооценка эффективности

| Критерий | Оценка | Комментарий |
|---|---|---|
| Размер | ⚠️ средний | evaluateSum + IR aggregation в frequency domain + миграция renderTimeTab SUM |
| Pre-flight audit | ✅ | Аудит legacy SUM IR + extension механизм отлажены |
| Гипотезы без данных | ✅ | Frequency domain coherent sum — стандарт |

## Контекст

После b140.3.3 / b140.3.4 band IR/Step использует широкий grid с
extension через target+Hilbert. Но SUM IR/Step (`renderTimeTab` SUM
ветка ~lines 2074-2472) — **legacy inline pipeline**, не использует
`evaluateSum`. Subsonic не виден в SUM IR.

Этот промт — миграция SUM IR на единый источник (последний шаг
unified SUM rebuild).

## Подход

Coherent sum **в frequency domain** на общей широкой grid:
1. Common IR freq grid (5 Hz – min(40k, sr/2·0.95)).
2. Per-band: target / extended_measurement / corrected на этом grid.
3. Apply polarity (sign) и alignment_delay (phase rotation `360·f·delay`).
4. Coherent sum в complex domain.
5. `compute_impulse` → impulse + step для каждой категории.

Frequency domain (не time) — потому что:
- alignment_delay естественно через phase rotation, не sample shift.
- Уже есть data в frequency domain (target, corrected, extended_meas).
- Один общий sr — один общий grid.

## Что нужно сделать

### 1. evaluateSum принимает `includeIr`

В `src/lib/band-evaluator.ts`:

```typescript
export interface SumEvalOptions {
  freq?: number[];
  includeIr?: boolean;  // НОВОЕ
}

export interface SumEvalResult {
  // ... existing
  ir?: {
    measurement?: { time: number[]; impulse: number[]; step: number[] };
    target?: { time: number[]; impulse: number[]; step: number[] };
    corrected?: { time: number[]; impulse: number[]; step: number[] };
  };
}
```

### 2. SUM IR aggregation

```typescript
if (options?.includeIr) {
  // Common IR sample rate: max of per-band measurement sr или 48000
  const irSr = Math.max(48000, ...bands
    .map(b => b.measurement?.sample_rate ?? 48000));
  const irFMax = Math.min(40000, irSr / 2 * 0.95);
  const irFreq = buildLogGrid(1024, 5, irFMax);

  // Для каждой категории (measurement / target / corrected)
  // вычисляем per-band response on irFreq, делаем coherent sum

  const irMeasMag = new Float64Array(irFreq.length);
  const irMeasIm  = new Float64Array(irFreq.length);
  const irTgtRe   = new Float64Array(irFreq.length);
  const irTgtIm   = new Float64Array(irFreq.length);
  const irCorrRe  = new Float64Array(irFreq.length);
  const irCorrIm  = new Float64Array(irFreq.length);
  // (Re/Im accumulators)

  for (let i = 0; i < bands.length; i++) {
    const b = bands[i];
    const sign: 1 | -1 = b.inverted ? -1 : 1;
    const delay = b.alignmentDelay ?? 0;

    // Target on irFreq
    if (b.targetEnabled) {
      const tResp = await invoke<TargetResponse>("evaluate_target",
        { target: b.target, freq: irFreq });
      const tPhase = await reconstructTargetPhase(
        irFreq, tResp.phase, b.target.high_pass, b.target.low_pass);
      // Coherent accumulate
      for (let j = 0; j < irFreq.length; j++) {
        const amp = Math.pow(10, tResp.magnitude[j] / 20) * sign;
        const phRad = (tPhase[j] + 360 * irFreq[j] * delay) * Math.PI / 180;
        irTgtRe[j] += amp * Math.cos(phRad);
        irTgtIm[j] += amp * Math.sin(phRad);
      }
    }

    // Measurement on irFreq (extension через target shape)
    if (b.measurement) {
      // extension через target shape (computeExtension) если target есть
      const extMeas = b.targetEnabled
        ? await computeExtension(
            b.measurement.freq, b.measurement.magnitude, b.measurement.phase,
            irFreq, /* targetMag on irFreq from above */)
        : await resampleOntoGrid(
            b.measurement.freq, b.measurement.magnitude,
            b.measurement.phase ?? null, irFreq);
      if (extMeas.mag && extMeas.phase) {
        for (let j = 0; j < irFreq.length; j++) {
          const amp = Math.pow(10, extMeas.mag[j] / 20) * sign;
          const phRad = (extMeas.phase[j] + 360 * irFreq[j] * delay) * Math.PI / 180;
          irMeasRe[j] += amp * Math.cos(phRad);
          irMeasIm[j] += amp * Math.sin(phRad);
        }
      }
    }

    // Corrected on irFreq
    if (b.measurement && b.targetEnabled) {
      // extension measurement + PEQ on irFreq + cross-section on irFreq
      // (тот же расчёт что в b140.3.4 но per-band)
      // ... compute extMeas, irPeq, irXs ...
      const corrMag = extMeas.mag.map((m, j) => m + irPeqMag[j] + irXsMag[j]);
      const corrPhase = (extMeas.phase ?? zeros).map((p, j) =>
        p + irPeqPhase[j] + irXsPhase[j]);
      // Apply per-band normalize (как в b140.3.1.1) и width-aware limit
      // ... (опционально, для consistency с SPL view)
      for (let j = 0; j < irFreq.length; j++) {
        const amp = Math.pow(10, corrMag[j] / 20) * sign;
        const phRad = (corrPhase[j] + 360 * irFreq[j] * delay) * Math.PI / 180;
        irCorrRe[j] += amp * Math.cos(phRad);
        irCorrIm[j] += amp * Math.sin(phRad);
      }
    }
  }

  // Convert sums to magnitude/phase
  const toMagPhase = (re: Float64Array, im: Float64Array) => {
    const mag: number[] = [];
    const phase: number[] = [];
    for (let j = 0; j < re.length; j++) {
      const amp = Math.sqrt(re[j] ** 2 + im[j] ** 2);
      mag.push(amp > 0 ? 20 * Math.log10(amp) : -200);
      phase.push(Math.atan2(im[j], re[j]) * 180 / Math.PI);
    }
    return { mag, phase };
  };

  const ir: SumEvalResult["ir"] = {};
  const measMP = toMagPhase(irMeasRe, irMeasIm);
  if (measMP.mag.some(v => v > -150)) {
    const r = await invoke<{time, impulse, step}>("compute_impulse",
      { freq: irFreq, magnitude: measMP.mag, phase: measMP.phase, sampleRate: irSr });
    ir.measurement = { time: r.time, impulse: r.impulse, step: r.step };
  }

  const tgtMP = toMagPhase(irTgtRe, irTgtIm);
  if (tgtMP.mag.some(v => v > -150)) {
    const r = await invoke<{time, impulse, step}>("compute_impulse",
      { freq: irFreq, magnitude: tgtMP.mag, phase: tgtMP.phase, sampleRate: irSr });
    ir.target = { time: r.time, impulse: r.impulse, step: r.step };
  }

  const corrMP = toMagPhase(irCorrRe, irCorrIm);
  if (corrMP.mag.some(v => v > -150)) {
    const r = await invoke<{time, impulse, step}>("compute_impulse",
      { freq: irFreq, magnitude: corrMP.mag, phase: corrMP.phase, sampleRate: irSr });
    ir.corrected = { time: r.time, impulse: r.impulse, step: r.step };
  }

  result.ir = ir;
}
```

Псевдокод — точная реализация на усмотрение Code, главное идея.

### 3. renderTimeTab SUM ветка → evaluateSum.ir

В `src/components/FrequencyPlot.tsx`, SUM ветка `renderTimeTab`
(~lines 2074-2472):

```typescript
// SUM mode IR/Step
if (sumMode()) {
  if (sumModeSignal() === "new") {
    // НОВОЕ: единый источник
    const sumResult = await evaluateSum(allBands, { includeIr: true });
    if (!sumResult.ir) return;

    // Render через .impulse / .step / .time для каждой категории
    if (sumResult.ir.measurement) renderIrCurve("Σ Measurement", sumResult.ir.measurement, ...);
    if (sumResult.ir.target) renderIrCurve("Σ Target", sumResult.ir.target, ...);
    if (sumResult.ir.corrected) renderIrCurve("Σ Corrected", sumResult.ir.corrected, ...);
    // ... остальная логика
  } else {
    // legacy inline pipeline (как было)
    // ... existing 400 lines ...
  }
}
```

Inline legacy остаётся — переключатель Legacy/New работает.

### 4. Vitest

```typescript
describe("evaluateSum — IR with unified source", () => {
  it("includeIr=false → no ir field");
  it("includeIr=true returns ir.measurement/target/corrected");
  it("subsonic toggle changes Σ target IR");
  it("PEQ changes Σ corrected IR");
  it("polarity inversion → cancellation in Σ measurement IR");
});
```

### 5. Bump

- `src-tauri/tauri.conf.json` → `0.1.140`.
- `src-tauri/src/lib.rs` startup → b140.3.5.
- `src/lib/version.ts` → b140.3.5.

## Acceptance

1. На SUM IR/Step в New режиме включение subsonic меняет Σ target и
   Σ corrected impulse/step.
2. Legacy SUM IR (`sumModeSignal === "legacy"`) работает как раньше.
3. existing 64+ vitest + 5 новых SUM IR PASS.

## Что НЕ делать

- Не удалять legacy SUM IR (переключатель остаётся).
- Не менять band IR (он уже мигрирован в b140.3.3/3.4).

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

- Один коммит: `feat: SUM IR/Step via unified evaluateSum source (b140.3.5)` + Co-Authored-By.
- Без нарратива.

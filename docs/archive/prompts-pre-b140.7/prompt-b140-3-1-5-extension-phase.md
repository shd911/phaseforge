# Промт для Code: b140.3.1.5 — extension через target + Hilbert phase

Текущий билд: 0.1.0-b140.3.1.4 → bump до 0.1.0-b140.3.1.5.

## Самооценка эффективности

| Критерий | Оценка | Комментарий |
|---|---|---|
| Размер | ⚠️ средний | resample logic + Hilbert invoke + boundary offset |
| Pre-flight audit | ✅ | Логика чётко описана, math стандартная |
| Гипотезы без данных | ✅ | Hilbert + offset — DSP стандарт |
| Учёт уроков | ✅ | b140.2.1.7 phase=0 урок учтён, теперь Hilbert |

## Контекст

Замеры делаются не на полном диапазоне (защита драйвера). После
фильтрации (Gaussian HP и т.д.) этот диапазон становится безопасным
и реально воспроизводится. Need: extension measurement за native
range через target shape для magnitude и Hilbert reconstruction для
phase.

## Алгоритм

### Magnitude extension (просто):
```
mag_offset = measurement_mag[native_boundary] − target_mag[native_boundary]
extended_mag[j] = measurement_mag[j]                        if j в native
                = target_mag[j] + mag_offset                if j вне native
```

### Phase extension (Hilbert):
```
1. recon_phase = Hilbert(extended_mag)  // через invoke compute_minimum_phase
2. На каждой границе native (low и high):
   phase_offset_low  = measurement_phase[low_boundary]  − recon_phase[low_boundary]
   phase_offset_high = measurement_phase[high_boundary] − recon_phase[high_boundary]
3. extended_phase[j] = measurement_phase[j]                       if j в native
                     = recon_phase[j] + phase_offset_low/high     if j вне native
```

Phase offset разный на нижней и верхней границе — для smooth
continuation на каждой стороне.

Если у полосы НЕТ phase (measurement.phase = null) → phase extension
даёт null (как было в b140.2.1.4 fence).

## Что нужно сделать

### 1. Расширить resampleOntoGrid

В `src/lib/band-evaluator.ts`:

```typescript
async function resampleOntoGrid(
  srcFreq: number[],
  srcMag: number[],
  srcPhase: number[] | null,
  targetFreq: number[],
  options?: {
    /** Target magnitude on targetFreq (для extension shape). */
    extensionTargetMag?: number[];
  },
): Promise<{ mag: number[] | null; phase: number[] | null }> {
  // 1. Standard interp through Rust (constant boundary clamping)
  const [, mag, phase] = await invoke(...);

  const fLo = srcFreq[0];
  const fHi = srcFreq[srcFreq.length - 1];

  if (!options?.extensionTargetMag) {
    // No target → fence -200 dB outside native (как в b140.2.1.4)
    const fenced = mag.map((v, i) => {
      const f = targetFreq[i];
      return f < fLo || f > fHi ? -200 : v;
    });
    return { mag: fenced, phase: phase /* fenced или null */ };
  }

  // 2. Find boundary indices on targetFreq (closest to fLo / fHi)
  const idxLo = targetFreq.findIndex(f => f >= fLo);
  let idxHi = -1;
  for (let i = targetFreq.length - 1; i >= 0; i--) {
    if (targetFreq[i] <= fHi) { idxHi = i; break; }
  }
  if (idxLo < 0 || idxHi < 0) return { mag, phase };  // edge case

  // 3. Magnitude extension
  const tgtAtLo = options.extensionTargetMag[idxLo];
  const tgtAtHi = options.extensionTargetMag[idxHi];
  const measAtLo = mag[idxLo];
  const measAtHi = mag[idxHi];
  const magOffsetLo = measAtLo - tgtAtLo;
  const magOffsetHi = measAtHi - tgtAtHi;

  const extendedMag = mag.map((v, i) => {
    const f = targetFreq[i];
    if (f >= fLo && f <= fHi) return v;  // native
    if (f < fLo) return options.extensionTargetMag![i] + magOffsetLo;
    return options.extensionTargetMag![i] + magOffsetHi;  // f > fHi
  });

  // 4. Phase extension via Hilbert reconstruction
  let extendedPhase: number[] | null = null;
  if (phase) {
    const reconPhase = await invoke<number[]>("compute_minimum_phase", {
      freq: targetFreq,
      magnitude: extendedMag,
    });
    const phaseOffsetLo = phase[idxLo] - reconPhase[idxLo];
    const phaseOffsetHi = phase[idxHi] - reconPhase[idxHi];
    extendedPhase = phase.map((v, i) => {
      const f = targetFreq[i];
      if (f >= fLo && f <= fHi) return v;  // native
      if (f < fLo) return reconPhase[i] + phaseOffsetLo;
      return reconPhase[i] + phaseOffsetHi;
    });
  }

  return { mag: extendedMag, phase: extendedPhase };
}
```

### 2. evaluateSum — передавать target для extension

После вычисления per-band target (уже на common grid):

```typescript
// Per-band corrected resample с extension через target
const resampled = await resampleOntoGrid(
  r.freq, r.correctedMag, r.correctedPhase ?? null, freq,
  { extensionTargetMag: perBandTarget[i]?.mag }  // target на common grid
);
```

То же для measurement (если будет добавлен в b140.3.2).

### 3. Vitest тесты

```typescript
describe("resampleOntoGrid — extension via target + Hilbert phase", () => {
  it("magnitude follows target shape outside native", async () => {
    // measurement: flat 0 dB на 1k-20k
    // target: Gaussian HP=1k (rolloff ниже 1k)
    // extension: на 100 Hz → ≈ target_mag + offset
  });

  it("phase smoothly continues from native via Hilbert", async () => {
    // На native boundary phase original
    // Вне native — recon + offset, нет скачка на boundary
  });

  it("no extensionTargetMag → fence -200 dB (b140.2.1.4 behaviour)", async () => {
    // Без target — bins вне native = -200
  });

  it("phase=null in measurement → phase=null in extended", async () => {
    // Power-sum case unchanged
  });
});
```

### 4. Bump

- `src-tauri/tauri.conf.json` → `0.1.140`.
- `src-tauri/src/lib.rs` startup → b140.3.1.5.
- `src/lib/version.ts` → b140.3.1.5.

## Acceptance

1. На 5wayNew Σ Corrected плавный без ступенек на границах native
   ranges каждой полосы.
2. Per-band corrected продолжается через target shape вне native.
3. Phase plot — без разрывов на границах native, плавный continuation.
4. existing 64+ vitest + 4 новых extension PASS.

## Что НЕ делать

- Не использовать target phase напрямую — только Hilbert от extended
  magnitude.
- Не extend measurement если у полосы нет target (нет HP/LP).
- Не trogat fence behaviour когда extensionTargetMag не передан
  (b140.2.1.4 fallback).

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

- Один коммит: `feat: extension via target + Hilbert phase reconstruction (b140.3.1.5)` + Co-Authored-By.
- Без нарратива.

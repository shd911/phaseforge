# Промт для Code: b140.2.1.4 — фикс extrapolation в resampleOntoGrid

Текущий билд: 0.1.0-b140.2.1.3 → bump до 0.1.0-b140.2.1.4.

## Контекст

Diagnostic подтвердил: `interp_single` в Rust делает constant
boundary extrapolation. Per-band measurement Band 5 (supertweeter,
native ≥ 1220 Hz) после ресемпла на общую сетку 20 Hz – 40 kHz
имеет phantom +53.78 dB на 5–1000 Hz (повтор y_data[0]).

В coherent sum фантомный вклад складывается с реальным
woofer'ом → ачх пищалки видна на басу.

Фикс: точечно в `resampleOntoGrid` (TypeScript) — занулять (-200 dB)
все bins вне исходного диапазона. Rust interp не трогаем (используется
в legitimate paths).

## Что нужно сделать

### 1. `resampleOntoGrid` в `src/lib/band-evaluator.ts`

Найти функцию (создаёт ресемпл per-band данных на common grid).
После получения mag/phase от Rust добавить fence:

```typescript
async function resampleOntoGrid(
  srcFreq: number[],
  srcMag: number[],
  srcPhase: number[] | null,
  targetFreq: number[],
): Promise<{ mag: number[] | null; phase: number[] | null }> {
  // ... existing invoke interpolate_log

  const fLo = srcFreq[0];
  const fHi = srcFreq[srcFreq.length - 1];

  // Mark out-of-range bins as silent (-200 dB amplitude → ≈ 0 в complex sum)
  const fenced_mag = mag.map((v, i) => {
    const f = targetFreq[i];
    return (f < fLo || f > fHi) ? -200 : v;
  });

  // Phase out-of-range можно тоже занулить — на amplitude=0 phase
  // безразлична, но 0 явное.
  const fenced_phase = phase ? phase.map((v, i) => {
    const f = targetFreq[i];
    return (f < fLo || f > fHi) ? 0 : v;
  }) : null;

  return { mag: fenced_mag, phase: fenced_phase };
}
```

### 2. Проверить все callsites resampleOntoGrid

Grep на функцию. Используется в `evaluateSum` для measurement,
target и corrected данных. Fence применяется ко всем — это правильно
(target тоже не должен экстраполироваться за пределы измерения,
если он привязан к measurement freq).

Однако: `evaluateBandFull` для одной полосы возвращает `targetMag` на
measurement grid (или standalone grid для FIR). Эти случаи **не
проходят через resampleOntoGrid** — fence их не затронет.
Проверить это явно перед коммитом.

### 3. Vitest тесты

Добавить в `src/lib/__tests__/evaluate-sum.test.ts`:

```typescript
describe("evaluateSum — extrapolation fence", () => {
  it("band with limited freq range does not contribute outside it", async () => {
    // Band 1: full range 20-20k
    // Band 2: limited 5k-20k (supertweeter)
    // Common grid: 20-20k
    // На 100 Hz Band 2 не должен вкладываться в Σ measurement.

    const bands = makeBandsWithMixedRanges([[20, 20000], [5000, 20000]]);
    const result = await evaluateSum(bands, {});

    const idx100 = result.freq.findIndex(f => f >= 100);
    // Σ measurement на 100 Hz = только Band 1 (Band 2 силенс)
    // Если Band 1 = 0 dB → Σ = 0 dB (один драйвер)
    expect(result.sumMeasurementMag![idx100]).toBeCloseTo(0, 1);

    const idx10k = result.freq.findIndex(f => f >= 10000);
    // На 10k обе полосы вносят → Σ = +6 dB (coherent двух flat)
    expect(result.sumMeasurementMag![idx10k]).toBeCloseTo(6.02, 1);
  });
});
```

### 4. Diff test после фикса

Запустить b140.2.1.2 diff test. После фикса:
- Σ measurement: max diff < 0.5 dB на всех частотах.
- Σ target: уже ≈ < 0.5 dB после b140.2.1.3 globalRef.
- Σ corrected: < 0.5 dB.

Если расхождение остаётся на специфических частотах — diagnostic.

### 5. Bump

- `src-tauri/tauri.conf.json` → `0.1.140`.
- `src-tauri/src/lib.rs` startup → b140.2.1.4.
- `src/lib/version.ts` → b140.2.1.4.

## Acceptance

1. На 5wayNew в New SUM ачх пищалки видна **только** в её native
   диапазоне (≥ 1220 Hz), на басу не появляется.
2. Visual паритет с Legacy на всех 4 кривых (Σ Meas/Target/Corrected
   и phase).
3. Existing 167+ vitest + 178+ cargo PASS.
4. Новый extrapolation fence тест PASS.

## Что НЕ делать

- Не трогать Rust interp_single / interp_1d (используется широко,
  риск регрессии).
- Не менять renderBandMode — там native grid каждой полосы, fence
  не нужен.
- Не трогать SUM IR (b140.2.2).

## Тестировать на `.dmg`

Открыть 5wayNew → SUM → New режим → проверить что Band 5 magnitude
не отображается на 5–120 Hz.

## Правила

- Один коммит: `fix: silence out-of-range bins in resampleOntoGrid (b140.2.1.4)` + Co-Authored-By.
- 7-vector review.
- Без нарратива.

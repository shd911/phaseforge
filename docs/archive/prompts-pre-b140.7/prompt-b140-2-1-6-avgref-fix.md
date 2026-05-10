# Промт для Code: b140.2.1.6 — avgRef + incoherent normalization parity

Текущий билд: 0.1.0-b140.2.1.5 → bump до 0.1.0-b140.2.1.6.

## Контекст

Аудит legacy renderSumMode локализовал главный источник 20–30 dB
сдвига между Legacy и New SUM.

**Diff #1 (главный):** Legacy в coherent sum для Σ target добавляет
`+ avgRef` (среднее reference level всех enabled bands) после
`20·log10(amplitude)` (FrequencyPlot.tsx:3691-3714). New этого не
делает (band-evaluator.ts:564). Поскольку avgRef ~ passband-average
полос (85–90 dB для типичной системы), весь Σ target в New смещён
вниз на эту величину.

**Diff #2:** Legacy для incoherent (power-sum) Σ corrected
нормализует к Σ target в полосе 200–2000 Гц (3803-3816). New не
нормализует — power sum возвращается без offset.

## Что нужно сделать

### 1. `coherentSum` принимает refOffset

В `src/lib/band-evaluator.ts`, функция `coherentSum`:

```typescript
function coherentSum(
  freq: number[],
  bandsData: Array<{ mag: number[]; phase: number[]; sign: 1 | -1; delay: number } | null>,
  refOffset?: number,  // НОВОЕ: добавляется к dB после 20·log10
): { mag: number[]; phase: number[] } | null {
  // ... existing complex sum ...
  for (let j = 0; j < n; j++) {
    const amplitude = Math.sqrt(re[j] * re[j] + im[j] * im[j]);
    mag[j] = amplitude > 0
      ? 20 * Math.log10(amplitude) + (refOffset ?? 0)
      : -200;
    phase[j] = Math.atan2(im[j], re[j]) * 180 / Math.PI;
  }
  return { mag, phase };
}
```

### 2. `evaluateSum` вычисляет avgRef и передаёт

После per-band evaluation:

```typescript
// avgRef — среднее target reference уровней всех enabled bands
// (порт из renderSumMode:3691-3693)
const refLevels: number[] = [];
for (let i = 0; i < bands.length; i++) {
  if (bands[i].targetEnabled && perBand[i].targetMag) {
    refLevels.push(perBand[i].refLevel);  // refLevel из BandEvalResult
  }
}
const avgRef = refLevels.length > 0
  ? refLevels.reduce((a, b) => a + b, 0) / refLevels.length
  : 0;

// Используется только для Σ target (legacy adds it там)
const targetSum = coherentSum(freq, targetData, avgRef);
```

Для Σ corrected и Σ measurement avgRef **не передаётся** (legacy
не добавляет его там — coherent sum corrected уже paritized через
per-band corrOffset, см. строки 3760-3764 + комментарий "No Σ offset
— per-band normalization already aligns each band").

### 3. Incoherent Σ corrected: нормализация к Σ target по реальной полосе системы

После power-sum fallback. Вместо фиксированного окна 200–2000 Гц
(как в legacy) — adaptive по реальной полосе пропускания: bins где
Σ target magnitude > peak − 20 dB. Работает для sub, full-range,
tweeter-only без отдельных edge cases.

```typescript
if (anyCorrectedMagWithoutPhase) {
  // ... existing power-sum code ...
  sumCorrectedMag = powerSum(...);

  if (sumCorrectedMag && targetSum?.mag) {
    // Найти target peak
    let targetPeak = -Infinity;
    for (let j = 0; j < freq.length; j++) {
      if (isFinite(targetSum.mag[j]) && targetSum.mag[j] > targetPeak) {
        targetPeak = targetSum.mag[j];
      }
    }
    const threshold = targetPeak - 20;  // полоса пропускания системы

    let dSum = 0, dN = 0;
    for (let j = 0; j < freq.length; j++) {
      const t = targetSum.mag[j];
      const c = sumCorrectedMag[j];
      if (!isFinite(t) || !isFinite(c)) continue;
      if (t < threshold) continue;       // вне полосы системы
      if (c < -150) continue;            // silenced fence
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

  sumCorrectedPhase = null;
  coherent = false;
}
```

### 4. Vitest тесты

Существующие тесты могут потребовать корректировки expected values
(где раньше было `+6.02 dB`, теперь `+6.02 + avgRef dB`). Проверить
все snapshot тесты на evaluateSum, обновить если нужно. Это не
регрессия — это правильное поведение.

Новые тесты:

```typescript
describe("evaluateSum — avgRef parity with legacy", () => {
  it("Σ target includes avgRef as additive offset", async () => {
    // 2 полосы с passband ≈ 0 dB → avgRef ≈ 0 → результат +6 dB как раньше
    // 2 полосы с passband ≈ +20 dB → avgRef ≈ 20 → +26 dB
    const bands = makeBandsWithPassband([20, 20]);
    const result = await evaluateSum(bands, {});
    const idx = passbandIdx;
    expect(result.sumTargetMag![idx]).toBeCloseTo(26.02, 1);
  });

  it("Σ corrected coherent does not add avgRef (already paritized)", async () => {
    const bands = makeBandsWithPassband([20, 20]);
    const result = await evaluateSum(bands, {});
    const idx = passbandIdx;
    // Σ corrected at +6 dB above target — НЕ +avgRef
    expect(result.sumCorrectedMag![idx]).toBeCloseTo(26.02, 1);
  });
});

describe("evaluateSum — incoherent power-sum normalization", () => {
  it("incoherent Σ corrected shifts to Σ target in passband", async () => {
    const bands = makeBandsWithPassband([20, 20]);
    bands[0].measurement!.phase = null;
    const result = await evaluateSum(bands, {});
    expect(result.coherent).toBe(false);
    // Power sum +3 dB → shifted to match target (+26 dB)
    const idx = passbandIdx;
    expect(result.sumCorrectedMag![idx]).toBeCloseTo(result.sumTargetMag![idx], 0);
  });
});
```

### 5. Обновить snapshot файлы

После изменения expected values existing snapshots могут разойтись.
Запустить vitest, обновить через `vi --update` если разница
объясняется новой логикой (добавление avgRef). Зафиксировать обновлённые
snapshots с комментарием в commit message.

### 6. Bump

- `src-tauri/tauri.conf.json` → `0.1.140`.
- `src-tauri/src/lib.rs` startup → b140.2.1.6.
- `src/lib/version.ts` → b140.2.1.6.

## Acceptance

1. На 5wayNew в New SUM Σ Target и Σ Corrected уровни совпадают с
   Legacy в пределах 0.5 dB.
2. Per-band кривые из b140.2.1.5 не сломаны.
3. Existing snapshot тесты обновлены (с комментарием почему).
4. Новые тесты на avgRef и incoherent normalization PASS.

## Что НЕ делать

- Не добавлять avgRef к Σ corrected coherent — там legacy не
  добавляет (per-band corrOffset уже выравнивает).
- Не добавлять avgRef к Σ measurement — там legacy не добавляет.
- Не трогать SUM IR (b140.2.2).

## Тестировать на `.dmg`

Открыть 5wayNew → SUM → New → уровни должны совпадать с Legacy.

## Правила

- Один коммит: `fix: avgRef in Σ target + incoherent normalization (b140.2.1.6)` + Co-Authored-By.
- 7-vector review.
- Без нарратива.

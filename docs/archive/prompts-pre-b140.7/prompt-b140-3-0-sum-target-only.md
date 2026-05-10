# Промт для Code: b140.3.0 — чистая SUM, только Σ Target

Текущий билд: 0.1.0-b140.2.2 → bump до 0.1.0-b140.3.0.

---

## Самооценка эффективности промта

| Критерий | Оценка | Комментарий |
|---|---|---|
| Pre-flight audit | ✅ | Полный аудит legacy уже сделан в b140.2.x. Не повторять. |
| Гипотезы без данных | ✅ | Все алгоритмы стандартные (coherent sum), не legacy mimicry |
| Acceptance измеримый | ✅ | Vitest на coherent sum, manual visual проверка только что показывается |
| Учёт прошлых уроков | ✅ | Никакого legacy mimicry — урок b140.2.x ясен |
| Защита от регрессий | ✅ | Legacy переключатель остаётся, чистый pipeline в стороне |
| Что НЕ делать явно | ✅ | Нет measurement, corrected, IR, нормализаций |
| Размер | ✅ малый | Один step, один тип кривой |

---

## Контекст

После b140.2.x каскада (11 итераций без паритета) — выкидываем
накопленный evaluateSum и начинаем с чистого листа. Принципы:

1. **Никакого legacy mimicry.** Не пытаемся воспроизвести avgRef,
   zoomCenter, corrOffset, extension — это магия, которая работала
   для legacy и не унифицируется.
2. **Honest pipeline.** Σ = простая coherent sum target curves
   per-band. Уровни — какие есть.
3. **Маленькие шаги.** Сначала только Σ Target. Потом постепенно
   measurement, corrected, IR — отдельными промтами.

## Шаг 1: только Σ Target

В этом промте — **только**:
- Per-band target curve через evaluate_target.
- Common freq grid.
- Coherent sum c polarity и alignment_delay.
- Отображение Σ Target в New SUM view.

**Что НЕ делаем:**
- ❌ Σ measurement
- ❌ Σ corrected
- ❌ SUM IR/Step
- ❌ avgRef, zoomCenter, corrOffset, extension target/trend, fence
- ❌ Per-band entries в legend для measurement/corrected

## Что нужно сделать

### 1. Переписать `evaluateSum` минимально

В `src/lib/band-evaluator.ts` — заменить текущую `evaluateSum`
(удалить накопленные фиксы) на минимальную:

```typescript
export interface SumEvalResult {
  freq: number[];
  sumTargetMag: number[] | null;
  sumTargetPhase: number[] | null;
}

export interface SumEvalOptions {
  freq?: number[];  // override common grid
}

export async function evaluateSum(
  bands: BandState[],
  options?: SumEvalOptions,
): Promise<SumEvalResult> {
  // 1. Common freq grid: union of all band measurement ranges
  // (или standalone log-grid 5-40000 если ни у кого нет measurement)
  const freq = options?.freq ?? buildCommonGrid(bands);

  // 2. Per-band target evaluation на общей сетке.
  // Target из band.target напрямую через evaluate_target Rust.
  const perBandTargetData: Array<{
    mag: number[]; phase: number[];
    sign: 1 | -1; delay: number;
  } | null> = [];

  for (const band of bands) {
    if (!band.targetEnabled) {
      perBandTargetData.push(null);
      continue;
    }
    const response = await invoke<TargetResponse>("evaluate_target", {
      target: band.target,
      freq,
    });
    // Phase reconstruction для Gaussian min-phase / subsonic
    const phase = await reconstructTargetPhase(
      freq, response.phase, band.target.high_pass, band.target.low_pass,
    );
    perBandTargetData.push({
      mag: response.magnitude,
      phase,
      sign: band.inverted ? -1 : 1,
      delay: band.alignmentDelay ?? 0,
    });
  }

  // 3. Coherent sum
  const sum = coherentSum(freq, perBandTargetData);

  return {
    freq,
    sumTargetMag: sum?.mag ?? null,
    sumTargetPhase: sum?.phase ?? null,
  };
}

function buildCommonGrid(bands: BandState[]): number[] {
  // Union of measurement ranges если есть measurement
  // Иначе standalone 5-40000 Hz, 512 точек
  let fMin = Infinity, fMax = -Infinity;
  for (const b of bands) {
    if (b.measurement) {
      fMin = Math.min(fMin, b.measurement.freq[0]);
      fMax = Math.max(fMax, b.measurement.freq[b.measurement.freq.length - 1]);
    }
  }
  if (!isFinite(fMin)) { fMin = 5; fMax = 40000; }
  return buildLogGrid(512, fMin, fMax);
}
```

`coherentSum` — оставить существующую функцию (она правильная).
`reconstructTargetPhase` — оставить.
`buildLogGrid` — оставить.

### 2. Удалить остальное в evaluateSum

Из старой версии `evaluateSum` удалить (или закомментировать с
TODO):
- globalRef, avgRef computation.
- corrOffset normalization.
- Power-sum fallback.
- perBandResampled.
- includeSumIr / SUM IR generation.
- sumMeasurementMag/Phase, sumCorrectedMag/Phase.
- coherentMeasurement, coherent флаги.

Это можно либо удалить навсегда, либо оставить в отдельном
deprecated файле как reference. На усмотрение Code-сессии. Главное —
чтобы `evaluateSum` была минимальной и понятной.

### 3. Упростить `renderSumModeNew`

В `src/components/FrequencyPlot.tsx` функция `renderSumModeNew`:

```typescript
async function renderSumModeNew(showPhase, showMag, showTarget) {
  const result = await evaluateSum(appState.bands, {});

  const uSeries = [{}];
  const uData: number[][] = [result.freq];
  const legend: LegendEntry[] = [];
  let sIdx = 1;

  if (showTarget && result.sumTargetMag) {
    uSeries.push({
      label: "Σ tgt (New)",
      stroke: SUM_TARGET_COLOR,
      width: 2.5, dash: [8, 4], scale: "mag",
    });
    uData.push(result.sumTargetMag);
    legend.push({
      label: "Σ target (New)", color: SUM_TARGET_COLOR, dash: true,
      visible: true, seriesIdx: sIdx, category: "target",
    });
    sIdx++;

    if (showPhase && result.sumTargetPhase) {
      uSeries.push({
        label: "Σ tgt ° (New)",
        stroke: SUM_TARGET_PHASE_COLOR,
        width: 1.5, dash: [4, 4], scale: "phase",
      });
      uData.push(wrapPhase(result.sumTargetPhase));
      legend.push({
        label: "Σ target ° (New)", color: SUM_TARGET_PHASE_COLOR,
        dash: true, visible: true, seriesIdx: sIdx, category: "target",
      });
      sIdx++;
    }
  }

  // НЕТ measurement, corrected, per-band кривых.
  // Они вернутся в b140.3.1, b140.3.2, etc.

  renderChart({ freq: result.freq, uSeries, uData, hasMeasurements: false, legend });
}
```

Per-band entries для legend grid — пока **убрать**. Каждый шаг
добавляет одну категорию.

### 4. Vitest тесты — переписать минимально

В `src/lib/__tests__/evaluate-sum.test.ts` — удалить все накопленные
тесты (на avgRef, fence, extension, perBandResampled, sumIr и т.д.).
Оставить только новые минимальные:

```typescript
describe("evaluateSum (minimal, b140.3.0) — Σ target only", () => {
  it("returns null for empty bands");
  it("two flat-target bands → coherent sum +6 dB at common passband");
  it("two bands inverted → cancellation");
  it("two bands with delay → phase rotation in result");
  it("targetEnabled=false drops band from sum");
});
```

Старые snapshot файлы — удалить.

### 5. Bump

- `src-tauri/tauri.conf.json` → `0.1.140`.
- `src-tauri/src/lib.rs` startup → b140.3.0.
- `src/lib/version.ts` → b140.3.0.

## Acceptance

1. `evaluateSum` возвращает только `sumTargetMag/Phase`.
2. На 5wayNew в New SUM видна **только** Σ Target кривая (mag и phase).
3. Σ Target — coherent sum target curves каждой полосы. Уровни в
   absolute SPL (не dBr). Может отличаться от Legacy визуально —
   это **намеренно** (никакого avgRef/zoomCenter magic).
4. 5+ vitest тестов PASS.
5. Cargo 178+ unchanged.

## Что НЕ делать

- Не добавлять measurement / corrected / IR в этот промт.
- Не пытаться повторить Legacy уровни магией.
- Не сохранять deprecated код в evaluateSum — лучше чистый файл.
- Не удалять переключатель Legacy/New — Legacy остаётся для
  пользователя.

## Тестировать на `.dmg`

Открыть 5wayNew → SUM → New → должна быть только Σ Target кривая.
Уровень — какой получается из coherent sum reference levels всех
target. Может быть отличный от Legacy.

## Правила

- Один коммит: `refactor: minimal SUM with target only (b140.3.0)` + Co-Authored-By.
- Без нарратива.

## End-of-prompt: автозапуск dev

После коммита автоматически:

```
pkill -f "PhaseForge" 2>/dev/null || true
lsof -ti:1420 | xargs kill -9 2>/dev/null || true
cd /Users/olegryzhikov/phaseforge && nohup cargo tauri dev > /tmp/phaseforge-dev.log 2>&1 &
```

Сообщить пользователю «dev запущен в фоне, логи в /tmp/phaseforge-dev.log».

---

## Постамбула: фактическая оценка после результата

После запуска заполнить:
- Сколько итераций потребовалось до passing visual test?
- Что не предусмотрел?
- Какие edge cases вылезли?

Если паритет с первой попытки — самооценка правильная, можно идти к
b140.3.1 (добавить Σ Measurement).

# Промт для Code: b139.4c — полная унификация всех расчётов

**Тип:** расширение единой сущности + удаление параллельных pipeline.
Bump до 0.1.0-b139.4c.

## Контекст

В b139.4b SPL вкладка частично мигрирована: target фаза приходит из
единой сущности, но **corrected фаза** (красная сплошная на SPL)
до сих пор считается inline в `renderBandMode` через
`addGaussianMinPhase` с условием только на min-phase Gaussian. Для
линейного Gaussian + защитного фильтра реконструкция пропускается —
corrected фаза = 0°, target правильная, расхождение видно глазу.

То же касается импульса/шага corrected кривой и SUM view.

Цель: BandEvaluator отдаёт **всё что нужно для отображения**.
FrequencyPlot читает только из результата, никаких параллельных
invoke цепей.

## Что нужно сделать

### 1. Расширение BandEvalResult

В `src/lib/band-evaluator.ts`:

```typescript
export interface BandEvalResult {
  freq: number[];
  measurementMag: number[] | null;
  measurementPhase: number[] | null;

  targetMag: number[] | null;
  targetPhase: number[] | null;

  peqMag: number[];
  peqPhase: number[];

  combinedTargetMag: number[] | null;        // target + peq
  combinedTargetPhase: number[] | null;

  // НОВЫЕ ПОЛЯ — corrected (то что слышит человек после коррекции)
  correctedMag: number[] | null;             // measurement + peq + xs (filters)
  correctedPhase: number[] | null;           // measurement + peq + xs + Gaussian/subsonic reconstruction

  // НОВЫЕ ПОЛЯ — cross-section (применённые HP/LP фильтры на отдельной кривой)
  crossSectionMag: number[] | null;
  crossSectionPhase: number[] | null;

  refLevel: number;

  fir?: { /* как было */ };

  // НОВЫЕ ПОЛЯ для IR/Step
  ir?: {
    measurement?: { time: number[]; impulse: number[]; step: number[] };
    target?: { time: number[]; impulse: number[]; step: number[] };
    corrected?: { time: number[]; impulse: number[]; step: number[] };
  };
}
```

### 2. Реализация corrected внутри evaluateBandFull

После вычисления target/peq/measurement добавить блок:

```typescript
// Cross-section (HP/LP filters applied to a flat reference line, used
// for the corrected curve and FIR pipeline).
let crossSectionMag: number[] | null = null;
let crossSectionPhase: number[] | null = null;
if (band.targetEnabled && (band.target.high_pass || band.target.low_pass)) {
  const xs = await invoke<{ magnitude: number[]; phase: number[] }>(
    "compute_cross_section",
    { freq, target: targetCurve, normDb: 0.0 },
  );
  crossSectionMag = xs.magnitude;
  crossSectionPhase = xs.phase;
}

// Corrected = measurement + PEQ + cross-section (magnitude additive in dB).
let correctedMag: number[] | null = null;
let correctedPhase: number[] | null = null;
if (measurement) {
  correctedMag = measurement.magnitude.map((m, i) =>
    m + (peqMag[i] ?? 0) + (crossSectionMag?.[i] ?? 0)
  );
  if (measurement.phase) {
    let basePhase = measurement.phase.map((p, i) =>
      p + (peqPhase[i] ?? 0) + (crossSectionPhase?.[i] ?? 0)
    );
    // Single source of truth: same Gaussian/subsonic reconstruction
    // that target uses, applied here once.
    basePhase = await reconstructTargetPhase(
      freq, basePhase,
      band.target.high_pass, band.target.low_pass,
    );
    correctedPhase = basePhase;
  }
}
```

`compute_cross_section` сигнатуру проверить в Rust (она уже есть,
используется legacy кодом). Если не возвращает phase — добавить
получение phase в Rust команду или собрать через две invoke.

### 3. IR / Step для всех трёх кривых

В `evaluateBandFull` если `req.includeIr`:

```typescript
let ir: BandEvalResult["ir"] = {};

// Measurement IR (raw)
if (measurement) {
  const r = await invoke<{ time: number[]; impulse: number[]; step: number[] }>(
    "compute_impulse",
    { freq: measurement.freq, magnitude: measurement.magnitude,
      phase: measurement.phase ?? new Array(measurement.freq.length).fill(0),
      sampleRate: measurement.sample_rate ?? null },
  );
  ir.measurement = { time: r.time, impulse: r.impulse, step: r.step };
}

// Target IR (model only)
if (targetMag && targetPhase) {
  const r = await invoke<{ time: number[]; impulse: number[]; step: number[] }>(
    "compute_impulse",
    { freq, magnitude: targetMag, phase: targetPhase,
      sampleRate: measurement?.sample_rate ?? 48000 },
  );
  ir.target = { time: r.time, impulse: r.impulse, step: r.step };
}

// Corrected IR (measurement + correction)
if (correctedMag && correctedPhase && measurement) {
  const r = await invoke<{ time: number[]; impulse: number[]; step: number[] }>(
    "compute_impulse",
    { freq, magnitude: correctedMag, phase: correctedPhase,
      sampleRate: measurement.sample_rate ?? 48000 },
  );
  ir.corrected = { time: r.time, impulse: r.impulse, step: r.step };
}
```

### 4. Миграция всех callers

#### renderBandMode (SPL)

Заменить inline вычисление `fullCorrected` / `fullCorrectedPhase`
(строки ~3190–3315) на чтение `evalRes.correctedMag` /
`evalRes.correctedPhase`. Удалить вызов `addGaussianMinPhase`,
удалить inline `compute_cross_section` invoke.

PEQ-only response curve тоже использует `evalRes.peqMag` / `evalRes.peqPhase`.

#### IR/Step view

Найти точку где рисуется impulse/step для отдельной полосы.
Переключить на `evalRes.ir.measurement` / `evalRes.ir.target` /
`evalRes.ir.corrected`.

Удалить inline `compute_impulse` invoke.

#### Group Delay

Если group delay рассчитывается на frontend от phase — продолжать,
но phase брать из соответствующего поля BandEvalResult
(`targetPhase` / `correctedPhase`).

### 5. SUM view (renderSumMode)

Расширить `band-evaluator.ts` функцией:

```typescript
export interface SumEvalResult {
  freq: number[];
  sumMag: number[];
  sumPhase: number[];
  perBand: BandEvalResult[];
  ir?: { time: number[]; impulse: number[]; step: number[] };
}

export async function evaluateSum(
  bands: BandState[],
  options?: { includeIr?: boolean; freq?: number[] },
): Promise<SumEvalResult>;
```

Внутри:
- Для каждой полосы вызвать `evaluateBandFull` (на общей freq grid).
- Сложить combinedTargetMag / combinedTargetPhase с учётом polarity
  (band.inverted) и alignment_delay (phase rotation на per-band уровне).
- Для IR — собрать суммарный impulse через compute_impulse от
  суммарных mag + phase.

`renderSumMode` в FrequencyPlot переключить на `evaluateSum`.
Удалить inline pipeline для SUM (legacy line ~3590).

### 6. Удалить дубликаты

После миграции:
- Удалить `addGaussianMinPhase` из `band-evaluation.ts` (если больше
  нет callers).
- Удалить `evaluateBand` из `band-evaluation.ts` (если больше нет
  callers).
- Удалить inline `compute_cross_section`, `compute_peq_complex`,
  `compute_impulse`, `evaluate_target` invoke из FrequencyPlot.tsx.
- Грэп должен показать что эти invoke остались только в
  `band-evaluator.ts` и `peq-optimize.ts` (последний — независимый
  pipeline, не трогаем).

### 7. Тесты

#### Vitest

- Existing snapshot тесты на target — без изменений (логика не
  трогается).
- НОВЫЕ snapshot тесты на `correctedMag` / `correctedPhase` для 6
  fixture конфигураций — фиксируют правильное поведение
  единственного пути.
- Equivalence тест: `evaluateBandFull` без includeIr возвращает
  те же target/corrected что и старая `evaluateBand` +
  `addGaussianMinPhase` для случая isGaussianMinPhase (legacy путь
  до удаления).

#### Cargo

Без изменений (Rust не трогаем кроме возможного расширения
`compute_cross_section` если там не возвращается phase).

### 8. Acceptance матрица для corrected phase на SPL

Те же 4 комбинации что и target:

| linear | subsonic | corrected phase в полосе пропускания | corrected phase в зоне subsonic (5–80 Hz) |
|---|---|---|---|
| true  | OFF | 0 | 0 |
| true  | ON  | 0 | min-phase Butterworth |
| false | OFF | min-phase Gaussian | ≈ 0 |
| false | ON  | min-phase Gaussian | min-phase Gaussian + Butterworth |

При flat measurement corrected должна **визуально совпадать с
target** во всех 4 случаях.

### 9. Bump

- `src-tauri/tauri.conf.json` → b139.4c.
- `src-tauri/src/lib.rs` startup-лог.
- skill `build-version`.

## Регрессионная проверка

- На SPL вкладке: target и corrected совпадают для flat measurement
  во всех 4 комбинациях.
- На IR/Step вкладке: impulse measurement / target / corrected
  отображаются отдельно, корректно реагируют на чекбоксы видимости.
- На SUM вкладке: сумма всех полос корректно отражает каждую полосу
  с её режимом фазы.
- На Export вкладке (b139.4b) — без регрессий.
- regression-checklist 5 пунктов на `.dmg b139.4c`.

## Что НЕ делать

- Не трогать peq-optimize (независимый pipeline для оптимизатора).
- Не трогать auto-align (низкоуровневый расчёт задержек).
- Не трогать Rust apply_filter / Composite mode.
- Не оставлять параллельные pipeline в FrequencyPlot.tsx — после
  этого этапа все вкладки читают только из BandEvaluator.

## Что прислать обратно

```
cargo: PASS count
vitest: PASS count + новые snapshot тесты на correctedMag/Phase

Manual .dmg b139.4c:
- 4 комбинации на SPL: target и corrected фаза совпадают для flat measurement
- IR/Step: 3 кривые (meas/target/corrected) рендерятся
- SUM: сумма полос корректна
- Удалённые функции: addGaussianMinPhase / inline invoke в FrequencyPlot — список или подтверждение что удалены
```

## Правила

- Один коммит: `refactor: full unification of band evaluation pipeline (b139.4c)`
  + Co-Authored-By.
- 7-vector review.
- Без нарратива.
- При провале acceptance — diagnostic, не слепой фикс.

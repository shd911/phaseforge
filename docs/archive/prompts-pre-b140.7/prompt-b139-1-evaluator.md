# Промт для Code: b139.1 — создать BandEvaluator параллельно со старым

ТЗ целиком: `docs/TZ-unified-evaluation.md`.
Текущий билд: 0.1.0-b139.0 → bump до 0.1.0-b139.1.

## Контекст

В b139.0 regression-checklist обнаружил что phase target subsonic
крутится **только на target SPL вкладке** (через `evaluateBand`).
SUM, Export, IR/Step имеют свои собственные inline pipeline, где
phase reconstruction не делается. Это архитектурная разветвлённость —
точечный фикс на каждой inline точке = патч поверх провала.

Этот этап создаёт **canonical BandEvaluator**, который заменит ВСЕ
inline pipeline. Параллельно со старым кодом — никакие callers не
переключаются. Только новый файл + тесты эквивалентности.

## Pre-flight

### Откатить diagnostic patch

```
cd /Users/olegryzhikov/phaseforge
git checkout src/lib/band-evaluation.ts src/components/FrequencyPlot.tsx
git status   # должно быть clean
```

### Audit existing inline pipeline (для информации)

Перед созданием evaluator — собрать факты о всех существующих
pipeline. Запустить и приложить вывод в первый коммит:

```
grep -n "evaluate_target\|compute_peq_complex\|compute_minimum_phase\|compute_impulse\|generate_model_fir\|response.phase" src/components/FrequencyPlot.tsx
grep -rn "addGaussianMinPhase\|evaluateBand" src/
```

Это reference список callsites которые в Этапах 2-4 будут переключены.

## Что нужно сделать

### 1. Новый файл `src/lib/band-evaluator.ts`

Главная функция `evaluateBandFull` плюс resource helper. Структура:

```typescript
import { invoke } from "@tauri-apps/api/core";
import { createResource, type Resource } from "solid-js";
import type { Measurement, FilterConfig, TargetResponse, PeqBand } from "./types";
import type { BandState } from "../stores/bands";
import {
  isGaussianMinPhase, gaussianFilterMagDb, subsonicMagDb,
  smoothingConfig,
} from "./plot-helpers";
import { hasActiveSubsonicProtect } from "./band-evaluation";

export interface BandEvalRequest {
  band: BandState;
  /** Если undefined — берётся freq измерения, иначе log-grid 20–20k 512 точек. */
  freq?: number[];
  /** Включить FIR coefficients (дорого). */
  includeFir?: boolean;
  /** Включить IR (impulse) и step response. */
  includeIr?: boolean;
}

export interface BandEvalResult {
  freq: number[];
  measurementMag: number[] | null;
  measurementPhase: number[] | null;

  // Pure target (HP × LP × shelves × tilt × subsonic)
  targetMag: number[] | null;
  targetPhase: number[] | null;

  // PEQ correction (включая phase!)
  peqMag: number[];
  peqPhase: number[];

  // Combined target + peq для отображения
  combinedTargetMag: number[] | null;
  combinedTargetPhase: number[] | null;

  refLevel: number;

  // Optional outputs
  fir?: { impulse: number[]; sampleRate: number };
  ir?: { impulse: number[]; step: number[]; time: number[] };
}

export async function evaluateBandFull(req: BandEvalRequest): Promise<BandEvalResult>;

export function createBandEvalResource(
  band: () => BandState,
  options?: {
    freq?: () => number[] | undefined;
    includeFir?: () => boolean;
    includeIr?: () => boolean;
  },
): Resource<BandEvalResult>;
```

### 2. Реализация phase reconstruction (полная и единая)

Внутри `evaluateBandFull` — единая логика phase для target curve,
покрывает все 4 комбинации Gaussian × subsonic:

```typescript
async function reconstructTargetPhase(
  freq: number[],
  basePhase: number[],
  hp: FilterConfig | null | undefined,
  lp: FilterConfig | null | undefined,
): Promise<number[]> {
  let phase = [...basePhase];

  // HP min-phase Gaussian: Hilbert от (Gaussian × Subsonic) magnitude
  if (isGaussianMinPhase(hp)) {
    let hpMag = gaussianFilterMagDb(freq, hp!, false);
    if (hasActiveSubsonicProtect(hp)) {
      const subDb = subsonicMagDb(freq, hp!.freq_hz / 8);
      hpMag = hpMag.map((db, i) => db + subDb[i]);
    }
    const hpPh = await invoke<number[]>("compute_minimum_phase", { freq, magnitude: hpMag });
    phase = phase.map((v, i) => v + hpPh[i]);
  }
  // HP linear-phase Gaussian + subsonic: subsonic-only Hilbert
  else if (hasActiveSubsonicProtect(hp) && hp!.linear_phase === true) {
    const subDb = subsonicMagDb(freq, hp!.freq_hz / 8);
    const subPh = await invoke<number[]>("compute_minimum_phase", { freq, magnitude: subDb });
    phase = phase.map((v, i) => v + subPh[i]);
  }

  // LP min-phase Gaussian
  if (isGaussianMinPhase(lp)) {
    const lpMag = gaussianFilterMagDb(freq, lp!, true);
    const lpPh = await invoke<number[]>("compute_minimum_phase", { freq, magnitude: lpMag });
    phase = phase.map((v, i) => v + lpPh[i]);
  }

  return phase;
}
```

Эта функция — единый источник истины для phase reconstruction. Все
callers (SPL, SUM, Export, IR/Step в Этапах 2-4) будут получать
правильную phase из неё.

### 3. PEQ phase: используем `compute_peq_complex` (не только mag)

Сейчас `fir-export.ts` передаёт только `peqMag` в `generate_model_fir`,
теряя PEQ phase contribution. В `evaluateBandFull` использовать
`compute_peq_complex` (возвращает mag + phase) и складывать в
`combinedTargetPhase`.

Внимание: это **намеренное изменение поведения** — после миграции FIR
будет учитывать PEQ phase. Это исправляет скрытый баг.

### 4. FIR generation (если includeFir)

Внутри `evaluateBandFull` если `includeFir`:

```typescript
const fir = await invoke<{ impulse: number[] }>("generate_model_fir", {
  freq,
  targetMag,                        // pure target, БЕЗ peq
  peqMag,                           // PEQ contribution
  modelPhase: combinedTargetPhase,  // полная phase target+peq, НЕ fill(0)
  config: {
    taps, sample_rate,
    max_boost_db, noise_floor_db, window,
    phase_mode: "MinimumPhase" /* всегда — phase передаётся явно */,
    iterations, freq_weighting, narrowband_limit,
    nb_smoothing_oct, nb_max_excess_db,
  },
});
```

Если `generate_model_fir` в MinimumPhase mode игнорирует `model_phase`
— проверить Rust код. При необходимости либо переключиться на режим
который использует `model_phase`, либо modify generate_model_fir в
Rust чтобы принимал явную phase. **Этого фикса в Этапе 1 НЕ делать**
— зафиксировать как known issue для Этапа 3 (FIR migration).

### 5. IR generation (если includeIr)

Внутри `evaluateBandFull` если `includeIr` и есть measurement:

```typescript
const corrected = await invoke<...>("compute_corrected_impulse", {
  measurementFreq: measurement.freq,
  measurementMag: measurement.magnitude,
  measurementPhase: measurement.phase,
  realizedFreq: freq,
  realizedMag: combinedTargetMag,   // или соответствующий вход
  realizedPhase: combinedTargetPhase,
  // ...
});
```

Если сигнатура `compute_corrected_impulse` отличается — посмотреть в
Rust commands и адаптировать. **Не выдумывать API**.

### 6. SolidJS resource helper

```typescript
export function createBandEvalResource(
  band: () => BandState,
  options?: {
    freq?: () => number[] | undefined;
    includeFir?: () => boolean;
    includeIr?: () => boolean;
  },
): Resource<BandEvalResult> {
  const [resource] = createResource(
    () => ({
      band: band(),
      freq: options?.freq?.(),
      includeFir: options?.includeFir?.() ?? false,
      includeIr: options?.includeIr?.() ?? false,
    }),
    async (req) => evaluateBandFull(req),
  );
  return resource;
}
```

### 7. Тесты эквивалентности

Файл `src/lib/__tests__/band-evaluator.test.ts`:

**A. Snapshot тесты на target phase для 4 Gaussian комбинаций.**

Прогнать `evaluateBandFull` для тех же 6 fixtures из b139.0
(`FIXTURE_CONFIGS`), захватить `targetPhase` (rounded), сравнить с
snapshot.

Это **новые** snapshots — они зафиксируют правильную phase
reconstruction для всех 4 случаев. В Этапах 2-4 будут перехватчики
для других views (SUM/IR/Export), которые должны выдавать те же
phase значения.

**B. Эквивалентность с `evaluateBand` для SPL case.**

Для каждой fixture:

```typescript
const oldResult = await evaluateBand(syntheticBand);
const newResult = await evaluateBandFull({ band: syntheticBand });

// Phase для случаев где evaluateBand правильный:
// - linear=false + subsonic any
// - linear=true + subsonic on (b138.4 фикс)
// - linear=true + subsonic off (тривиально)

const maxDiff = Math.max(...oldResult.targetPhase.map((v, i) =>
  Math.abs(v - newResult.targetPhase[i])));

expect(maxDiff).toBeLessThan(1e-9);  // должны совпадать побитово
```

Magnitude эквивалентность тоже проверить.

**C. Test-double для invoke.**

Для unit-тестов нужен mock invoke. Использовать `vi.mock`:

```typescript
vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string, args: any) => {
    if (cmd === "evaluate_target") return mockEvaluateTarget(args);
    if (cmd === "compute_minimum_phase") return mockMinPhase(args);
    if (cmd === "compute_peq_complex") return mockPeqComplex(args);
    if (cmd === "get_smoothed") return args.magnitude;
    throw new Error(`Unmocked command: ${cmd}`);
  }),
}));
```

Mock implementations должны зеркалить Rust output (для Gaussian:
phase=0; для Hilbert: вернуть детерминированные значения; для PEQ:
zeros). Это даёт reproducible тесты без запуска Tauri.

### 8. Что НЕ делается в этом этапе

- Никаких изменений в `src/lib/band-evaluation.ts` (старая
  `evaluateBand` + `addGaussianMinPhase` остаются).
- Никаких изменений в `src/components/FrequencyPlot.tsx`.
- Никаких изменений в `src/lib/fir-export.ts`.
- Никаких изменений в `src/stores/bands.ts`, `peq-optimize.ts`.
- Никаких изменений в Rust.
- Никаких удалений `evaluate_target_standalone`.

Только **новые файлы**: `src/lib/band-evaluator.ts` и
`src/lib/__tests__/band-evaluator.test.ts`.

### 9. Bump версии

- `src-tauri/tauri.conf.json` — version + productName/title до b139.1.
- `src-tauri/src/lib.rs` — startup-лог.
- После билда — skill `build-version`.

## Acceptance

1. `src/lib/band-evaluator.ts` создан, экспортирует `evaluateBandFull`,
   `createBandEvalResource`, типы.
2. `src/lib/__tests__/band-evaluator.test.ts` создан.
3. **Snapshot тесты** для 6 fixtures × `targetPhase` × `targetMag` —
   все проходят, snapshot закоммичены.
4. **Эквивалентность тест:** для каждой fixture
   `evaluateBandFull({band})` даёт `targetMag` и `targetPhase`,
   совпадающие с `evaluateBand(band) + addGaussianMinPhase(...)` с
   точностью 1e-9.
5. `npm test` зелёный — все existing + новые тесты проходят.
6. `cargo test` зелёный (Rust не трогали).
7. Старые snapshots (b139.0 golden) не изменились.
8. **regression-checklist 10 пунктов** на `.dmg b139.1` проходит
   идентично b139.0 (мы не меняли production callers, поведение должно
   быть тем же — то есть пункт 3 всё ещё провален, это ожидаемо до
   Этапа 4).

## Регрессионная проверка

Поскольку никаких изменений в production callers, `.dmg` должен
вести себя **идентично b139.0**. То что фаза subsonic не крутится
на SUM/IR/Export — это известно и будет починено в Этапах 2-4.

Главное:
- Все existing unit и snapshot тесты зелёные.
- `evaluateBandFull` корректно реконструирует phase для всех 4
  Gaussian комбинаций (видно по snapshot тестам).
- Эквивалентность с `evaluateBand` доказана (видно по equivalence
  тесту).

## Учёт уроков b138 каскада

1. **Audit before write.** Перед созданием helpers (если они нужны) —
   grep на существующие. Использовать `subsonicMagDb`,
   `gaussianFilterMagDb`, `hasActiveSubsonicProtect` из существующих
   модулей. Не дублировать.

2. **Diagnostic-first при провале equivalence теста.** Если новый
   evaluator даёт diff > 1e-9 со старым — НЕ "поправить" слепо.
   Вывести diff массивы, найти точку расхождения, понять причину.

3. **Никаких изменений в callers.** Если в коде нужно что-то
   "временно" поправить в FrequencyPlot — стоп, это уже Этап 2.

4. **Версия в заголовке** = b139.1 — обязательная проверка перед
   regression-checklist.

## Тестировать на `.dmg`

После сборки запустить
`src-tauri/target/release/bundle/dmg/PhaseForge_0.1.139-1_aarch64.dmg`,
проверить версию в заголовке = b139.1, прогнать
`docs/regression-checklist.md` все 10 пунктов. Поведение должно быть
идентично b139.0.

## Правила (CLAUDE.md)

- Один коммит: `feat: BandEvaluator parallel pipeline (b139.1)`
  + Co-Authored-By.
- 7-vector review.
- Без нарратива прогресса.
- Unit тесты обязательны (snapshot + equivalence).
- `cargo tauri build` для финальной сборки.

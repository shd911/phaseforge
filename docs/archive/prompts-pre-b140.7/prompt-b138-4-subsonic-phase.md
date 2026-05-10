# Промт для Code: b138.4 — subsonic min-phase даже при linear-phase Gaussian

**Тип:** функциональный фикс. Bump до 0.1.0-b138.4.

## Контекст и уроки из b138.0–b138.3

Каскад фиксов b138 показал что новое поле `subsonic_protect` живёт
в **множестве дублирующих копировальных функций**, а phase pipeline
**разветвлён по нескольким файлам**. Каждая итерация ловила одну
точку утечки. Этот промт должен закрыть оба класса проблем сразу.

Уроки:
- **b138.2:** локальная `unwrapFilter` в ControlPanel.tsx дублировала
  `unwrapFilterConfig` из bands.ts — и не несла поле. Плюс mirror
  в `FilterBlock.test.tsx`.
- **b138.3:** Hilbert считался от чистого Gaussian magnitude, subsonic
  не входил в input.
- **b138.4 (текущий):** `addGaussianMinPhase` ветвится по `linear_phase`
  — linear ветка вообще не вызывалась.

Защитный subsonic — это реальный физический фильтр в DSP-процессоре,
всегда минимально-фазовый, **независимо от выбранного режима phase
основного Gaussian**.

## Pre-fix audit (обязателен ДО изменений)

### 1. Audit всех точек копирования FilterConfig

Запустить:

```
grep -rn "filter_type:.*\." src/ | grep -v "test\." | grep -v "//"
```

Проверить **каждую** функцию которая копирует FilterConfig поле за
полем — несёт ли она `subsonic_protect`. Известные точки (на момент
b138.3):

- `src/stores/bands.ts` — `unwrapFilterConfig` ✓ (несёт)
- `src/components/ControlPanel.tsx` — `unwrapFilter` ✓ (несёт после
  b138.2)
- `src/components/ControlPanel.tsx` — `withOverride` ✓ (несёт)
- `src/components/FilterBlock.test.tsx` — mirror ✓ (несёт после b138.2)
- `src/lib/project-io.ts` — `cloneFilter` ✓ (несёт после b138)

Если grep показывает новые точки которые **не** несут
`subsonic_protect` — добавить копирование. Если все несут — пропустить.

### 2. Audit всех точек phase pipeline

Запустить:

```
grep -rn "compute_minimum_phase\|targetPhase\|target_phase" src/ src-tauri/src/
```

Перечислить все места где target phase вычисляется или используется.
Известные точки (на момент b138.3):

- `src/lib/band-evaluation.ts` — `evaluateBand` + `addGaussianMinPhase`
- Возможно: `src/components/FrequencyPlot.tsx` — плот target curve
  (использует ли свой pipeline или общий через evaluateBand?)
- Возможно: FIR export (`src/lib/fir-export.ts` или Rust commands)
- Возможно: SUM view расчёт суммы фаз полос
- Возможно: cross-section computation

Для **каждой** точки понять — если там target Gaussian с subsonic,
правильно ли учитывается subsonic phase. Если используется общая
функция (`evaluateBand` или `addGaussianMinPhase`) — фикс ниже
автоматически распространяется. Если дублирует — применить тот же
mod.

Не делать слепо. Прочитать каждую точку, оценить, нужна ли правка.

## Решение

Добавить отдельную ветку в `addGaussianMinPhase` — если Gaussian HP
`linear_phase=true`, но `subsonic_protect=true`, всё равно посчитать
Hilbert от **только subsonic** magnitude и добавить к existing phase.

## Что нужно сделать

### 1. Хелпер subsonic magnitude в `src/lib/plot-helpers.ts`

Добавить экспортируемую функцию:

```typescript
/** Compute Butterworth 8th order HP magnitude in dB for subsonic filter.
 *  Flat 0 dB above cutoff, -48 dB/oct rolloff below. */
export function subsonicMagDb(freq: number[], cutoffHz: number): number[] {
  return freq.map(f => {
    if (f <= 0) return -400;
    const ratio = Math.pow(cutoffHz / f, 16);  // (f_c / f)^(2 * order)
    const lin = Math.sqrt(1 / (1 + ratio));
    return lin > 1e-20 ? 20 * Math.log10(lin) : -400;
  });
}
```

Формула строго совпадает с Rust `apply_filter` subsonic mod (та же
`(f_sub / f)^16`) — это критично для согласованности magnitude между
target evaluation и Hilbert input.

### 2. Расширение `src/lib/band-evaluation.ts:addGaussianMinPhase`

Сейчас функция обрабатывает только `isGaussianMinPhase(hp)` и
`isGaussianMinPhase(lp)`. Добавить **третью ветку** для
linear-phase Gaussian HP с subsonic_protect:

```typescript
import { smoothingConfig, isGaussianMinPhase, gaussianFilterMagDb, subsonicMagDb } from "./plot-helpers";

export async function addGaussianMinPhase(
  freq: number[],
  phase: number[],
  hp: FilterConfig | null | undefined,
  lp: FilterConfig | null | undefined,
): Promise<number[]> {
  let result = phase;

  if (isGaussianMinPhase(hp)) {
    // Existing branch: linear_phase=false → Hilbert от (Gaussian × Subsonic)
    let hpMag = gaussianFilterMagDb(freq, hp!, false);
    if (hp!.subsonic_protect === true && hp!.freq_hz > 40) {
      const fSub = hp!.freq_hz / 8;
      const subDb = subsonicMagDb(freq, fSub);
      hpMag = hpMag.map((db, i) => db + subDb[i]);
    }
    const hpPh = await invoke<number[]>("compute_minimum_phase", { freq, magnitude: hpMag });
    result = result.map((v, i) => v + hpPh[i]);
  } else if (
    hp && hp.filter_type === "Gaussian" && hp.linear_phase === true
    && hp.subsonic_protect === true && hp.freq_hz > 40
  ) {
    // NEW branch: linear-phase Gaussian, but subsonic must still be min-phase.
    // Hilbert от subsonic-only magnitude (без Gaussian части).
    const subDb = subsonicMagDb(freq, hp.freq_hz / 8);
    const subPh = await invoke<number[]>("compute_minimum_phase", { freq, magnitude: subDb });
    result = result.map((v, i) => v + subPh[i]);
  }

  if (isGaussianMinPhase(lp)) {
    const lpMag = gaussianFilterMagDb(freq, lp!, true);
    const lpPh = await invoke<number[]>("compute_minimum_phase", { freq, magnitude: lpMag });
    result = result.map((v, i) => v + lpPh[i]);
  }

  return result;
}
```

Также пересоберить существующую (b138.3) inline subsonic mod через
вызов `subsonicMagDb` (DRY) — формула одна, дублировать в двух местах
опасно.

### 3. Условие вызова `addGaussianMinPhase` в `evaluateBand`

Сейчас (band-evaluation.ts:103):

```typescript
if (targetPhase && freq && (isGaussianMinPhase(band.target.high_pass) || isGaussianMinPhase(band.target.low_pass))) {
  targetPhase = await addGaussianMinPhase(freq, targetPhase, band.target.high_pass, band.target.low_pass);
}
```

Расширить условие — также вызывать когда linear-phase Gaussian с
subsonic_protect:

```typescript
const hasLinearGaussianSubsonic = band.target.high_pass?.filter_type === "Gaussian"
    && band.target.high_pass.linear_phase === true
    && band.target.high_pass.subsonic_protect === true
    && band.target.high_pass.freq_hz > 40;

if (targetPhase && freq && (
    isGaussianMinPhase(band.target.high_pass)
    || isGaussianMinPhase(band.target.low_pass)
    || hasLinearGaussianSubsonic
)) {
  targetPhase = await addGaussianMinPhase(freq, targetPhase, band.target.high_pass, band.target.low_pass);
}
```

Без этого расширения функция не вызовется для linear-phase Gaussian
HP, и subsonic phase не добавится.

### 4. FIR export pipeline

Проверить точку построения target curve в FIR export (вероятно
`src/lib/fir-export.ts` или Rust commands). Если использует
`addGaussianMinPhase` или `evaluateBand` — фикс автоматически работает.
Если строит target отдельно — убедиться что subsonic phase учитывается.

**Не менять FIR pipeline вслепую** — сначала прочитать как он
формирует target curve. Если общая функция — пропустить. Если
дублирует логику — применить аналогичный mod.

### 5. Bump версии

- `src-tauri/tauri.conf.json` — version + productName/title до b138.4.
- `src-tauri/src/lib.rs` — startup-лог.
- После билда — skill `build-version`.

## Acceptance — матрица 2×2 + edge cases

Базовая комбинация: Gaussian HP=632 Гц.

| linear_phase | subsonic_protect | Phase в Gaussian зоне (200–2000 Гц) | Phase в subsonic зоне (5–40 Гц) |
|---|---|---|---|
| true  | OFF | 0  | 0 |
| true  | ON  | 0  | min-phase Butterworth (≈ -720° к нулю) |
| false | OFF | min-phase Gaussian | ~0 (за чертой Gaussian rolloff) |
| false | ON  | min-phase Gaussian | min-phase combined (Gaussian × Butterworth) |

Каждая клетка должна явно проверяться. Никаких пропусков.

Дополнительно:
- HP=30 Гц (≤40), любые комбинации → subsonic не применяется ни к
  magnitude ни к phase. Чекбокс disabled.
- HP не Gaussian (LR4 / Butterworth) → `addGaussianMinPhase` не
  должна делать subsonic вклад вообще.
- Toggle subsonic ON→OFF→ON в режиме linear_phase=true → phase в
  зоне 5–40 Гц соответственно появляется/исчезает.
- Toggle subsonic в любом режиме → b136 stale flag НЕ загорается.
- Cmd+Z (b132) после toggle — phase возвращается к предыдущему
  состоянию.
- Save → Open → состояние восстановлено, phase отрисована корректно.

## FIR export проверка

После основного фикса — собрать FIR со всеми четырьмя комбинациями
из таблицы. В каждом случае проверить что **phase response финального
FIR** соответствует ожидаемому из таблицы. Если FIR pipeline дублирует
target построение (не использует `evaluateBand` или
`addGaussianMinPhase`) — применить тот же mod.

Сейчас не известно точно как FIR строит target. Pre-fix audit (выше)
должен это выяснить.

## Регрессионная проверка

- b131-b138.3 целы.
- Magnitude target — без изменений.
- Чекбокс subsonic переключается (b138.2 не сломан).
- LP filter без subsonic — phase реконструкция работает как раньше.

## Что НЕ трогать

- Rust `apply_filter` — magnitude формула subsonic уже корректна.
- `isGaussianMinPhase` логика — без изменений.
- LP-обработка в addGaussianMinPhase — subsonic применяется только к HP.

## Тестировать на `.dmg` (не на dev)

После сборки **обязательно** запустить
`src-tauri/target/release/bundle/dmg/PhaseForge_0.1.138-4_aarch64.dmg`
и пройти acceptance матрицу.

**Перед тестом убедиться что версия в заголовке окна = b138.4.** В
прошлой итерации b138.3 была сомнительна — Кирилл не был уверен
свежая ли сборка. Версия должна явно соответствовать заявленной.

## Diagnostic-first при провале

Если хотя бы одна клетка матрицы не соответствует ожидаемому **после
этого фикса** — НЕ предлагать второй слепой фикс. Запросить
diagnostic logging вокруг подозрительной точки (значение `c()` в
checkbox handler / содержимое magnitude перед `compute_minimum_phase`
/ что возвращает Hilbert / что попадает в FIR target). Урок из
b138.0→b138.1: слепой фикс без evidence = регрессия.

## Правила (CLAUDE.md)

- Один коммит: `fix: subsonic stays min-phase even with linear-phase
  Gaussian (b138.4)` + Co-Authored-By.
- 7-vector review.
- Без нарратива прогресса.
- `cargo tauri build` для финальной сборки.

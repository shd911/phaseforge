# Промт для Code: b137 — частотно-зависимый Q-cap и маркировка

ТЗ целиком: `docs/TZ-q-envelope.md`.
Текущий билд: 0.1.0-b136 → bump до 0.1.0-b137.

## Принципиальный выбор (зафиксировано)

- Cap и warning thresholds **плавно** зависят от частоты, интерполяция
  по `log2(f)`.
- Опорные точки **захардкожены**.
- Маркировка — **в списке PEQ И на графике**.
- Popup только **по клику** на иконку `⚠`, не hover-tooltip.
- Существующие PEQ-полосы с Q выше нового cap при первой загрузке —
  **не трогаем**, только маркируем.

## Что нужно сделать

### 1. Rust: `src-tauri/src/peq/q_envelope.rs` (новый)

Функции `q_cap_at(freq_hz)` и `q_warn_at(freq_hz)` по формулам:

```
Опорные точки: log2(200), log2(2000)
q_cap:  плато 12 (f≤200), линейно по log2(f), плато 4  (f≥2000)
q_warn: плато 8  (f≤200), линейно по log2(f), плато 3  (f≥2000)
```

Реализация — см. ТЗ, секция Rust часть. Подключить в `peq/mod.rs`:

```rust
pub mod q_envelope;
pub use q_envelope::{q_cap_at, q_warn_at};
```

Unit-тесты обязательны: плато на краях, точка в геометрической середине
(632.456 Гц → cap=8, warn=5.5), warn < cap для всех проверочных точек.

### 2. Замена clamp в LMA и greedy

Места правок (см. ТЗ для точных строк и diff):

- `src-tauri/src/peq/lma.rs:238-243` — внутри LMA-итерации, заменить
  `Q_MAX` на `q_cap_at(freq)` для зоны внутри passband. Зона выше LP
  (`Q_MAX_ABOVE_LP`) остаётся как есть.
- `src-tauri/src/peq/peq/mod.rs:287-289` — post-processing clamp.
- `src-tauri/src/peq/peq/mod.rs:252` — seed Q при добавлении новой
  полосы. Использовать частоту worst_idx.
- `src-tauri/src/peq/greedy.rs:127` — `q.clamp(Q_MIN, Q_MAX)` →
  `q.clamp(Q_MIN, q_envelope::q_cap_at(peak_freq))`.
- `src-tauri/src/peq/greedy.rs:235` — если частота доступна в контексте,
  заменить аналогично; если нет — оставить `Q_MAX` как fallback (LMA
  потом дожмёт).
- `src-tauri/src/peq/lma.rs:285` (penalty term для зоны выше LP) — не
  трогать. Если penalty есть и для зоны внутри passband на основе Q_MAX —
  заменить.

Константа `Q_MAX = 10.0` в `peq/types.rs` — **оставить** как absolute
fallback. Она всё ещё используется в test assertions и точках где
частота недоступна.

### 3. Frontend: `src/lib/peq-quality.ts` (новый)

Зеркало Rust-функции `q_warn_at`:

```typescript
export function qWarnAt(freqHz: number): number;
export function highQIndices(bands: PeqBand[]): Set<number>;
```

Frozen (disabled) полосы не флагуются.

### 4. Маркировка в списке PEQ

Найти компонент списка PEQ-полос (вероятно в `ControlPanel.tsx` или
выделенный `PeqList.tsx`). Рядом с записью, если `i ∈ highQIndices` —
кнопка-иконка `⚠`:

```jsx
<button
  class="peq-warn-icon"
  title=""
  aria-label="Высокая добротность"
  onClick={(e) => { e.stopPropagation(); openHighQPopup(band.peqBands[i], i); }}
>⚠</button>
```

CSS — см. ТЗ.

`title=""` обязателен, чтобы system-tooltip не вмешивался (по
требованию — popup только по клику).

`stopPropagation` нужен чтобы клик на иконку не выделял всю строку
полосы.

### 5. Маркировка на графике

`src/components/FrequencyPlot.tsx` — найти место где рендерятся
маркеры PEQ-полос (по `peqBands` или `peq_bands`). Для каждого маркера
проверять `highQIndices(band.peqBands).has(i)`. Если да —
жёлтая обводка `#d97706`, `stroke-width: 2`. Внутреннюю заливку не
менять.

Учесть что маркеры могут рендериться для разных полос в SUM-режиме —
проверять highQ для каждой полосы отдельно.

### 6. Компонент `src/components/HighQWarningPopup.tsx` (новый)

Promise-based API через сигнал, аналогично `MeasurementAnalysisDialog`:

```typescript
const [_state, _setState] = createSignal<{ band: PeqBand; index: number } | null>(null);

export function openHighQPopup(band: PeqBand, index: number): void {
  _setState({ band, index });
}

export const isHighQPopupOpen = () => _state() !== null;

export default function HighQWarningPopup() { ... }
```

Layout — см. ТЗ. Использует `pn-overlay` + `pn-dialog` как другие
диалоги. Закрытие: Escape, клик на overlay, кнопка «Закрыть».

Смонтировать в `App.tsx` и добавить в `isModalOpen` guard (чтобы
shortcuts не срабатывали когда popup открыт).

### 7. Bump версии

- `src-tauri/tauri.conf.json` — version + productName/title до b137.
- `src-tauri/src/lib.rs` — startup-лог.
- После билда — skill `build-version`.

## Acceptance

1. Optimize на замере с резонансом на 1 кГц → результирующая полоса
   имеет Q ≤ 6.07 (cap@1000 Hz).
2. Загрузка старого `.pfproj` с Q=10 на 1 кГц → полоса не меняется,
   появляется иконка `⚠` в списке и жёлтая обводка маркера на графике.
3. Клик по `⚠` открывает popup с частотой, текущим Q, порогом,
   рекомендацией.
4. Closure popup: Escape, клик вне, кнопка «Закрыть».
5. Frozen (disabled) полоса с Q=10 не маркируется.
6. Drag PEQ-маркера до Q выше warn → иконка появляется после debounce.
7. Drag вниз до Q ниже warn → иконка пропадает.
8. Cmd+Z после Optimize → маркеры и иконки пересчитываются по
   восстановленному состоянию.
9. Unit-тесты для `q_cap_at` / `q_warn_at` проходят (cargo test).
10. Hybrid phase mode (export hybrid_phase=true) — Q-cap применяется
    одинаково.

## Регрессионная проверка

- b131-b136 целы.
- Optimize / Optimize All работают.
- Frozen bands не сломаны (disabled полоса корректно запекается в
  measurement, остальной бюджет распределяется).
- b136 stale flag — изменение target всё ещё триггерит маркировку.
- FIR Export — без изменений в логике, диалог stale (b136) и
  диалог high-Q popup могут существовать независимо.
- `Q_MAX_ABOVE_LP = 2.5` для зон выше LP — не сломан (по-прежнему
  дополнительное ограничение там).

## Что НЕ трогать

- Константа `Q_MAX = 10.0` в `types.rs` — fallback для мест где частота
  недоступна.
- `Q_MAX_ABOVE_LP = 2.5` и связанная логика для зон выше LP.
- `Q_MIN` — нижняя граница не меняется.
- Существующие PEQ-полосы пользователя при первой загрузке проекта в
  b137 — только маркируем, не пересчитываем Q.

## Тестировать на `.dmg`

После сборки запустить
`src-tauri/target/release/bundle/dmg/PhaseForge_0.1.137_aarch64.dmg`
и пройти acceptance. Не полагаться только на `cargo tauri dev`.

## Правила (CLAUDE.md)

- Один коммит: `feat: frequency-dependent Q cap with warning markers (b137)`
  + Co-Authored-By.
- 7-vector review.
- Без нарратива прогресса.
- `cargo tauri build` для финальной сборки.

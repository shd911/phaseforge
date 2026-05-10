# Промт для Code: b132 — откат на 2-3 шага в рамках сессии (light state)

ТЗ целиком: `docs/TZ-project-lifecycle.md` (Часть 2).
Текущий билд: 0.1.0-b131 → bump до 0.1.0-b132.

## Принципиальный выбор

Снимок хранит **только лёгкое состояние**: PEQ-полосы, target, имена полос,
флаги, alignmentDelay, параметры PEQ/FIR/export. Массивы измерений
(`measurement.freq/magnitude/phase`, `originalPhase`) в снимок **не**
включаются. Это держит память на единицах КБ и убирает риски
рассинхронизации PEQ с пересчитанным измерением.

Действия с измерениями (импорт, удаление, замена, merge, floor bounce) —
**не** пушат в стек. Защита для них реализуется отдельными
подтверждениями в момент действия (это вне scope b132).

## Что нужно сделать

### 1. Стор истории `src/stores/history.ts`

```typescript
type LightBand = {
  id: string;
  name: string;
  peqBands: PeqBand[];
  target: TargetCurve;
  targetEnabled: boolean;
  inverted: boolean;
  linkedToNext: boolean;
  alignmentDelay: number;
};

type HistoryEntry = {
  bands: LightBand[];
  activeBandId: string;
  nextBandNum: number;
  peqParams: { tolerance, maxBands, gainRegularization, peqFloor,
               peqRangeMode, peqDirectLow, peqDirectHigh };
  firParams: { iterations, freqWeighting, narrowbandLimit,
               nbSmoothingOct, nbMaxExcess, maxBoost, noiseFloor };
  exportParams: { sampleRate, taps, window, hybridPhase };
  label: string;
  ts: number;
};

const MAX_HISTORY = 5;

export function pushHistory(label: string): void;
export function undo(): void;
export function redo(): void;
export function clearHistory(): void;
export const canUndo: () => boolean;
export const canRedo: () => boolean;
export const lastUndoLabel: () => string | null;
export const lastRedoLabel: () => string | null;
```

`pushHistory`: собирает лёгкое состояние из `appState` + сигналов
PEQ/FIR/export, клонирует через `JSON.parse(JSON.stringify(...))`, кладёт
в `undoStack`. При overflow — `shift()`. После любого push очищает
`redoStack`.

`undo`: текущее лёгкое состояние пушит в `redoStack`, последний снимок из
`undoStack` накатывает через `applyLightSnapshot(entry)`.

`applyLightSnapshot`:
- внутри `batch()`;
- для каждой полосы из снимка ищет соответствующий `appState.bands[i].id`,
  переносит `measurement` и `settings` из current как есть;
- если id из снимка нет в current — создаёт полосу с `measurement: null,
  settings: null, firResult: null`;
- если в current есть полоса, которой нет в снимке — она удаляется;
- порядок полос — как в снимке.

### 2. Триггеры (где вызывать `pushHistory`)

Снимок ДО действия:

- `runAutoOptimize` в `peq-optimize.ts` — перед запуском.
- `bands.ts`: `addBand`, `removeBand`, `renameBand`, изменение
  `target.high_pass/low_pass`, toggle `targetEnabled`, `inverted`,
  `linkedToNext`, bulk-операции PEQ.
- Сдвиг `alignmentDelay` (ручной).

**НЕ пушить:**
- импорт / удаление / замена measurement;
- merge NF+FF;
- floor bounce apply;
- auto-align;
- UI-only действия (смена вкладки, toggle show_*, смена активной полосы).

### 3. Debounce для PEQ-драга и ползунков

Использовать `peqDragging` сигнал. Логика:

- При начале драга — сохранить snapshot ДО изменения в локальный буфер
  (НЕ в стек).
- При `peqDragging=false` + `setTimeout(300ms)` — `pushHistory("PEQ adjust")`,
  при этом фактически в `undoStack` кладётся буферизованный pre-drag
  снимок, не текущее состояние.

Аналогично для ползунков `q/gain/freq` в FilterBlock — debounced push
с pre-edit снимком.

### 4. UI: Undo/Redo в тулбаре

Две кнопки в основном toolbar (рядом с File меню): ↶ Undo, ↷ Redo.

- Disabled если `canUndo()/canRedo() === false`.
- `title` тултип: «Откатить: Optimize» / «Повторить: Add band».
- Шорткаты: Cmd+Z (undo), Shift+Cmd+Z (redo).
- Глобальный listener в `App.tsx`. Не срабатывает когда фокус
  внутри `<input>`, `<textarea>` или элемента с `contenteditable`.

### 5. Очистка истории

`clearHistory()` зовётся:
- в `newProject()` — после reset state;
- в `doLoad()` — после `restoreState`;
- при закрытии текущего проекта (если такой сценарий есть).

### 6. Bump версии

- `src-tauri/tauri.conf.json` — version + productName/title до b132.
- `src-tauri/src/lib.rs` — startup-лог.
- После билда — skill `build-version` для переименования артефактов.

## Acceptance

1. Auto Optimize → Cmd+Z возвращает PEQ-полосы и target в pre-Optimize.
2. Add band → Cmd+Z удаляет добавленную полосу. Cmd+Shift+Z возвращает.
3. Drag PEQ-маркера за один непрерывный drag = одна запись в стеке.
4. Cmd+Z после серии правок ползунка Gain (debounced) откатывает к
   состоянию до начала редактирования.
5. Стек обрезается на 5: после 6 действий старейшее теряется.
6. New / Open чистит стек — Undo сразу disabled.
7. Cmd+Z в текстовом инпуте не срабатывает как Undo.
8. Импорт измерения — Undo НЕ откатывает (поведение по дизайну).
9. Полоса с `measurement` после Undo сохраняет своё измерение
   (берётся из current, не из снимка).

## Регрессионная проверка

- b131 не сломан: Cmd+Q при dirty показывает диалог.
- Save / Save As / Open / Recent.
- Auto-align, floor bounce, merge NF+FF, экспорт FIR/PEQ.
- Память: 5 снимков должны весить < 1 МБ суммарно (без массивов
  измерений). Проверить через `performance.memory` или DevTools.

## Правила (из CLAUDE.md)

- Один коммит: `feat: session undo/redo (b132)` + Co-Authored-By.
- 7-vector review.
- Без нарратива прогресса. Только изменённые пути + одна строка результата.
- `cargo tauri dev` для проверки, `cargo tauri build` для финальной сборки.

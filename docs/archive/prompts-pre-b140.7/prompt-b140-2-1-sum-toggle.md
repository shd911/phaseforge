# Промт для Code: b140.2.1 — UI переключатель Legacy/New SUM

Текущий билд: 0.1.0-b140.2.0.5 → bump до 0.1.0-b140.2.1.

## Контекст

`evaluateSum` готова и покрыта тестами с паритетом legacy
(b140.2.0.5). Этот этап интегрирует её параллельно с inline
aggregation в `renderSumMode`. UI получает переключатель — оба
pipeline активны одновременно, пользователь сравнивает визуально.

Default режим — **Legacy** (минимум сюрпризов). New активируется
явно.

## Что нужно сделать

### 1. Signal для выбора SUM режима

В `src/stores/bands.ts` (или подходящий стор):

```typescript
export type SumMode = "legacy" | "new";

export const [sumMode, setSumMode] = createSignal<SumMode>("legacy");
```

Сохранять выбор в `localStorage` чтобы переключение запоминалось
между сессиями:

```typescript
const STORED = localStorage.getItem("phaseforge.sumMode");
if (STORED === "new" || STORED === "legacy") setSumMode(STORED);

createEffect(() => {
  localStorage.setItem("phaseforge.sumMode", sumMode());
});
```

### 2. UI переключатель рядом с чекбоксами слоёв

Найти точку рендера чекбоксов `MEAS / TARGET / CORR` для SUM view.
Из аудита — рядом с уровнем 250–280 в FrequencyPlot или в legend
area для isSum() ветки.

Добавить toggle-кнопку (две pill-кнопки или переключатель):

```jsx
<Show when={isSum()}>
  <div class="sum-mode-toggle">
    <span class="hint">Сумма:</span>
    <button
      class={`pill ${sumMode() === "legacy" ? "active" : ""}`}
      onClick={() => setSumMode("legacy")}
    >Legacy</button>
    <button
      class={`pill ${sumMode() === "new" ? "active" : ""}`}
      onClick={() => setSumMode("new")}
    >New</button>
  </div>
</Show>
```

CSS в `App.css`:

```css
.sum-mode-toggle {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  margin-left: var(--space-md);
}
.sum-mode-toggle .hint { font-size: 12px; color: #aaa; }
.sum-mode-toggle .pill {
  padding: 2px 8px;
  font-size: 11px;
  border: 1px solid #444;
  background: transparent;
  color: #aaa;
  cursor: pointer;
  border-radius: 3px;
}
.sum-mode-toggle .pill.active {
  background: var(--accent);
  color: white;
  border-color: var(--accent);
}
```

### 3. Branching в `renderSumMode`

В начале функции:

```typescript
async function renderSumMode(showPhase, showMag, showTarget) {
  if (sumMode() === "new") {
    return await renderSumModeNew(showPhase, showMag, showTarget);
  }
  // ... existing legacy code
}
```

`renderSumModeNew` — новая параллельная функция, использует
`evaluateSum` и формирует то же `uPlot.Series` / `uData` что
legacy. Структура легче — суммирование уже сделано в evaluateSum.

```typescript
async function renderSumModeNew(showPhase, showMag, showTarget) {
  const result = await evaluateSum(state.bands, {});
  const freq = result.freq;
  const uSeries: uPlot.Series[] = [{}];
  const uData: number[][] = [freq];
  const legend: LegendEntry[] = [];
  let sIdx = 1;

  // Аналогично legacy: добавляем sumMeas / sumTarget / sumCorrected
  // в uData / uSeries / legend, читая из result.

  if (showTarget && result.sumTargetMag) {
    uSeries.push({ label: "Σ Target dB", stroke: SUM_TARGET_COLOR, ... });
    uData.push(result.sumTargetMag);
    legend.push({ ... });
    sIdx++;
  }

  if (showMag && result.sumCorrectedMag) {
    const corrLabel = result.coherent ? "Σ Corrected dB" : "Σ Corrected dB (incoh)";
    uSeries.push({ label: corrLabel, stroke: SUM_CORRECTED_COLOR, ... });
    uData.push(result.sumCorrectedMag);
    legend.push({ ... });
    sIdx++;
  }

  // ... аналогично для phase когда result.coherent === true

  // chart render через uPlot
}
```

### 4. Аналогично для SUM IR/Step (renderTimeTab SUM ветка)

Если `sumMode() === "new"` — использовать `evaluateSum({ includeIr: true })`
и `result.ir` для отрисовки. Старая ветка остаётся для legacy.

В b140.2.1 — **только** SUM frequency view. SUM IR можно
оставить на legacy до b140.2.2 если scope большой. Но если просто —
включить IR ветку в этом этапе тоже.

### 5. Visual indicator режима

В легенде или title когда активен New:
- Префикс "(New)" в labels.
- Или badge "Σ New" в углу chart.

Это помогает пользователю понять какой режим он смотрит.

### 6. Bump

- `src-tauri/tauri.conf.json` → `0.1.140`.
- `src-tauri/src/lib.rs` startup → b140.2.1.
- `src/lib/version.ts` → b140.2.1.
- skill `build-version`.

## Acceptance

1. Toggle Legacy/New виден на вкладке freq когда выбран SUM.
2. Переключение между Legacy/New триггерит re-render.
3. По умолчанию — Legacy. Выбор сохраняется в localStorage.
4. Legacy режим — bit-exact как в b140.2.0.5 (никаких изменений).
5. New режим использует `evaluateSum`, отображает данные оттуда.
6. На простых проектах (2-3 полосы с phase, без alignment delay,
   без полярности) — оба режима выглядят визуально идентично.
7. Existing 178+ cargo / 160+ vitest тестов PASS.

## Регрессионная проверка

- regression-checklist 5 manual UI пунктов.
- На реальном проекте Кирилла переключение Legacy/New не ломает
  рендер.
- Возврат на Legacy после New — visual identical как при старте.

## Что НЕ делать

- Не удалять legacy код.
- Не делать SUM FIR export — отдельная задача.
- Не трогать non-SUM views.
- Не делать default = New — потенциальный сюрприз.

## Тестировать на `.dmg`

После сборки — реальный проект с 2-3 полосами:
1. Открыть на вкладке SUM.
2. Toggle Legacy → проверить что выглядит как раньше.
3. Toggle New → визуально похоже.
4. Включить полярность одной полосы — оба режима реагируют.
5. Включить alignment delay — оба реагируют.

Если есть видимая разница — это и есть data для сравнения, не
обязательно регрессия.

## Правила

- Один коммит: `feat: Legacy/New SUM toggle for parallel pipeline (b140.2.1)` + Co-Authored-By.
- 7-vector review.
- Без нарратива.

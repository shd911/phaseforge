# Промт для Code: диагностика регрессии b135.2

**Тип:** временный диагностический патч. Без bump версии. Без коммита.
Только `cargo tauri dev`.

## Контекст

В b135.2 после рефакторинга `MeasurementAnalysisDialog.tsx` (убраны
чекбоксы) пропал диалог при импорте «грязного» замера — того самого
файла, на котором в b135.1 диалог с findings показывался корректно.
Чистые замеры — диалог «Замер выглядит чисто» работает. Регрессия
только на грязных.

## Что нужно сделать

### 1. `src/lib/measurement-actions.ts`

В функции `runAnalysis`, перед вызовом `openMeasurementAnalysis`,
добавить лог:

```typescript
console.log("[runAnalysis] result", {
  bandId,
  fileName: m.name,
  n_findings: result.findings.length,
  ids: result.findings.map(f => f.id),
});
console.log("[runAnalysis] calling openMeasurementAnalysis");
```

### 2. `src/components/MeasurementAnalysisDialog.tsx`

В функции `openMeasurementAnalysis` — лог входа:

```typescript
export function openMeasurementAnalysis(req: OpenRequest): void {
  console.log("[Dialog] openMeasurementAnalysis called", {
    bandId: req.bandId,
    fileName: req.fileName,
    n_findings: req.result.findings.length,
  });
  setOpen(req);
}
```

В компоненте `MeasurementAnalysisDialog`, внутри `<Show when={_open()}>`
блока, перед return JSX добавить лог рендера:

```typescript
{(req) => {
  console.log("[Dialog] rendering with findings", req().result.findings.length);
  // ... остальной код
}}
```

### 3. Запуск и сбор данных

`cargo tauri dev` из `/Users/olegryzhikov/phaseforge`.

Импортировать **тот же грязный замер**, который в b135.1 триггерил
диалог с findings.

## Что прислать обратно

Из DevTools Console окна приложения (Cmd+Opt+I) — все строки
с префиксами `[runAnalysis]` и `[Dialog]` после клика Import.

Также из терминала где запущен dev — строку
`analyze_measurement: name=..., len=..., findings=N`.

Прислать оба блока. На основе данных — точечный фикс.

## Что НЕ делать

- Не предлагать гипотезы о причине.
- Не менять логику диалога.
- Не коммитить изменения — это временный патч.

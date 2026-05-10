# Промт для Code: b135.1 — hotfix модалки «проблем не обнаружено»

**Тип:** UI hotfix. Bump до 0.1.0-b135.1, отдельный коммит.

## Контекст диагностики

После b135 пользователь не видит модалку при импорте чистых замеров.
Логи показали: детекторы работают корректно, на FF-замере динамика
честно возвращают `findings = []`. Дыра в integration-логике —
модалка вообще не открывается когда findings пусты, хотя по ТЗ
она должна показывать «Замер выглядит чисто, проблем не обнаружено».

Пороги детекторов в `analysis/mod.rs` менять **не нужно**, они
правильно отрабатывают.

## Что нужно сделать

### 1. `src/lib/measurement-actions.ts`

Функция `runAnalysis`, убрать условие `if (result.findings.length > 0)`.
Модалку открываем всегда:

```typescript
async function runAnalysis(bandId: string, m: Measurement, bandName: string): Promise<void> {
  try {
    const result = await invoke<AnalysisResult>("analyze_measurement", { measurement: m });
    setBandAnalysis(bandId, result);
    openMeasurementAnalysis({ bandId, bandName, fileName: m.name, result });
  } catch (e) {
    console.warn("analyze_measurement failed:", e);
  }
}
```

Также убрать временный диагностический `console.log("[Analysis] ...")`,
если оставался от предыдущего шага.

### 2. `src/components/MeasurementAnalysisDialog.tsx`

Добавить ветку рендеринга для пустого `findings`. Когда
`result.findings.length === 0`, показывать упрощённый layout:

```
┌─ Анализ замера: 6,5 FF.txt ──────────────────────┐
│                                                  │
│   ✓                                              │
│   Замер выглядит чисто                           │
│   Анализ не выявил подозрительных участков.      │
│                                                  │
│                                       [ Закрыть ] │
└──────────────────────────────────────────────────┘
```

При закрытии (Закрыть / Escape / ×) — `setBandAnalysisDismissed(bandId, true)`,
чтобы модалка не открывалась повторно при последующих
`load_project` для этой полосы.

### 3. `src/lib/project-io.ts`

В коде где модалка показывается после `restoreState` — оставить
текущую логику «только если findings.length > 0». Чистые findings
при reopen проекта спамить пользователя не нужно — он их уже видел
при импорте. Здесь ничего менять не надо, проверь что логика
действительно такая.

### 4. Откат диагностических логов

В `src-tauri/src/analysis/mod.rs` — оставить `tracing::info!` про
итоговые findings (полезен в проде), убрать debug-логи причин
отказа каждого детектора. Они нужны были только для разовой
диагностики.

В `src-tauri/src/lib.rs` — откатить временное изменение
`with_env_filter("phaseforge::analysis=debug,...")` обратно к
проектному дефолту.

### 5. TODO — отдельной задачей, НЕ в этом хотфиксе

В логе диагностики обнаружено: `analyze_measurement: ... sr=None`.
Это значит `measurement.sample_rate` не выставляется при импорте
.txt-файла. Это может ломать другие места (FIR export использует
sample_rate). Создать запись в `docs/TODO.md` (или эквиваленте
если уже есть) с описанием:

> sr=None в Measurement после import .txt — проверить io/parser.rs,
> возможно нужен fallback на 48000 если sample_rate в файле
> отсутствует. Влияет на: analyze_measurement, FIR export, IR computation.

Не фиксить в этом хотфиксе.

### 6. Bump версии

- `src-tauri/tauri.conf.json` — version + productName/title до b135.1.
- `src-tauri/src/lib.rs` — startup-лог.
- После билда — skill `build-version`.

## Acceptance

1. Импорт чистого замера (FF замер драйвера) → модалка показывает
   «Замер выглядит чисто», кнопка Закрыть.
2. Импорт замера с явным noise floor снизу → модалка показывает
   findings как раньше (regression проверка b135).
3. После закрытия «всё чисто» модалки — `analysis_dismissed = true`
   в проекте (видно в `.pfproj` json).
4. Re-open проекта где analysis_dismissed=true И findings.length=0 →
   модалка не появляется.
5. Escape закрывает модалку «всё чисто».
6. Merge NF+FF — модалка показывается всегда (даже если findings пусты).

## Регрессионная проверка

- b131-b135 не сломаны.
- Применение рекомендаций (когда findings есть) → session undo работает.
- Импорт нескольких замеров подряд — модалка появляется на каждый.

## Правила (CLAUDE.md)

- Один коммит: `fix: show "all clear" dialog on clean measurements (b135.1)`
  + Co-Authored-By.
- 7-vector review.
- Без нарратива прогресса.
- `cargo tauri build` для финальной сборки.

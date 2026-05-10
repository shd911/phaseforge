# Промт для Code: b135.3 — UX подсказки в диалоге анализа

**Тип:** UX hotfix + откат диагностики. Bump до 0.1.0-b135.3.

## Контекст

После b135.2 диалог с findings показывается, но логика для пользователя
непонятна: применил одну рекомендацию → блок серый. Что делать со
второй? Видно ли что есть «Применить все» внизу? Связаны ли findings
между собой? Нет визуальной подсказки.

## Что НЕ трогать (зона рисков b135.2)

В b135.2 рефакторинг диалога создал регрессии (history flood,
applied-set leak, stale req), которые были закрыты аудитом. **Не
трогать** в этом хотфиксе:

- `_appliedIds` module-level signal и его сброс в `openMeasurementAnalysis`.
- `applyFinding(req, f, rec)` — не менять сигнатуру.
- `applyAll()` и обёртку `beginInteraction("...")` / `commitInteraction()`
  вокруг цикла применения.
- `markApplied()` и логику обновления Set.
- Закрытие диалога через `close()` и `setBandAnalysisDismissed`.

Все правки этого хотфикса — чисто визуальные: добавление subtitle,
пересчёт текста кнопки на signal, перестановка двух кнопок в JSX.

## Что нужно сделать

### 1. Откатить диагностические логи

В `src/lib/measurement-actions.ts` — удалить:

- `console.log("[runAnalysis] result", ...)` перед вызовом
  `openMeasurementAnalysis`;
- `console.log("[runAnalysis] calling openMeasurementAnalysis")`.

В `src/components/MeasurementAnalysisDialog.tsx` — удалить:

- `console.log("[Dialog] openMeasurementAnalysis called", ...)` внутри
  `openMeasurementAnalysis`;
- `console.log("[Dialog] rendering with findings", ...)` внутри
  `<Show when={_open()}>` callback.

### 2. Подсказка над списком findings

В блоке findings (внутри `<Show when={findings().length > 0}>`) перед
строкой «Найдено: …» добавить subtitle:

```jsx
<div style={{ "margin-bottom": "8px", "font-size": "13px", color: "var(--text-primary)" }}>
  Выберите рекомендации, которые применить. Можно по одной или сразу все.
</div>
```

### 3. Счётчик на кнопке «Применить все»

Сейчас:

```jsx
<button ... disabled={allApplied()}>Применить все</button>
```

Заменить текст на динамический:

```jsx
const remainingCount = () => findings().length - appliedFindingIds().size;

// ...

<button ... disabled={allApplied()}>
  {remainingCount() === findings().length
    ? "Применить все"
    : `Применить оставшиеся (${remainingCount()})`}
</button>
```

Когда ещё ничего не применено — текст «Применить все».
Когда применили часть — «Применить оставшиеся (N)», N убывает.
Когда всё применено — disabled (как сейчас).

### 4. Порядок кнопок внизу

Сейчас «Применить все» слева, «Закрыть» справа. Поменять местами:
«Закрыть» слева, «Применить все» справа как primary action. Это
соответствует macOS/иerative HIG — primary в правом нижнем углу.

```jsx
<div class="pn-buttons" style={{ "margin-top": "16px" }}>
  <button class="dlg-btn" onClick={close}>Закрыть</button>
  <Show when={findings().length > 0}>
    <button
      class="dlg-btn dlg-btn-primary"
      onClick={applyAll}
      disabled={allApplied()}
    >...</button>
  </Show>
</div>
```

Если у класса `pn-buttons` есть `justify-content: flex-end` или
аналогичное — порядок отрисовки определяется HTML, должно сработать
автоматически. Если есть особый стайл который реверсит — поправить.

### 5. Bump версии

- `src-tauri/tauri.conf.json` — version + productName/title до b135.3.
- `src-tauri/src/lib.rs` — startup-лог.
- После билда — skill `build-version`.

## Acceptance

1. При открытии диалога с findings виден subtitle с инструкцией.
2. Кнопка «Применить все» имеет текст «Применить все» когда ничего
   не применено.
3. После применения 1 из 3 findings кнопка показывает
   «Применить оставшиеся (2)».
4. После применения всех — кнопка disabled.
5. «Закрыть» слева, «Применить все» справа.
6. Чистый замер: subtitle и счётчика нет (только сообщение
   «Замер выглядит чисто»).
7. Cmd+Z после «Применить все» откатывает весь батч одним шагом
   (b132 + beginInteraction не сломан).
8. В DevTools Console при импорте нет строк `[runAnalysis]` и `[Dialog]`.

## Регрессионная проверка

- b131-b135.2 целы.
- Применение одиночной рекомендации работает как раньше.
- Escape закрывает диалог.
- При загрузке проекта с `analysis_dismissed=true` диалог не появляется.

**Тестировать обязательно на собранном `.dmg`**, не только в `cargo tauri dev`.
В b135.2 диалог корректно работал в dev, но первая проверка велась на
старом dmg — выяснилось ложное расхождение. После сборки запустить
`PhaseForge_0.1.135-3_aarch64.dmg` из bundle/dmg/ и импортировать
тот же грязный замер `8pttRAW.txt`.

## Правила (CLAUDE.md)

- Один коммит: `fix: clearer apply-all UX in analysis dialog (b135.3)`
  + Co-Authored-By.
- 7-vector review.
- Без нарратива прогресса.
- `cargo tauri build` для финальной сборки.

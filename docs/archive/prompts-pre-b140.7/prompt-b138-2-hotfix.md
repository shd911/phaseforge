# Промт для Code: b138.2 — фикс пропавшего subsonic_protect в FilterBlock

**Тип:** одна точечная правка + откат диагностики. Bump до 0.1.0-b138.2.

## Контекст

Диагностика b138.1 показала корень проблемы. Локальная функция
`unwrapFilter` в `src/components/ControlPanel.tsx` (строка 122)
**не копирует** поле `subsonic_protect` при передаче FilterConfig в
`<FilterBlock>`. Поэтому в FilterBlock `c()!.subsonic_protect` всегда
`undefined`, toggle через `!(undefined === true)` даёт `true` каждый
раз.

Фикс — добавить копирование поля в `unwrapFilter`.

## Что нужно сделать

### 1. Точечная правка `src/components/ControlPanel.tsx`

Функция `unwrapFilter` (строка 122–132). Сейчас:

```typescript
function unwrapFilter(f: ...): FilterConfig | null {
  if (!f) return null;
  return {
    filter_type: f.filter_type,
    order: f.order,
    freq_hz: f.freq_hz,
    shape: f.shape,
    linear_phase: f.linear_phase,
    q: f.q,
  };
}
```

Заменить на:

```typescript
function unwrapFilter(f: ...): FilterConfig | null {
  if (!f) return null;
  return {
    filter_type: f.filter_type,
    order: f.order,
    freq_hz: f.freq_hz,
    shape: f.shape,
    linear_phase: f.linear_phase,
    q: f.q,
    subsonic_protect: f.subsonic_protect ?? null,
  };
}
```

### 2. Откат диагностических логов

Удалить из `src/components/ControlPanel.tsx` (около строки 1219–1230)
все `console.log("[CB] ...")` в обработчике onChange чекбокса. Вернуть
к чистому варианту:

```typescript
onChange={() => {
  const newValue = !(c()!.subsonic_protect === true);
  props.onChange(withOverride({ subsonic_protect: newValue }));
}}
```

Удалить из `src/stores/bands.ts` все `console.log("[setHP] ...")` и
`queueMicrotask` лог в `setBandHighPass`.

Если в `FrequencyPlot.tsx` ещё остались `console.log("[EvalTarget] ...")`
от прошлой диагностики — удалить.

Если в `src-tauri/src/target/mod.rs` остались `tracing::info!("apply_filter HP: ...")` или
`tracing::debug!` — удалить.

Если в `src-tauri/src/lib.rs` env_filter поднят до `debug` для target —
вернуть к проектному default `info`.

### 3. Bump версии

- `src-tauri/tauri.conf.json` — version + productName/title до b138.2.
- `src-tauri/src/lib.rs` — startup-лог.
- После билда — skill `build-version`.

## Acceptance

1. Gaussian HP=632 Гц → чекбокс ВКЛ автоматом, на графике target
   ниже 79 Гц виден дополнительный спад.
2. Снимаешь галку → spike исчезает, под чекбоксом warning
   «⚠ Защита отключена…».
3. Ставишь обратно → spike возвращается.
4. Последовательность ON → OFF → ON → OFF → ON → OFF — каждый клик
   меняет состояние. Это главный регресс-тест.
5. В DevTools Console после клика — пусто, никаких `[CB]`/`[setHP]` логов.
6. В терминале с `cargo tauri dev` нет `apply_filter HP:` логов.
7. Save → Open → последнее состояние чекбокса восстановлено.

## Регрессионная проверка

- b131-b138.1 целы.
- Все cargo тесты проходят.
- Optimize / FIR export для Gaussian HP с/без subsonic — без регрессий.
- Изменение subsonic_protect не ставит peqStale=true (b136).
- Round-trip через `.pfproj`: subsonic_protect=false сохраняется
  (раньше unwrapFilter в ControlPanel мог терять поле перед save).

## Что НЕ трогать

- `unwrapFilterConfig` в `bands.ts` — там поле уже корректно копируется.
- Логика чекбокса b138.1 — она правильная, проблема была в источнике
  пропа.
- Rust apply_filter для Gaussian + subsonic — работает корректно
  (видно из того что spike появлялся при первом ВКЛ).

## Тестировать на `.dmg`

После сборки запустить
`src-tauri/target/release/bundle/dmg/PhaseForge_0.1.138-2_aarch64.dmg`
и пройти acceptance pp. 1-7.

## Правила (CLAUDE.md)

- Один коммит: `fix: include subsonic_protect in ControlPanel unwrapFilter (b138.2)`
  + Co-Authored-By.
- 7-vector review.
- Без нарратива прогресса.
- `cargo tauri build` для финальной сборки.

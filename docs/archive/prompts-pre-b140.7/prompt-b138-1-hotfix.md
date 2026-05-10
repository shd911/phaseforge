# Промт для Code: b138.1 — фикс subsonic-чекбокса

**Тип:** UX hotfix. Bump до 0.1.0-b138.1.

## Контекст

Чекбокс «Защитный subsonic» не отключается. Логи показали что
`onChange` срабатывает каждый раз с `newValue=true`. Причина:
controlled checkbox в SolidJS — чтение `e.currentTarget.checked` race-ит
с reactivity (Solid выставляет `checked` после render, и DOM возвращает
не то значение).

Прагматичный фикс: toggle через signal, не через DOM.

## Что нужно сделать

### 1. `src/components/ControlPanel.tsx`, обработчик чекбокса subsonic

**Сейчас** (около строки 423):

```typescript
onChange={(e) => {
  const newValue = e.currentTarget.checked;
  console.log("[Subsonic] checkbox change", {
    newValue,
    bandId: activeBand()?.id,
    hpBefore: { ...activeBand()?.target.high_pass },
  });
  props.onChange(withOverride({ subsonic_protect: newValue }));
  queueMicrotask(() => console.log("[Subsonic] after store update", {
    hpAfter: { ...activeBand()?.target.high_pass },
  }));
}}
```

**Заменить на:**

```typescript
onChange={() => {
  const newValue = !(c()!.subsonic_protect === true);
  props.onChange(withOverride({ subsonic_protect: newValue }));
}}
```

Ключевое: `newValue` вычисляется из **текущего значения signal**, не из
`e.currentTarget.checked`. Это устраняет race с Solid reactivity.

Заодно удаляются `console.log` диагностические — они больше не нужны.

### 2. Откатить остальные диагностические логи

В `src/components/FrequencyPlot.tsx` — удалить `console.log("[EvalTarget] FrequencyPlot:perBand", ...)`.

В `src-tauri/src/target/mod.rs` — если в `apply_filter` остались
`tracing::info!("apply_filter HP: ...")` или `tracing::debug!("apply_filter HP: out mag ...")`,
удалить их.

В `src-tauri/src/lib.rs` — если `with_env_filter` поднят до `debug` для
target/analysis, откатить к проектному дефолту (`info`).

### 3. Bump версии

- `src-tauri/tauri.conf.json` — version + productName/title до b138.1.
- `src-tauri/src/lib.rs` — startup-лог.
- После билда — skill `build-version`.

## Acceptance

1. Gaussian HP=2500 Гц → чекбокс ВКЛ автоматом, target ниже 312 Гц
   (cutoff/8) имеет дополнительный крутой спад.
2. Снимаешь галку → spike исчезает, target ниже плавный Gaussian.
   Под чекбоксом появляется warning «⚠ Защита отключена…».
3. Ставишь галку обратно → spike возвращается, warning исчезает.
4. Несколько последовательных кликов: ON → OFF → ON → OFF — каждый
   клик меняет состояние.
5. Save → Open → последнее состояние чекбокса восстановлено.
6. В DevTools Console при клике чекбокса нет `[Subsonic]` или `[EvalTarget]`
   логов.
7. В терминале с `cargo tauri dev` нет `apply_filter HP:` логов.

## Регрессионная проверка

- b131-b138 целы.
- Все тесты `cargo test` проходят.
- Optimize / FIR export для Gaussian HP с/без subsonic — без регрессий.
- Изменение subsonic_protect не ставит peqStale=true (b136).

## Что НЕ трогать

- Логика `subsonic_protect` в Rust `apply_filter` — она работает корректно
  (видно из того что при первом включении spike появляется).
- `unwrapFilterConfig` в `bands.ts` — корректно сохраняет false.
- `withOverride` — корректно передаёт override через spread.
- Дефолт `subsonic_protect = true` при смене filter_type на Gaussian.
- Логика disabled при freq_hz ≤ 40.

## Тестировать на `.dmg`

После сборки запустить
`src-tauri/target/release/bundle/dmg/PhaseForge_0.1.138-1_aarch64.dmg`
и пройти acceptance pp. 1-7.

## Правила (CLAUDE.md)

- Один коммит: `fix: subsonic checkbox toggle uses signal not DOM (b138.1)`
  + Co-Authored-By.
- 7-vector review.
- Без нарратива прогресса.
- `cargo tauri build` для финальной сборки.

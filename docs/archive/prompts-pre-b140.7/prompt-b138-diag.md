# Промт для Code: диагностика b138 — subsonic toggle не меняет график

**Тип:** временный диагностический патч. Без bump версии. Без коммита.
Только `cargo tauri dev`.

## Контекст

В b138 чекбокс «Защитный subsonic» переключается визуально (галка
ставится/снимается), но кривая target на графике не реагирует.
Subsonic либо не передаётся в Rust, либо не применяется, либо график
не перерисовывается.

## Что нужно сделать

### 1. Frontend: лог в обработчике чекбокса

Найти функцию которая обрабатывает изменение чекбокса subsonic
(вероятно в `src/components/ControlPanel.tsx`). Перед записью в
store / signal — лог:

```typescript
console.log("[Subsonic] checkbox change", {
  newValue: e.currentTarget.checked,
  bandId: activeBand()?.id,
  hpBefore: { ...activeBand()?.target.high_pass },
});
```

После записи — второй лог:

```typescript
console.log("[Subsonic] after store update", {
  hpAfter: { ...activeBand()?.target.high_pass },
});
```

### 2. Frontend: лог в точке вызова evaluate_target

Найти места где `invoke("evaluate_target", { target, freq })` или
аналог вызывается (вероятно в `FrequencyPlot.tsx`, в effect
пересчитывающем target curve, и в `peq-optimize.ts`). В каждой такой
точке перед invoke — лог:

```typescript
console.log("[EvalTarget] calling with", {
  callsite: "<имя_функции_или_файла>",
  hp: target.high_pass,
  lp: target.low_pass,
});
```

### 3. Rust: лог в apply_filter

В `src-tauri/src/target/mod.rs`, функция `apply_filter` (или где
обрабатывается HP). Перед условием subsonic — лог:

```rust
if cfg.is_high_pass {  // или как там определяется HP
    tracing::info!(
        "apply_filter HP: type={:?} freq={} subsonic_protect={:?}",
        cfg.filter_type, cfg.freq_hz, cfg.subsonic_protect
    );
}
```

Внутри ветки subsonic application — ещё один лог:

```rust
if cfg.filter_type == FilterType::Gaussian
    && cfg.subsonic_protect == Some(true)
    && cfg.freq_hz > 40.0
{
    tracing::info!(
        "apply_filter HP: applying subsonic, f_subsonic={}",
        cfg.freq_hz / 8.0
    );
    // ... existing application code
}
```

И финальный лог в выходе apply_filter (уровень debug):

```rust
tracing::debug!(
    "apply_filter HP: out mag at 12.5Hz = {:.3}",
    /* достать magnitude в точке 12.5 Гц через freq grid */
);
```

### 4. Поднять уровень логов

В `src-tauri/src/lib.rs` найти `tracing_subscriber` инициализацию.
Временно поднять до debug для модулей target и mod:

```rust
.with_env_filter("phaseforge_lib::target=debug,phaseforge_lib::analysis=info,info")
```

## Запуск и сбор данных

```
cd /Users/olegryzhikov/phaseforge
cargo tauri dev
```

Создать новую полосу (или взять существующую) с Gaussian HP=100 Гц.
Открыть DevTools → Console (Cmd+Opt+I в окне приложения).

Шаги:
1. Убедиться что чекбокс «Защитный subsonic» виден.
2. Кликнуть чекбокс (если стоял — снять; если не стоял — поставить).
3. Дождаться перерисовки графика (или её отсутствия).

## Что прислать обратно

**Из DevTools Console:**
- Все строки с префиксом `[Subsonic]` (от чекбокса).
- Все строки с префиксом `[EvalTarget]` после клика.

**Из терминала с `cargo tauri dev`:**
- Все строки `apply_filter HP:` после клика.

Прислать оба блока сюда.

## Что НЕ делать

- Не предлагать гипотезы о причине.
- Не менять логику. Только логирование.
- Не коммитить — это временный патч, после диагностики откатим.
- Не делать bump версии.

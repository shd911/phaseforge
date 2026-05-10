# Промт для Code: диагностика b138.1 — toggle всё ещё не работает

**Тип:** временный диагностический патч. Без bump версии. Без коммита.

## Контекст

В b138.1 onChange читает `c()!.subsonic_protect` через signal, а не DOM.
Кирилл всё ещё не может снять галку. Значит проблема не в чтении DOM —
где-то значение `false` перезаписывается обратно в `true` или вообще
не доходит до store.

Нужно увидеть **полный pipeline** от клика до финального store-state.

## Что нужно сделать

### 1. `src/components/ControlPanel.tsx` — checkbox onChange

В обработчике onChange (около строки 423) — заменить на расширенный
лог-вариант:

```typescript
onChange={() => {
  const before = c()!;
  console.log("[CB] before", JSON.stringify(before));
  const newValue = !(before.subsonic_protect === true);
  console.log("[CB] computed newValue", newValue);
  const newConfig = withOverride({ subsonic_protect: newValue });
  console.log("[CB] passing to props.onChange", JSON.stringify(newConfig));
  props.onChange(newConfig);
  console.log("[CB] sync after onChange, c() =", JSON.stringify(c()));
  queueMicrotask(() => {
    console.log("[CB] microtask, c() =", JSON.stringify(c()));
    console.log("[CB] microtask, activeBand HP =", JSON.stringify(activeBand()?.target.high_pass));
  });
}}
```

### 2. `src/stores/bands.ts` — setBandHighPass

В начале функции (после `const idx = bandIndex(bandId);`):

```typescript
console.log("[setHP] CALLED with config =", JSON.stringify(config));
console.log("[setHP] state HP before =", JSON.stringify(state.bands[idx]?.target.high_pass));
```

Внутри `batch(() => { ... })`, после второго `setState("bands", idx, "target", "high_pass", config)` (около строки 573):

```typescript
console.log("[setHP] state HP after setState =", JSON.stringify(state.bands[idx].target.high_pass));
```

В самом конце функции, после `markDirty()`:

```typescript
queueMicrotask(() => {
  console.log("[setHP] microtask state HP =", JSON.stringify(state.bands[idx].target.high_pass));
});
```

### 3. Запуск

```
cd /Users/olegryzhikov/phaseforge
cargo tauri dev
```

Создать новую полосу (или открыть существующую) с Gaussian HP, например
2500 Hz. Открыть DevTools (Cmd+Opt+I) → Console.

**Сценарий:**
1. Подождать пока загрузится — состояние стабильное.
2. Кликнуть на чекбокс subsonic один раз. Подождать секунду.
3. Кликнуть второй раз. Подождать секунду.

## Что прислать обратно

Из DevTools Console — **полный** список всех строк, начинающихся с
`[CB]` и `[setHP]` после двух кликов. Не свёрнутые объекты — нужно
видеть `JSON.stringify` строки целиком.

Прислать всё в порядке появления (Console сохраняет хронологию).

## Что НЕ делать

- Не предлагать гипотезы.
- Не менять логику обработчика, signal, store.
- Не коммитить.
- Не делать bump.

# Промт для Code: b140.3.7 — убрать delay suffix из label corrected

Текущий билд: 0.1.0-b140.3.6 → bump до 0.1.0-b140.3.7.

## Самооценка эффективности

| Критерий | Оценка | Комментарий |
|---|---|---|
| Размер | ✅ малый | Удаление 4 строк |
| Pre-flight audit | ✅ | Корень локализован (label с variable suffix) |

## Контекст

Per-band corrected label на step view содержит value `alignment_delay`
(например `"Band 1 corr+XO [+0.50 ms]"`). При auto-align delay
меняется → label меняется → `bandVisMap` (кэш visibility по label)
теряет соответствие → fallback на `visible:true` для ранее выключенных
кривых.

Delay уже отображается отдельно в Delay column таблицы — duplicate
в label не нужен.

## Что нужно сделать

### В `src/components/FrequencyPlot.tsx`, ~строка 2773-2778

Сейчас:

```typescript
const cf = bandColorFamily(rawBd.bandColor);
const bdDelay = untrack(() =>
  appState.bands.find(b => b.name === rawBd.bandName)?.alignmentDelay ?? 0);
const dlSuffix = Math.abs(bdDelay) > 1e-6
  ? ` [${bdDelay >= 0 ? "+" : ""}${(bdDelay * 1000).toFixed(2)} ms]`
  : "";
addIrStepPair(
  rawBd.bandName + " corr+XO" + dlSuffix,
  cf.corrected, cf.correctedPhase,
  rawBd.timeMs, rawBd.impulse, rawBd.step,
  1.5, "corrected", false,
);
```

Заменить на:

```typescript
const cf = bandColorFamily(rawBd.bandColor);
addIrStepPair(
  rawBd.bandName + " corr+XO",  // stable label, без delay value
  cf.corrected, cf.correctedPhase,
  rawBd.timeMs, rawBd.impulse, rawBd.step,
  1.5, "corrected", false,
);
```

Удалить `bdDelay`, `dlSuffix` если они больше нигде не используются
после этого блока.

### Bump

- `src-tauri/tauri.conf.json` → `0.1.140`.
- `src-tauri/src/lib.rs` startup → b140.3.7.
- `src/lib/version.ts` → b140.3.7.

## Acceptance

1. На IR/Step выключить корректированную кривую → нажать AUTO →
   кривая остаётся выключенной.
2. Delay value показывается в Delay column таблицы (как раньше).

## End-of-prompt автозапуск dev

```
osascript -e 'tell application "PhaseForge" to quit' 2>/dev/null || true
pkill -9 -f -i "phaseforge" 2>/dev/null || true
pkill -9 -f "tauri dev" 2>/dev/null || true
pkill -9 -f "tauri-driver" 2>/dev/null || true
sleep 1
lsof -ti:1420 | xargs kill -9 2>/dev/null || true
lsof -ti:5173 | xargs kill -9 2>/dev/null || true
sleep 1
cd /Users/olegryzhikov/phaseforge && nohup cargo tauri dev > /tmp/phaseforge-dev.log 2>&1 &
```

## Правила

- Один коммит: `fix: stable label for per-band corrected on step view (b140.3.7)` + Co-Authored-By.
- Без нарратива.

# Промт для Code: b140.3.1.3 — ужесточить excess threshold до 0.1 dB

Текущий билд: 0.1.0-b140.3.1.2 → bump до 0.1.0-b140.3.1.3.

## Самооценка эффективности

| Критерий | Оценка | Комментарий |
|---|---|---|
| Размер | ✅ малый | Одна константа + обновлённые тесты |
| Гипотезы без данных | ✅ | b140.3.1.2 работает, просто ужесточение |

## Что нужно сделать

### 1. В `src/lib/band-evaluator.ts`

В функции `limitExcessByWidth`:

```typescript
const EXCESS_THRESHOLD = 0.1;  // было 1.0
```

Логика clip остаётся та же:
- `widthOct ≤ 1/8` → не трогать
- `widthOct ≥ 1/2` → hard clip к `target + 0.1 dB`
- Между → линейная интерполяция

### 2. Обновить тесты

В `src/lib/__tests__/evaluate-sum.test.ts`:
- Тесты с expected values на `+1 dB` → `+0.1 dB`.
- Если есть тесты которые проверяют что небольшие excess (например +0.5 dB)
  не лимитируются — теперь они **должны** лимитироваться.

Обновить через `vitest --update` если используются snapshots.

### 3. Bump

- `src-tauri/tauri.conf.json` → `0.1.140`.
- `src-tauri/src/lib.rs` startup → b140.3.1.3.
- `src/lib/version.ts` → b140.3.1.3.

## Acceptance

1. Wide excess (1 octave) clips к `target + 0.1 dB`.
2. Narrow peak (1/8 oct) preserved.
3. Existing tests обновлены под новый threshold.

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

- Один коммит: `tweak: excess threshold 1.0 → 0.1 dB (b140.3.1.3)` + Co-Authored-By.
- Без нарратива.

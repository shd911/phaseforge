# Промт для Code: b140.3.6 — derive step через cumsum от resampled impulse

Текущий билд: 0.1.0-b140.3.5 → bump до 0.1.0-b140.3.6.

## Самооценка эффективности

| Критерий | Оценка | Комментарий |
|---|---|---|
| Размер | ✅ малый | Точечная правка в `addIrStepPair` |
| Pre-flight audit | ✅ | Корень локализован: independent linear interp ломает cumsum |
| Гипотезы без данных | ✅ | Стандартный cumulative integration |

## Контекст

Аудит показал: corrected step выглядит смещённым относительно
impulse потому что resample делается **независимо** для обоих.
Linear interpolation step как обычных данных ломает свойство
`step[i+1] − step[i] ∝ impulse[i] × dt`.

Решение: после resample impulse derive step **заново** через
cumsum, не resample оригинальный step.

## Что нужно сделать

### В `src/components/FrequencyPlot.tsx`, функция `addIrStepPair` (~строка 2677)

Сейчас:

```typescript
const irData = sameGrid ? applyDb(impulse) : resampleOnto(bandTimeMs, impulse);
const stData = sameGrid ? applyDb(step) : resampleOnto(bandTimeMs, step);
```

Заменить на:

```typescript
const irData = sameGrid ? applyDb(impulse) : resampleOnto(bandTimeMs, impulse);

let stData: number[];
if (sameGrid) {
  stData = applyDb(step);
} else {
  // Derive step from resampled impulse via cumsum to preserve
  // impulse↔step relationship (linear resample of step alone
  // breaks cumsum property).
  const resampledImpulseLinear = resampleOnto(bandTimeMs, impulse);
  let acc = 0;
  const stepDerivedRaw: number[] = [];
  for (const v of resampledImpulseLinear) {
    acc += v;
    stepDerivedRaw.push(acc);
  }
  // Normalize to peak 100% (matches Rust impulse.rs:152-167 behaviour)
  let peak = 0;
  for (const v of stepDerivedRaw) {
    const av = Math.abs(v);
    if (av > peak) peak = av;
  }
  const stepNorm = peak > 0
    ? stepDerivedRaw.map(v => (v / peak) * 100)
    : stepDerivedRaw;
  stData = applyDb(stepNorm);
}
```

`applyDb` оставляем как есть — он применяется уже к нормализованному
step.

### Проверить точные строки

`addIrStepPair` сейчас вызывается несколько раз для разных кривых
(measurement, target, corrected, SUM аналоги). Проверить что фикс
покрывает все.

Ключевая логика в одной функции — обновляется одно место.

### Vitest

Не критично — fronted UI behavior. Manual test на dmg достаточно.
Если есть желание — synthetic test:

```typescript
it("step derived from resampled impulse via cumsum", () => {
  // Synthetic: impulse = [1, 0, 0, 0] на time [0, 1, 2, 3]
  // After resample на time [0, 0.5, 1, 1.5, 2, 2.5, 3]
  // → impulse interpolated, step должен быть proper cumsum
});
```

### Bump

- `src-tauri/tauri.conf.json` → `0.1.140`.
- `src-tauri/src/lib.rs` startup → b140.3.6.
- `src/lib/version.ts` → b140.3.6.

## Acceptance

1. На band IR/Step view corrected step **визуально согласован** с
   corrected impulse (peak step соответствует tail impulse, нет
   "плывущего" смещения).
2. Target IR/Step и Measurement IR/Step не сломаны.
3. SUM IR/Step (если использует тот же `addIrStepPair`) тоже корректен.
4. existing 64+ vitest PASS.

## Что НЕ делать

- Не менять Rust compute_impulse — он корректен.
- Не менять resampleOnto algorithm — он линейный для magnitude data,
  это правильно.
- Не trogat normalize step в Rust — он остаётся.

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

- Один коммит: `fix: derive step via cumsum from resampled impulse (b140.3.6)` + Co-Authored-By.
- Без нарратива.

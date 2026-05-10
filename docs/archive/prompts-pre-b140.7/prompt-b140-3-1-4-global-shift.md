# Промт для Code: b140.3.1.4 — global shift вместо локального clip

Текущий билд: 0.1.0-b140.3.1.3 → bump до 0.1.0-b140.3.1.4.

## Самооценка эффективности

| Критерий | Оценка | Комментарий |
|---|---|---|
| Размер | ✅ малый | Перепись одной функции |
| Pre-flight audit | ✅ | Логика чётко описана пользователем |
| Гипотезы без данных | ✅ | Чистый алгоритм global shift |

## Контекст

Текущий `limitExcessByWidth` (b140.3.1.2) локально clip'ит **только
bins внутри regions excess**. Это нарушает консистентность кривой —
clip создаёт ступеньки, остальная часть corrected остаётся
неизменной.

Правильная логика: zone `passband ± 1 oct` — это **инструмент
контроля**. Если в ней обнаружен wide excess → опустить **ВСЮ**
corrected кривую (по всему freq grid) на величину достаточную
чтобы нивелировать excess до threshold.

## Алгоритм

```
1. excess[j] = corrected[j] - target[j] в зоне passband ± 1 oct
2. Найти все regions где excess > 0.1 dB
3. Для каждого region:
   - widthOct = log2(f_end / f_start)
   - maxExcess = max(excess[j]) в region
4. Игнорировать узкие regions (widthOct ≤ 1/8 oct) — это резонансы
5. Считать "wide regions" с widthOct ≥ 1/2 oct
6. Если есть wide regions:
   - shift = max(maxExcess across wide regions) − 0.1
   - corrected[j] -= shift для ВСЕХ j (весь freq grid, не только zone)
7. Для regions 1/8 < widthOct < 1/2 → soft factor:
   - factor = (widthOct − 1/8) / (1/2 − 1/8)
   - effectiveExcess = maxExcess × factor
   - shift = max(effectiveExcess) − 0.1 (если > 0)
8. Phase не трогаем
```

Если обнаружено несколько wide regions — берётся самый высокий
maxExcess (наихудший случай).

## Что нужно сделать

### 1. Переписать `limitExcessByWidth` в band-evaluator.ts

Переименовать на `applyGlobalShiftIfWideExcess` (или оставить
старое имя — на усмотрение):

```typescript
function applyGlobalShiftIfWideExcess(
  freq: number[],
  corrected: number[],
  target: number[],
  hpFreqHz: number | null,
  lpFreqHz: number | null,
): number[] {
  const EXCESS_THRESHOLD = 0.1;
  const NARROW_OCT = 1/8;
  const WIDE_OCT = 1/2;

  // Control zone: passband ± 1 octave
  const pbLow = hpFreqHz ? hpFreqHz * 1.5 : 20;
  const pbHigh = lpFreqHz ? lpFreqHz * 0.7 : 20000;
  const zoneLow = Math.max(20, pbLow / 2);
  const zoneHigh = Math.min(20000, pbHigh * 2);

  // Find regions of excess > threshold within zone
  let regionStart = -1;
  let regionMaxExcess = 0;
  let maxRequiredShift = 0;

  const finalize = (start: number, end: number, maxEx: number) => {
    const f0 = freq[start];
    const f1 = freq[end];
    const widthOct = Math.log2(f1 / f0);
    let factor: number;
    if (widthOct <= NARROW_OCT) factor = 0;
    else if (widthOct >= WIDE_OCT) factor = 1;
    else factor = (widthOct - NARROW_OCT) / (WIDE_OCT - NARROW_OCT);
    if (factor === 0) return;
    const effectiveExcess = maxEx * factor;
    const required = effectiveExcess - EXCESS_THRESHOLD;
    if (required > maxRequiredShift) maxRequiredShift = required;
  };

  for (let j = 0; j < freq.length; j++) {
    const inZone = freq[j] >= zoneLow && freq[j] <= zoneHigh;
    const ex = inZone && isFinite(corrected[j]) && isFinite(target[j])
      ? corrected[j] - target[j]
      : 0;
    const isExcess = inZone && ex > EXCESS_THRESHOLD;

    if (isExcess) {
      if (regionStart < 0) {
        regionStart = j;
        regionMaxExcess = ex;
      } else if (ex > regionMaxExcess) {
        regionMaxExcess = ex;
      }
    } else if (regionStart >= 0) {
      finalize(regionStart, j - 1, regionMaxExcess);
      regionStart = -1;
      regionMaxExcess = 0;
    }
  }
  if (regionStart >= 0) finalize(regionStart, freq.length - 1, regionMaxExcess);

  // Apply global shift to entire corrected curve
  if (maxRequiredShift > 0) {
    return corrected.map(v => isFinite(v) ? v - maxRequiredShift : v);
  }
  return corrected;
}
```

### 2. Применение в evaluateSum

Существующий вызов `limitExcessByWidth(...)` после per-band normalize:
```typescript
correctedMag = applyGlobalShiftIfWideExcess(
  freq, correctedMag, pbTarget.mag,
  bands[i].target.high_pass?.freq_hz ?? null,
  bands[i].target.low_pass?.freq_hz ?? null,
);
```

UI per-band corrected (`perBandCorrected[i].mag`) тоже shifted —
consistency между per-band и Σ.

### 3. Σ-level — НЕ применяем global shift

Σ Corrected = coherent sum уже shifted per-band кривых.
Дополнительный Σ-level shift не нужен (избегаем legacy avgRef magic).
Если на Σ-уровне всё равно widе excess — это значит несколько полос
складываются и формируют bump, который физически реальный — не маскировать.

### 4. Vitest — обновить тесты

Существующие тесты на `limitExcessByWidth`:
- "Wide excess (1 octave) clips to target + 0.1 dB" — теперь
  поведение: вся кривая shifted вниз. Проверить что в region
  excess стал = 0.1 dB, а в остальных частях кривая ушла на
  ту же величину.
- "Narrow peak (1/8 oct) preserved" — кривая не сдвигается, peak
  остаётся.
- "Medium width (1/4 oct) soft transition" — соответствующий factor.
- "Excess outside zone не клип" — точная формулировка изменится:
  excess вне zone не trigger global shift, но если внутри zone
  есть wide excess — вся кривая (включая bins вне zone) тоже
  shifted.

Добавить:
- "Multiple wide regions → maximum required shift applied"
- "Global shift moves entire corrected curve uniformly"

### 5. Bump

- `src-tauri/tauri.conf.json` → `0.1.140`.
- `src-tauri/src/lib.rs` startup → b140.3.1.4.
- `src/lib/version.ts` → b140.3.1.4.

## Acceptance

1. На 5wayNew bass corrected уровень опускается **по всему freq
   range** так чтобы wide excess в bass passband не превышал target
   на >0.1 dB.
2. Узкие room modes / driver резонансы (1/8 oct) сохраняются — не
   trigger global shift.
3. existing 64+ vitest + обновлённые limit тесты PASS.

## Что НЕ делать

- Не делать Σ-level global shift (только per-band).
- Не сдвигать phase.
- Не сдвигать target — он остаётся неизменным.

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

- Один коммит: `feat: global shift on wide excess instead of local clip (b140.3.1.4)` + Co-Authored-By.
- Без нарратива.

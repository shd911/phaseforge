# Промт для Code: b140.3.1.6 — crossover handles + чистка PEQ dots state

Текущий билд: 0.1.0-b140.3.1.5 → bump до 0.1.0-b140.3.1.6.

## Самооценка эффективности

| Критерий | Оценка | Комментарий |
|---|---|---|
| Размер | ✅ малый | Два точечных фикса в FrequencyPlot |
| Pre-flight audit | ✅ | Точно локализованы оба бага |
| Гипотезы без данных | ✅ | Аудит показал точные строки |

## Контекст

Аудит локализовал:

**Баг 1:** `renderSumModeNew` не передаёт `crossovers` в `renderChart()`.
Mouse handlers (`handleXoMouseDown`, etc.) условны — активируются
только если `currentCrossovers.length > 0`. В new SUM режиме crossover
handles не рисуются и не работают.

**Баг 2:** `activePeqDots` (mutable state, строка 172) устанавливается
в `renderBandMode` (строка 3156) и **не очищается** при переходе в
SUM режим. Draw hook (строки 1341-1383) использует это состояние
без проверки `isSum()` — и пытается рисовать PEQ dots на series с
индексом который актуален только для band mode. В SUM это другие
series, точки попадают «не туда» — это и есть «непонятные точки».

## Что нужно сделать

### Fix 1: crossovers в renderSumModeNew

В `src/components/FrequencyPlot.tsx`, функция `renderSumModeNew`:

```typescript
async function renderSumModeNew(showPhase, showMag, showTarget) {
  const result = await evaluateSum(appState.bands, {});

  // ... existing uSeries / uData / legend building ...

  // Collect crossovers — same logic as getCrossovers() in legacy
  const crossovers: Crossover[] = [];
  for (let i = 0; i < appState.bands.length - 1; i++) {
    const lp = appState.bands[i].target.low_pass;
    if (!lp) continue;
    const xoFreq = lp.freq_hz;
    // dB level on Σ target curve at xoFreq
    let dbLevel: number | null = null;
    if (result.sumTargetMag) {
      const idx = result.freq.findIndex(f => f >= xoFreq);
      if (idx >= 0 && isFinite(result.sumTargetMag[idx])) {
        dbLevel = result.sumTargetMag[idx];
      }
    }
    crossovers.push({
      bandIdx: i,
      freq: xoFreq,
      dbLevel,
      // другие поля если требуются
    });
  }

  // Очистить PEQ dots state (Fix 2)
  activePeqDots = null;

  renderChart({
    freq: result.freq,
    uSeries,
    uData,
    hasMeasurements: false,
    legend,
    crossovers,  // ← добавлено
  });
}
```

Если в legacy `getCrossovers()` имеет более сложную логику (например
учитывает HP следующей полосы или linkedToNext) — переиспользовать
существующую функцию `getCrossovers()` если она доступна.

### Fix 2: Сбрасывать activePeqDots в SUM modes

В `src/components/FrequencyPlot.tsx`:

В **обеих** функциях (`renderSumMode` legacy и `renderSumModeNew`)
в самом начале:

```typescript
async function renderSumMode(...) {
  activePeqDots = null;  // PEQ dots only valid in band mode
  // ... rest
}

async function renderSumModeNew(...) {
  activePeqDots = null;
  // ... rest
}
```

Альтернативно — добавить в draw hook (строка 1341-1383) явную
проверку:

```typescript
if (activePeqDots && activeBand() && !isSum()) {
  // ... draw dots
}
```

Лучше **оба фикса** — clear state + guard в draw hook (defensive).

### 3. Vitest

Не критично — это UI behavior. Manual test на dmg достаточно.

### 4. Bump

- `src-tauri/tauri.conf.json` → `0.1.140`.
- `src-tauri/src/lib.rs` startup → b140.3.1.6.
- `src/lib/version.ts` → b140.3.1.6.

## Acceptance

1. На 5wayNew в New SUM режиме видны **crossover handles** (точки
   на пересечениях полос с Σ target curve).
2. Drag-drop crossover handle меняет HP/LP frequency полос — как в
   Legacy.
3. Double-click открывает CrossoverDialog.
4. **Никаких "непонятных точек"** при переключении band mode → SUM mode.
5. existing 64+ vitest PASS.

## Что НЕ делать

- Не менять crossover algorithm.
- Не реорганизовывать draw hooks beyond этих двух fix'ов.

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

- Один коммит: `fix: crossover handles in New SUM + clear PEQ dots state (b140.3.1.6)` + Co-Authored-By.
- Без нарратива.

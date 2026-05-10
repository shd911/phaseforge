# Промт для Code: b140.8.2 — investigate slope dropdown desync + linking

**Тип:** investigation + tests + targeted fix. Bump до b140.8.2.

## Step 0

```
cd /Users/olegryzhikov/phaseforge
git status
git log --oneline -5
```

## Воспроизведение bugs (user reported)

User видит на одном проекте:
- Band 1 LP=Bessel 200 Hz: показывает "Slope 6 dB/oct" (на одном render)
- Band 1 LP=Bessel 200 Hz: показывает "Slope 18 dB/oct" (на следующем switch)

Без изменения filter, просто переключение bands. **Slope dropdown
desync с актуальным filter.order**.

Также:
- Linking HP/LP в проектах с 5 полосами после удаления band 2 — не
  линкует band 1 и band 3.
- Bessel filter response не consistent (нужно проверить).

## Что нужно сделать

### 1. Bump до b140.8.2

- `tauri.conf.json` → b140.8.2.
- `lib.rs` → b140.8.2.
- `version.ts` → b140.8.2.

### 2. Read-only investigation

#### 2.1 Slope dropdown render

Прочесть `src/components/ControlPanel.tsx` секции которые рендерят
HP/LP filter UI (b140.7.13 Slope dropdown):
- Где `orderToSlope(filterType, order)` вызывается?
- Где значение dropdown selected option берётся?
- Есть ли default fallback когда filter.order не существует или
  равен 0?
- Что происходит при switch band — re-render same component с разным
  filter prop, или unmount/remount?

#### 2.2 Filter linking

Найти код filter link mechanism (link icon между HP/LP):
```
grep -rn "link\|linked\|filterLink" src/ | head -30
```

- Какие поля синхронизируются при link enabled?
- Куда хранится link state — per-band или per-project?
- При delete band — как обновляется link references?

#### 2.3 Bessel response

В `src-tauri/src/target/mod.rs::bessel_lp_complex/bessel_hp_complex`:
- Какая реализация (analog Bessel polynomials? bilinear)?
- Order=N → какой реальный slope?
- Сравнить с UI dropdown options для Bessel (мы дали [6, 12, 18, 24,
  30, 36, 42, 48], т.е. order × 6).

### 3. Записать findings в `docs/slope-bug-investigation.md`

Структура:
```
# Slope dropdown desync investigation (b140.8.2)

## Render path
- file:line where slope is derived for dropdown display
- file:line where slope change handler updates filter.order
- Race condition / stale state issue
- Specific scenario reproducing desync

## Linking mechanism
- file:line link state storage
- file:line propagation logic
- Edge case with delete band

## Bessel implementation
- file:line filter_response
- Slope mapping

## Proposed fixes
- Bullet list specific line edits
- Without DSP changes если возможно
```

### 4. Add tests (TS-side, vitest)

В `src/components/__tests__/` или ближайшей test папке:

```ts
import { describe, it, expect } from 'vitest';
import { orderToSlope, slopeToOrder, availableSlopes } from '...';

describe('Slope mapping', () => {
  it('LR order 4 → 48 dB/oct, 48 → order 4', () => {
    expect(orderToSlope('LinkwitzRiley', 4)).toBe(48);
    expect(slopeToOrder('LinkwitzRiley', 48)).toBe(4);
  });
  
  it('Bessel order 3 → 18 dB/oct, 18 → order 3', () => {
    expect(orderToSlope('Bessel', 3)).toBe(18);
    expect(slopeToOrder('Bessel', 18)).toBe(3);
  });
  
  it('Roundtrip preserves order for all filter types', () => {
    for (const type of ['LinkwitzRiley', 'Butterworth', 'Bessel', 'Custom']) {
      for (const slope of availableSlopes(type)) {
        const order = slopeToOrder(type, slope);
        expect(orderToSlope(type, order)).toBe(slope);
      }
    }
  });

  it('availableSlopes does not include orderless values for type', () => {
    expect(availableSlopes('LinkwitzRiley')).not.toContain(18);  // LR has only 12N
    expect(availableSlopes('Butterworth')).toContain(18);
  });
});
```

Если вижу что desync — добавить тест на render component:

```ts
describe('FilterBlock slope dropdown', () => {
  it('displays slope matching given filter.order on initial render', () => {
    // mount FilterBlock with filter={ type: 'Bessel', order: 3, ... }
    // expect dropdown selected value === '18 dB/oct'
  });
  
  it('updates display when band switches with different filter.order', () => {
    // mount with order=3, then re-render with order=1
    // expect display changes to '6 dB/oct'
  });
});
```

(Точная mount API — solid-testing-library или similar, что у проекта.)

### 5. Apply targeted fixes

После investigation — конкретные line-edits в render path / linking
logic. Без DSP refactor.

После fix:
- Vitest tests все PASS.
- Manual UI re-test desync scenario из user report.

### 6. Run all tests, write logs to project for Cowork to read

```
cd /Users/olegryzhikov/phaseforge
mkdir -p .test-logs
cd src-tauri && cargo test --lib > /Users/olegryzhikov/phaseforge/.test-logs/cargo-lib.log 2>&1
cd /Users/olegryzhikov/phaseforge && npx vitest run > /Users/olegryzhikov/phaseforge/.test-logs/vitest.log 2>&1
echo "=== summary ===" > /Users/olegryzhikov/phaseforge/.test-logs/summary.log
grep "test result" /Users/olegryzhikov/phaseforge/.test-logs/cargo-lib.log >> /Users/olegryzhikov/phaseforge/.test-logs/summary.log
grep -E "Test Files|Tests" /Users/olegryzhikov/phaseforge/.test-logs/vitest.log | head -5 >> /Users/olegryzhikov/phaseforge/.test-logs/summary.log
```

Cowork прочитает `.test-logs/summary.log` и full files напрямую. User
не копирует output вручную.

180+ cargo / 103+ vitest + новые tests PASS.

В `.gitignore` добавить `.test-logs/`.

### 7. Build + UI verify

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

Title bar = b140.8.2.

User scenarios:
- Открыть существующий project flat.
- Switch между bands несколько раз — Slope dropdown остаётся
  consistent.
- Bessel order=3 показывает "18 dB/oct".
- Linking HP/LP propagates slope correctly.

### 8. Commit (после fix verify)

```
git add -A
git commit -m "$(cat <<'EOF'
fix: slope dropdown desync + filter linking edge cases (b140.8.2)

Investigation findings и details:
- ...

Tests added: vitest unit tests на slope mapping invariants и
component render correctness.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

## Что НЕ делать

- Не патчить filter UI без investigation findings.
- Не изменять filter.order semantics в JSON storage.
- Не трогать DSP code.

## Acceptance

- `docs/slope-bug-investigation.md` written.
- Vitest tests PASS на slope mapping и render.
- UI: Switch bands многократно — dropdown остаётся consistent.
- Linking propagates slope.
- Bessel response корректный для slope dropdown values.

## Правила

- Investigation first.
- Tests before fix.
- Без нарратива.

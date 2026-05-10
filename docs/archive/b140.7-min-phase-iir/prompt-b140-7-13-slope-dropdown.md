# Промт для Code: b140.7.13 — Order field → Slope dropdown в UI

**Тип:** UI/UX. Bump до b140.7.13. Коммит после verify в UI.

## Step 0

```
cd /Users/olegryzhikov/phaseforge
git status
```

Должно быть clean (b140.7.12 закоммичен).

## Цель

Заменить "Order: N" numeric input на "Slope: X dB/oct" dropdown в UI
HP/LP секциях. Internal model `order` остаётся, конвертация в обе
стороны при render и при изменении.

Backward compat: существующие проекты сохраняют `order` в JSON, при
load UI показывает соответствующий slope. Save format не меняется.

## Маппинг slope ↔ order

```ts
// Per filter type:
function orderToSlope(filterType: string, order: number): number {
  switch (filterType) {
    case "LinkwitzRiley":
      return order * 12;  // PhaseForge convention: (BU-N)² = 2N order = 12N dB/oct
    case "Butterworth":
    case "Bessel":
    case "Custom":
      return order * 6;   // Standard: order N = 6N dB/oct
    case "Gaussian":
      return 0;  // не используется (Gaussian uses shape parameter)
    default:
      return 0;
  }
}

function slopeToOrder(filterType: string, slope: number): number {
  switch (filterType) {
    case "LinkwitzRiley":
      return Math.round(slope / 12);
    case "Butterworth":
    case "Bessel":
    case "Custom":
      return Math.round(slope / 6);
    case "Gaussian":
      return 1;
    default:
      return 1;
  }
}
```

## Доступные slopes per filter type (dropdown options)

```ts
function availableSlopes(filterType: string): number[] {
  switch (filterType) {
    case "LinkwitzRiley":
      return [12, 24, 36, 48, 60, 72, 84, 96];  // orders 1..8 × 12
    case "Butterworth":
    case "Bessel":
    case "Custom":
      return [6, 12, 18, 24, 30, 36, 42, 48];   // orders 1..8 × 6
    case "Gaussian":
      return [];  // hide slope dropdown for Gaussian
    default:
      return [];
  }
}
```

## Что нужно сделать

### 1. Bump до b140.7.13

- `tauri.conf.json` → b140.7.13.
- `lib.rs` → b140.7.13.
- `version.ts` → b140.7.13.

### 2. Найти UI компонент HP/LP filter input

```
grep -rn "Order\|order" src/components/ControlPanel.tsx | head -30
```

Скорее всего секции с label "Order" и numeric input для filter.order.

### 3. Заменить input на dropdown

В `src/components/ControlPanel.tsx` (или где живёт filter UI):
- Helper functions `orderToSlope`, `slopeToOrder`, `availableSlopes`
  (export или inline).
- Заменить numeric Order input на `<select>` с opциями из
  `availableSlopes(filterType)`.
- Label: "Slope" вместо "Order".
- Display value: `${slope} dB/oct`.
- onChange: новый slope → `slopeToOrder` → store filter.order.
- При render: `filter.order → orderToSlope → selected option`.

Если filter type = Gaussian → скрыть slope dropdown (использовать
shape control как раньше).

При смене filter type — пересчитать slope в order для нового type
(или оставить order как есть, slope dropdown пересчитает display).

### 4. Backward compat

Существующие projects в JSON хранят `order`. При load — UI показывает
соответствующий slope. Save format не меняется (продолжает писать
`order`).

Tests существующие на `cfg.order` остаются валидными.

### 5. Vitest

Если есть test на FilterBlock или ControlPanel rendering — проверить
что dropdown отображает правильный slope для заданного order.

```
cd /Users/olegryzhikov/phaseforge && npm run test 2>&1 | tail -10
```

104+ vitest должно остаться.

### 6. Build + UI verify

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

User verify:
- Title bar = b140.7.13.
- Открыть существующий проект (например flat).
- HP/LP секции должны показать **"Slope: 48 dB/oct"** dropdown
  (вместо "Order: 4" для LR4).
- Изменить slope → response должен перерисоваться корректно.
- Сохранить → закрыть → открыть — корректно загружается.

### 7. Commit

```
git add -A
git commit -m "$(cat <<'EOF'
feat: replace Order field with Slope dropdown in HP/LP UI (b140.7.13)

Filter order parameter ambiguous due to PhaseForge LR convention
(internally (BU-N)² = 2N order). UI now shows actual slope in
dB/oct, eliminating confusion: LR Order=4 → "Slope: 48 dB/oct".

LR slopes: 12, 24, 36, 48, 60, 72, 84, 96 dB/oct (orders 1-8 × 12).
BU/Bessel/Custom: 6, 12, 18, 24, 30, 36, 42, 48 (orders 1-8 × 6).
Gaussian uses shape parameter — slope hidden.

Internal model unchanged (filter.order in JSON), save format intact.
Existing projects load with auto-mapped slope display.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

### 8. End-of-prompt — в шаге 6.

## Что НЕ делать

- Не менять save schema (filter.order остаётся).
- Не трогать DSP code.
- Не менять internal API (Tauri commands).
- Не править existing tests на cfg.order.

## Acceptance

- UI HP/LP показывает "Slope: X dB/oct" dropdown.
- Существующий проект LR4 шоу "48 dB/oct".
- Смена slope обновляет response.
- Save/load round-trip works.
- 104+ vitest PASS.

## Правила

- Без нарратива.
- Один short report после verify.

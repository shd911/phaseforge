# Промт для Code: b140.7.7 — fixed shift=N/2 + clamp plot at Nyquist

**Тип:** bug fix continuation b140.7.6. Bump до b140.7.7, коммит после
verify.

## Step 0

```
cd /Users/olegryzhikov/phaseforge
git status
```

## Контекст

После b140.7.6 (centering peak at N/2):
- Magnitude/REW работает на всех sr ✓.
- Phase на 176k OK, на 48k имеет lift на Nyquist (≠ 0).
- User: "REPhase 0 фазы на 1/2 частоты — у нас при 176 да, при 48 нет".

Корень: `shift = half - peak_idx` варьируется по peak_idx.
- peak=0 → shift=N/2 (чётный) → Nyquist correction=0 ✓
- peak=1 → shift=N/2-1 (нечётный) → Nyquist correction=π ✗
- peak=227 → shift=N/2-227 → разные wrap effects

REPhase использует фиксированный shift = N/2 (impulse delay = N/2 точно),
не подстраивается под peak_idx. Peak в WAV лежит at N/2 + peak_idx
(близко к середине, но не точно). REW happy всё равно.

## Что нужно сделать

### 1. Bump до b140.7.7

- `tauri.conf.json` → b140.7.7.
- `lib.rs` → b140.7.7.
- `version.ts` → b140.7.7.

### 2. Fixed shift=N/2 в iir_path.rs

Изменить:
```rust
// БЫЛО:
let shift = half.saturating_sub(raw_peak_idx);

// СТАЛО:
let shift = half;  // Always exactly N/2 — matches REPhase, gives even
                    // shift → Nyquist phase = 0 на всех sr.
```

Логирование:
```rust
tracing::info!(
    "[IIR PATH] centered: raw_peak={} → new_peak={} (shift=N/2={})",
    raw_peak_idx, raw_peak_idx + shift, shift
);
```

Padding logic тот же (пад leading zeros, truncate tail).

### 3. Delay correction в realized_phase

Update формулу: shift всегда N/2.
```rust
// При вычислении realized_phase из FFT(impulse), вычесть linear delay:
// corrected_phase[k] = raw_phase[k] + 2π·k·shift/N
// shift = N/2:
// correction[k] = 2π·k·(N/2)/N = π·k
// At Nyquist (k=N/2): correction = π·N/2. For N=65536: π·32768 = even·π → 0 mod 2π.
```

Если уже использует `shift` переменную — автоматически работает корректно
после изменения значения shift.

### 4. Plot extrapolation за Nyquist (BAND-AID)

При sr<88k (sr=44.1/48k), log grid 5..40k уходит за Nyquist (22.05k или
24k). Realized_mag/phase за Nyquist — extrapolation, может давать garbage.

В функции resample log→linear or interpolation:
- Найти где realized_mag/phase resampled на 5..40k log grid.
- При sr/2 < 40000: для freq > sr/2 → set realized_mag = noise_floor_db
  (уже так?), realized_phase = 0 (или последнее in-range значение).

Это display polish — основной phase fix в шаге 2-3.

### 5. Cargo + vitest

```
cd src-tauri && cargo test --lib 2>&1 | tail -10
cd /Users/olegryzhikov/phaseforge && npm run test 2>&1 | tail -10
```

185+ cargo / 104+ vitest.

Тесты на peak position могут потребовать update — теперь ожидание `peak_idx
≈ N/2 + raw_peak_idx`, не строго N/2.

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
- Title bar = b140.7.7.
- Band 3 HP=2000 sr=48k → Export tab → курсор на Nyquist (24k):
  Phase должна быть ≈ 0° (или wrap к 0).
- Band 1 LP=200 sr=48k → то же.
- 176k bands → должно остаться корректным.
- REW load WAV → magnitude shape остаётся правильной.

### 7. Commit (только после verify)

```
git add -A
git commit -m "$(cat <<'EOF'
fix: fixed shift=N/2 for REW phase consistency at sr=44.1/48k (b140.7.7)

After b140.7.6 centering, shift = N/2 - peak_idx varied by peak position.
For odd shifts (peak_idx=1 case at sr=48k), Nyquist phase compensation
wrapped to π instead of 0 — visible as phase lift at Nyquist on 48k
plots, while sr=176k (peak naturally at idx=0) had correct shift=N/2.

Match REPhase convention exactly: always pad with N/2 leading zeros
regardless of natural peak position. Peak lands at N/2 + peak_idx
(close to middle, REW peak detection still happy). Linear-phase
delay term = exactly N/2 samples → even-multiple of π at Nyquist
→ phase = 0.

Plot phase at sr<88k beyond Nyquist clamped to 0 (was extrapolated
garbage).

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

### 8. End-of-prompt automation — в шаге 6.

## Что НЕ делать

- Не убирать centering целиком — он нужен для REW magnitude.
- Не trying других подходов phase compensation — fixed shift самый
  простой и матчит REPhase.

## Acceptance

- Title bar = b140.7.7.
- Phase plot на sr=48k Band 3 HP=2000: на Nyquist phase ≈ 0°.
- 176k не регрессировал.
- REW shape всех бендов остаётся правильным.
- 185+ cargo + 104+ vitest PASS.
- Commit с co-author.

## Правила

- Без нарратива.
- Один report.

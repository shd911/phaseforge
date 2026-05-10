# Промт для Code: b140.7.9 — phase correction по peak_pos, не shift

**Тип:** one-line fix + bump. Коммит после verify.

## Step 0

```
cd /Users/olegryzhikov/phaseforge
git status
```

## Контекст

Diag b140.7.8 показал:
- 48k: raw_peak=1, shift=32768, **new_peak=32769**
- 176k: raw_peak=0, shift=32768, **new_peak=32768**

Phase correction в обоих случаях по shift=32768. На 176k peak совпадает
с shift → корректно. На 48k peak = shift+1 → 1 sample residual delay
→ **π** на Nyquist (180°).

Math: residual phase at bin k = 2π·k·(peak_pos - correction)/N. При
correction = peak_pos → 0. При correction = shift и peak_pos = shift +
raw_peak_idx → 2π·k·raw_peak_idx/N. На k=Nyquist это π·raw_peak_idx.
Для odd raw_peak_idx → π.

## Что нужно сделать

### 1. Bump до b140.7.9

- `tauri.conf.json` → b140.7.9.
- `lib.rs` → b140.7.9.
- `version.ts` → b140.7.9.

### 2. One-line fix в iir_path.rs

В `src-tauri/src/fir/iir_path.rs` (около строки 409):

```rust
// БЫЛО:
let delay_samples = shift as f64;

// СТАЛО:
let delay_samples = (shift + raw_peak_idx) as f64;
```

`raw_peak_idx` — это начальная позиция peak в исходном cascade impulse
(до padding). Должна быть доступна как переменная в scope (мы её уже
используем для логирования `[IIR PATH] padded leading: raw_peak={}`).

### 3. Update [CENTER DIAG] log

Поменять чтобы выводил actual peak position и correction match:

```rust
let peak_pos = shift + raw_peak_idx;
tracing::info!(
    "[CENTER DIAG] pad_shift={} raw_peak={} peak_pos={} correction_samples={} N={}",
    shift, raw_peak_idx, peak_pos, peak_pos, n
);
```

(correction_samples = peak_pos после фикса, всегда матчит)

### 4. Cargo + vitest

```
cd src-tauri && cargo test --lib 2>&1 | tail -5
cd /Users/olegryzhikov/phaseforge && npm run test 2>&1 | tail -5
```

185+ cargo / 104+ vitest должно остаться.

### 5. Build + verify

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
- Title bar = b140.7.9.
- Band 3 HP=2000 sr=48k → Export → курсор на 23.5 кГц / 24 кГц:
  Phase должна быть **близко к 0°** (не +180°).
- 176k не должен регрессировать.

### 6. Commit (после REW PASS)

```
git add -A
git commit -m "$(cat <<'EOF'
fix: phase correction uses actual peak position not shift (b140.7.9)

After centering, peak in padded impulse lies at sample
(shift + raw_peak_idx), not at sample shift. Phase delay correction
must match actual peak position to fully remove linear-phase term.

For sr=176k HP=2000 raw_peak_idx=0 → correction=shift=N/2 (matches).
For sr=48k HP=2000 raw_peak_idx=1 → previous correction=N/2 left
1-sample residual, manifesting as 180° phase offset at Nyquist.

Fix: delay_samples = shift + raw_peak_idx.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

## Что НЕ делать

- Не добавлять Hilbert / cepstral.
- Не менять padding logic.
- Не убирать diag tracing (оставим до полной стабилизации).

## Acceptance

- Phase Band 3 HP=2000 sr=48k Nyquist ≈ 0°.
- 176k без регрессии.
- 185+ cargo + 104+ vitest PASS.

## Правила

- One-line изменение.
- Без нарратива.

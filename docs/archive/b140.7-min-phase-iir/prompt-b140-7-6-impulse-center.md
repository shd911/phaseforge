# Промт для Code: b140.7.6 — center impulse at N/2 для REW compatibility

**Тип:** feature + bug fix. Bump до b140.7.6, коммит после verify.

## Step 0

```
cd /Users/olegryzhikov/phaseforge
git status
git diff --stat
```

Должны быть modified files от b140.7.5 (revert f32, diag tracing).

## Контекст

Подтверждено diag b140.7.5:
- IIR cascade impulse корректный.
- WAV writer корректный (readback bytes-perfect).
- Python FFT WAV даёт идеальный LR8 HP/LP shape.
- REW показывает broken только для случаев где peak ≠ at idx=0
  (peak at idx=1 или idx=227 etc.).

Reference REPhase analysis (user скриншоты):
- REPhase **всегда** использует centering="middle" + "use closest perfect
  impulse", даже для Minimum-Phase Filters mode.
- impulse delay = 32500 samples (≈ N/2 для taps=65000) на ВСЕХ sr.
- Peak at middle → REW always handles correctly.

Решение для PhaseForge: после IIR cascade pad с leading zeros так
чтобы peak был at idx=N/2. Math: H'(f) = H(f) · exp(-j·π·k) — same
magnitude, additional linear-phase delay (= N/2 samples ≈ 683 ms @
48k). Для DRC/FIR это нормально, HQP конвольвер компенсирует.

## Что нужно сделать

### 1. Bump до b140.7.6

- `tauri.conf.json` → b140.7.6.
- `lib.rs` → b140.7.6.
- `version.ts` → b140.7.6.

### 2. Реализация centering в iir_path.rs

В `src-tauri/src/fir/iir_path.rs::generate_min_phase_fir_iir`,
после генерации raw impulse через `cascade_impulse(...)` и **до**
вычисления realized_mag/phase + WAV save:

```rust
// b140.7.6: center peak at N/2 for REW + REPhase-style compatibility.
// REW expects peak at middle (matches REPhase centering="middle").
// Adds N/2 samples of latency, accepted in DRC/FIR workflows.
let n = impulse.len();
let half = n / 2;
let raw_peak_idx = impulse.iter().enumerate()
    .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
    .map(|(i, _)| i).unwrap_or(0);
let shift = half.saturating_sub(raw_peak_idx);

let centered: Vec<f64> = if shift > 0 {
    let mut out = vec![0.0; n];
    let copy_len = n - shift;
    for i in 0..copy_len {
        out[i + shift] = impulse[i];
    }
    out
} else {
    impulse  // peak уже at >= N/2 — оставляем
};

tracing::info!(
    "[IIR PATH] centered: raw_peak={} → new_peak≈{} (shift={})",
    raw_peak_idx, half, shift
);

// Дальше уже работает с centered вместо impulse
let impulse = centered;
```

Поместить **до** computation realized_mag/phase в IirPathOutput, чтобы
UI plot тоже отражал centered impulse.

### 3. Обновить causality metric

Causality после centering ≈ 50% (peak at N/2 — половина энергии до peak,
половина после). Это **математическое** свойство centered impulse, не bug.
Возможно UI status display бы показывал "Centered (50%)" вместо просто
"Causal: 50%". Но это косметика — не правим в этом промте.

### 4. Cleanup diag tracing

Удалить:
- `[IIR PATH DIAG] output: ...` в `iir_path.rs`
- `[EXPORT WAV DIAG]` в `lib.rs::export_fir_wav`
- `[READBACK DIAG]` в `lib.rs::export_fir_wav`
- `[EXPORT WAV DIAG TS]` в TS files
- `[DIAG ACTIVE]` startup marker в `lib.rs`

Оставить только обычные info логи (`generate_model_fir_iir`, `[IIR PATH]`).

### 5. Cargo + vitest

```
cd src-tauri && cargo test --lib 2>&1 | tail -10
cd /Users/olegryzhikov/phaseforge && npm run test 2>&1 | tail -10
```

Должно быть **185+ cargo / 104+ vitest PASS**.

Возможно сломаются tests которые проверяли `peak_idx <= 5` — пересохранить
golden snapshots с новым ожиданием peak_idx ≈ N/2.

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
- Title bar = b140.7.6.
- Открыть Band 3 HP=2000 sr=48k → Export WAV.
- В терминале:
  ```bash
  python3 docs/wav-fft-compare.py "<path>"
  ```
  Должен показать тот же HP shape (точная magnitude та же).
- Загрузить WAV в REW → должен показать корректный HP shape.
- Pre-ring и Causal в UI status: Pre-ring увеличится до ≈ N/2/sr,
  Causal будет ≈ 50% — это **ожидаемо** для centered impulse.
- Проверить sr=176.4k также — должно остаться корректным.

### 7. Commit (только после REW PASS)

```
git add -A
git commit -m "$(cat <<'EOF'
fix: center IIR impulse at N/2 for REW compatibility (b140.7.6)

REW expects FIR impulse peak at middle of buffer (matches REPhase
centering="middle" convention used in both Linear-Phase and
Minimum-Phase modes). PhaseForge IIR cascade naturally produced
peak at idx=0 (HP) or near-zero idx (LP min-phase), which REW
misinterprets at sr=44.1/48k specifically — auto-detected peak
position breaks REW's frequency response visualization.

Fix: after cascade_impulse, pad with leading zeros so peak lands
at idx=N/2. Math: |H'(f)| = |H(f)|, additional N/2 sample linear-
phase delay (683 ms @ 48k, 186 ms @ 176k). Accepted convention for
DRC/FIR workflows; HQP convolver and similar compensate latency
automatically.

Confirmed by Python FFT compare (docs/wav-fft-compare.py): WAV
content was always mathematically correct LR8 HP/LP shape; only
REW visualization was affected.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

### 8. End-of-prompt automation — уже сделана в шаге 6.

## Что НЕ делать

- Не убирать IIR cascade или routing.
- Не менять WAV format (Float64 ОК).
- Не трогать FFT path.
- Не делать centering toggleable в UI — стандарт всегда middle.

## Acceptance

- Title bar = b140.7.6.
- REW для 48k WAV (Band 1, 2, 3) показывает корректный shape.
- Python FFT для тех же WAVs идентичен предыдущему.
- 176k WAV всё ещё корректный.
- 185+ cargo + 104+ vitest PASS.
- Diag tracing убран.
- Commit с co-author.

## Правила

- Без нарратива.
- Один short report после verify.

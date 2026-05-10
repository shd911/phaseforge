# Промт для Code: b140.7.10 — separate plot computation from WAV centering

**Тип:** architecture refactor (small). Bump до b140.7.10. Коммит после
verify.

## Step 0

```
cd /Users/olegryzhikov/phaseforge
git status
```

Должны быть modified файлы от b140.7.6/7/8/9 — всё ещё некомитнуто.

## Math обоснование (из b140.7.9 audit)

`FFT(raw_impulse)[k] ≡ analytical_filter_response(f=k·sr/N)`. То есть
raw cascade impulse FFT **уже** даёт точно analytical filter response
включая natural filter group delay.

После padding leading zeros — добавляется artificial linear-phase delay,
который требует correction. Correction = shift избыточно или
недостаточно при разных raw_peak_idx, и portable hacks (delay = shift +
peak_idx) ломают модели где natural delay уже большой (LP).

Архитектурное решение: разделить.

- `realized_mag/phase` для **UI plot** computes from FFT(raw_impulse)
  **до** centering — точно матчит analytical model.
- **Centered impulse** используется ТОЛЬКО для output WAV (REW
  compatibility).

Никаких phase corrections.

## Что нужно сделать

### 1. Bump до b140.7.10

- `tauri.conf.json` → b140.7.10.
- `lib.rs` → b140.7.10.
- `version.ts` → b140.7.10.

### 2. Refactor в iir_path.rs

В `src-tauri/src/fir/iir_path.rs::generate_min_phase_fir_iir`:

```rust
// Шаг A: Compute raw cascade impulse
let raw_impulse = cascade_impulse(&biquads, n_taps);

// Шаг B: realized_mag/phase from FFT(raw_impulse) — analytical truth
let realized_mag_lin: Vec<f64>;
let realized_phase_lin: Vec<f64>;
{
    // FFT raw_impulse → magnitude + phase on linear grid
    let n_bins = n_taps / 2 + 1;
    let mut fft_input = raw_impulse.clone();
    let spectrum = real_fft(&mut fft_input);
    realized_mag_lin = spectrum.iter().map(|c| c.norm()).collect();
    realized_phase_lin = spectrum.iter().map(|c| c.arg()).collect();
}

// Шаг C: Resample mag/phase на log grid 5..40k для plot — как было
let realized_mag_log = resample_to_log(...);
let realized_phase_log = resample_to_log(...);

// Шаг D: Centering — для WAV save только
let raw_peak_idx = raw_impulse.iter().enumerate()
    .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
    .map(|(i, _)| i).unwrap_or(0);
let half = n_taps / 2;
let shift = half;
let mut wav_impulse = vec![0.0; n_taps];
let copy_len = n_taps - shift;
for i in 0..copy_len {
    wav_impulse[i + shift] = raw_impulse[i];
}
tracing::info!("[IIR PATH] centered for WAV: raw_peak={} → wav_peak={} (shift=N/2={})",
    raw_peak_idx, raw_peak_idx + shift, shift);

// Шаг E: tail-taper применять к wav_impulse (опционально, как раньше)

// Шаг F: Возврат
IirPathOutput {
    impulse: wav_impulse,        // centered — для WAV save
    realized_mag: realized_mag_log,   // от raw_impulse — для plot
    realized_phase: realized_phase_log,
    ...
}
```

**Удалить полностью**:
- `let delay_samples = ...;` блок.
- `phase correction` (вычитание линейной фазы из realized_phase).
- `[CENTER DIAG]` лог о nyq_correction.

Оставить:
- `[IIR PATH] centered for WAV: ...` лог (полезно для diag).
- Tail-taper логика на `wav_impulse` (если применима).

### 3. Causality computation

`causality` метрика должна reflektить the centered impulse (поскольку
оно идёт в WAV). После centering peak at N/2 + raw_peak_idx, energy
distributed.

Можно либо:
- Compute causality на `wav_impulse` (отражает то что сохраняется).
- Compute causality на `raw_impulse` (отражает natural filter property).

Решить: **на wav_impulse** (consistency с тем что показано в UI status).

### 4. Cargo + vitest

```
cd src-tauri && cargo test --lib 2>&1 | tail -10
cd /Users/olegryzhikov/phaseforge && npm run test 2>&1 | tail -10
```

185+ cargo / 104+ vitest. Тест на iir_lr4_lp_200_realized_phase_matches_target
теперь должен PASS (raw FFT даёт точно analytical phase).

Тесты на peak position (`min_phase_impulse_peaks_at_zero_*`) теперь
проверяют peak в `wav_impulse` — должны pass с peak ≈ N/2.

### 5. Build + UI verify

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
- Title bar = b140.7.10.
- Band 3 HP=2000 sr=48k → Export → курсор по passband:
  Phase Model° == FIR° (≤ 1° error).
- Курсор на 23 кГц (около Nyquist для 48k):
  Phase близко к 0° (нет 180° artifact).
- 176k не регрессировал.
- Band 1 LP=200 → Phase Model° == FIR° в passband.
- REW: WAV magnitude корректный shape (как было после b140.7.6).

### 6. Commit (после verify PASS)

```
git add -A
git commit -m "$(cat <<'EOF'
fix: separate UI plot from WAV centering in IIR path (b140.7.10)

UI plot realized_mag/phase now computed from FFT(raw_cascade_impulse)
directly — matches analytical filter response exactly, including
natural group delay. No phase correction hacks.

WAV impulse centered (peak at ~N/2) for REW compatibility.

Eliminates phase parity artifacts at sr/2 boundary and over-subtraction
of natural filter delay (b140.7.9 broke LP=200 phase test).

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

### 7. End-of-prompt — в шаге 5.

## Что НЕ делать

- Не менять cascade_impulse.
- Не трогать FFT path code (для не-IIR cases).
- Не убирать centering из WAV save — REW нужен.

## Acceptance

- 185+ cargo PASS (включая ранее упавший iir_lr4_lp_200_realized_phase).
- 104+ vitest.
- UI plot phase Band 3 HP=2000 sr=48k Nyquist ≈ 0°.
- UI plot phase Band 1 LP=200 sr=48k matches model в passband.
- REW magnitude shape корректен на всех sr.
- Commit с co-author.

## Правила

- Без нарратива в чате.
- Один short report после verify.

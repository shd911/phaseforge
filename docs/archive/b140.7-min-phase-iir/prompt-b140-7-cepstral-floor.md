# Промт для Code: b140.7 — cepstral floor для min-phase reconstruction

**Тип:** bug fix. Bump до b140.7, коммит после verify.

## Step 0 — проверка состояния

```
cd /Users/olegryzhikov/phaseforge
git diff --stat
```

Должны быть modified файлы только от текущего diagnostic патча
(`[DIAG-MP]` логи + `[DIAG ACTIVE]` маркер). Если что-то ещё —
STOP, обсудить.

## Root cause (подтверждён диагностикой и трасировкой кода)

`src-tauri/src/dsp/phase.rs:10-51` `minimum_phase_from_magnitude` —
cepstral алгоритм без floor для `ln|H|` перед FFT. На sparse linear
FFT grid (большая часть бинов прижата к noise_floor=-150 dB):

- ln(10^(-150/20)) = -17.27 — большая «полка» на неактивных бинах.
- В quefrency-домене это high-quefrency feature → constant group
  delay в реконструированной min-phase → impulse peak shifted from idx=0.

Data correlation:
| active_bins | peak idx | Pre-ring (ms @48k) |
|---|---|---|
| 7.2% (LP=200) | 204 | 4.25 |
| 72.0% (BP) | 32 | 0.67 |
| 94.2% (HP=2000) | 2 | 0.04 |

Iterative_refine divergence на Band 3 (max_err 22→31 dB) — derivative
effect: peak idx≠0 + window not centered on peak → realized mag
ripple → refinement интерпретирует фазовый артефакт как mag error
и амплифицирует. Уйдёт автоматически после фикса.

## Что нужно сделать

### 1. Bump

- `src-tauri/tauri.conf.json` — title → `"PhaseForge — b140.7"`.
- `src-tauri/src/lib.rs` — startup log → `b140.7`.

### 2. Fix в minimum_phase_from_magnitude

`src-tauri/src/dsp/phase.rs`, функция `minimum_phase_from_magnitude`.
Перед FFT(ln|H|) — clamp ln|H| к минимум `-60 dB` (= ln(10^(-60/20))
= -6.9078).

```rust
const CEPSTRAL_FLOOR_DB: f64 = -60.0;
let cepstral_floor_ln = (10f64).powf(CEPSTRAL_FLOOR_DB / 20.0).ln(); // ≈ -6.9078

// Перед FFT — clamp ln|H|
for v in log_mag.iter_mut() {
    if *v < cepstral_floor_ln {
        *v = cepstral_floor_ln;
    }
}
```

Точное место и имя переменной (`log_mag`, `ln_mag`, etc.) — на
усмотрение Code, главное **clip ln|H| снизу к -6.9078** перед FFT
в cepstrum domain.

Константа экспортируется как `pub const CEPSTRAL_FLOOR_DB`.

### 3. Cleanup диагностики

Убрать все `[DIAG-MP]` логи из `src-tauri/src/fir/mod.rs` и
`src-tauri/src/fir/helpers.rs`. Убрать `[DIAG ACTIVE] MP-peak` строку
из `src-tauri/src/lib.rs`.

### 4. Cargo тесты

Новый файл (или в `src-tauri/src/dsp/phase.rs` под `#[cfg(test)]`):

```rust
#[test]
fn min_phase_peak_at_zero_for_sparse_spectrum() {
    let n_fft = 65536;
    let sr = 48000.0;
    
    // Three scenarios mirroring production bands
    for (label, hp_hz, lp_hz, expected_max_idx) in [
        ("LP=200 (7% active)", None, Some(200.0), 50),
        ("BP 200-2000 (72% active)", Some(200.0), Some(2000.0), 50),
        ("HP=2000 (94% active)", Some(2000.0), None, 50),
    ] {
        // Build sparse magnitude on linear FFT grid:
        // active in passband, noise_floor outside
        let mut lin_mag = vec![1e-7_f64; n_fft / 2 + 1]; // -140 dB floor
        for k in 0..lin_mag.len() {
            let f = sr * k as f64 / n_fft as f64;
            let pass = match (hp_hz, lp_hz) {
                (Some(hp), Some(lp)) => f >= hp && f <= lp,
                (Some(hp), None) => f >= hp,
                (None, Some(lp)) => f <= lp,
                _ => true,
            };
            if pass { lin_mag[k] = 1.0; }
        }
        
        let phase = minimum_phase_from_magnitude(&lin_mag, n_fft);
        // Reconstruct impulse via IFFT
        let impulse = build_impulse(&lin_mag, &phase, n_fft);
        
        let peak_idx = impulse.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(i, _)| i).unwrap();
        
        assert!(peak_idx <= expected_max_idx,
            "{}: peak idx={} > {} (cepstral floor leak)",
            label, peak_idx, expected_max_idx);
    }
}
```

Если `build_impulse` helper не существует — написать локально в тесте
(IFFT через rustfft на complex H = mag * exp(j*phase)).

### 5. Регрессионные тесты

Запустить полный test suite:

```
cd src-tauri && cargo test --lib
```

Должно быть **179 + новый** PASS. Если что-то существующее упало —
STOP, проанализировать (возможно existing test ожидает старое
поведение со сдвигом).

### 6. Frontend регрессии

```
cd /Users/olegryzhikov/phaseforge && npm run test
```

Должно быть **104 PASS**.

### 7. Verify в UI после rebuild

После dev запуска:
- Открыть проект, sr=48000 (или 176.4k — bug на обеих).
- Band 1 LP=200 → Export → check status: **Pre-ring 0.00 ms**,
  **Causal 100%**.
- Band 2 BP 200-2000 → Export → то же.
- Band 3 HP=2000 → Export → то же.
- Проверить REW (если возможно): экспортированный WAV — фаза совпадает
  с моделью.

### 8. Commit

```
git add -A
git commit -m "$(cat <<'EOF'
fix: cepstral floor in min-phase reconstruction (b140.7)

minimum_phase_from_magnitude lacked ln|H| floor before FFT. On sparse
linear FFT grid (large noise_floor regions) ln|H|≈-17 created big
log-mag plateau → high-quefrency feature → constant group delay
in min-phase → impulse peak shifted from idx=0.

Symptoms: Pre-ring 4.25 ms (Band 1 LP=200) instead of 0 ms,
Causal 51% instead of 100%, REW phase did not match model
(linear-phase rotation up by frequency).

Fix: clamp ln|H| min to -60 dB (cepstral_floor) before FFT in
minimum_phase_from_magnitude.

Refinement divergence on HP=2000 (max_err 22→31 dB across iters)
disappears as derivative effect once peak is at idx=0.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

### 9. End-of-prompt automation

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

Сообщить:
- Версия в startup log = b140.7.
- Cargo tests / vitest — счётчик PASS.
- Pre-ring и Causal на трёх полосах.

## Что НЕ делать

- Не менять composite_phase_inner / iterative_refine.
- Не трогать noise_floor_db config (отдельный параметр, оставить -150).
- Не менять API generate_model_fir.
- Не делать вариант (B) с топологией SPL — отложен.

## Acceptance

- Pre-ring = 0.00 ms на всех трёх полосах.
- Causal = 100% (или ≥99%).
- iterative_refine на Band 3 max_err не растёт по iter.
- 179+ cargo + 104 vitest PASS.

## Правила

- Commit только после all checks PASS.
- Без нарратива.

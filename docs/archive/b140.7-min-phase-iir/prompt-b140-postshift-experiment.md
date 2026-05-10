# Промт для Code: experimental post-IFFT shift для min-phase

**Тип:** experimental fix без bump и коммита. Цель — выяснить какие
тесты fail после post-shift, до решения о merge.

## Step 0

```
cd /Users/olegryzhikov/phaseforge
git status
git diff --stat
```

Должно быть clean (диагностика откачена). Если modified — STOP.

Также добавить startup-маркер:
```rust
tracing::info!("[DIAG ACTIVE] post-shift: rotate impulse peak to idx=0");
```

## Контекст

Cepstral floor подход (вариант 1) не работает: -60 dB не убирает peak
shift на sparse spectra, -10 dB ломает magnitude fidelity. Cliff 150
dB между passband и noise_floor неизбывен в текущей архитектуре.

Альтернатива (вариант 2): post-IFFT shift импульса. Math: после
initial IFFT находим peak idx, делаем circular `rotate_left(peak_idx)`.
Magnitude сохраняется точно (circular shift = exp(-jωτ) множитель).
Удаляется **constant group delay** — артефакт cepstral leak — а это и
есть та linear-phase компонента которой в true min-phase быть не должно.

REPhase делает peak-at-0 by construction. Это эталон.

## Что нужно сделать

### 1. Apply post-shift

Найти место сразу после initial IFFT в FIR pipeline (где сейчас стоит
`[DIAG-MP] post-IFFT peak: idx=...` лог в diagnostic — там же).

Добавить:

```rust
// b140.7 experiment: post-IFFT shift to remove cepstral-leak constant group delay.
// Min-phase impulse should peak at samples[0]; cepstral artifact on sparse spectra
// shifts peak. Circular rotation preserves magnitude exactly while removing the
// linear-phase component that should not exist in true min-phase.
let peak_idx = impulse.iter().enumerate()
    .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
    .map(|(i, _)| i).unwrap_or(0);
if peak_idx > 0 {
    impulse.rotate_left(peak_idx);
}
```

Применять **только для linear_phase_main=false** (path Min-Phase /
Composite). Linear-phase path — peak в центре by construction, не
трогать.

Если IFFT повторяется в `iterative_refine` — применить тот же shift
после **каждой** итерации. Это обеспечит сходимость на peak-at-0.

### 2. Запустить полный test suite

```
cd src-tauri && cargo test --lib 2>&1 | tail -80
```

Записать в `docs/postshift-test-report.md`:
- Сколько PASS / FAIL.
- Полный list FAIL tests с краткой причиной (assert message).
- Для каждого FAIL: классифицировать
  - **(a) golden snapshot / hash** — captured buggy behavior, нужен update.
  - **(b) functional invariant** (magnitude RMS, energy conservation,
    realized matches target) — STOP, post-shift ломает функциональность.
  - **(c) phase-related** — ожидаемо, post-shift меняет phase константой,
    update.

### 3. NEW тест на peak-at-zero

Добавить под `#[cfg(test)]` в `src-tauri/src/dsp/phase.rs` или
`src-tauri/src/fir/mod.rs`:

```rust
#[test]
fn min_phase_impulse_peaks_at_zero_for_sparse_spectrum() {
    // Three production-like scenarios
    for (label, hp, lp, expected_max_peak_idx) in [
        ("LP=200 (7% active)", None::<f64>, Some(200.0_f64), 5),
        ("BP 200-2000 (72%)", Some(200.0), Some(2000.0), 5),
        ("HP=2000 (94%)", Some(2000.0), None, 5),
    ] {
        // ... build sparse mag, generate FIR, find peak ...
        assert!(peak_idx <= expected_max_peak_idx,
            "{}: peak idx={} > {} (post-shift not applied)",
            label, peak_idx, expected_max_peak_idx);
    }
}
```

### 4. Acceptance в чате

Прислать report:
1. Cargo: X PASS / Y FAIL.
2. Для каждого FAIL — категория (a/b/c).
3. Если только (a) и (c) — рекомендуется merge (надо обновить snapshots).
4. Если есть (b) — STOP, не merge, обсудить.

### 5. Что НЕ делать

- Не bumping.
- Не коммитить.
- Не обновлять golden snapshots на этом этапе (только классифицировать).
- Не запускать UI verify до решения о merge.
- Если cargo чисто PASS на новом тесте — обязательно проверить, что
  shift действительно применяется (через лог `peak_idx=N` rotated to 0).

### 6. End-of-prompt

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

В startup должна быть строка `[DIAG ACTIVE] post-shift: rotate
impulse peak to idx=0`.

## Правила

- Без нарратива.
- Один report.
- Без коммита до решения user-а.

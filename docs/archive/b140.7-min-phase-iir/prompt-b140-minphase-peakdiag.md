# Промт для Code: лог peak-index после initial IFFT для Min-Phase Composite

**Тип:** диагностический log. Без bump, без коммита.

## Контекст

Подтверждено: все Lin-φ checkbox off → linear_phase_main=false →
**effective_linear=false** → должен быть true min-phase путь с peak
в samples[0]. Но импульс имеет Pre-ring 4.25 ms (Band 1 LP=200,
sr=176.4k) — peak смещён от начала.

Подозрение из `docs/min-phase-trace.md` раздел 8.2: initial IFFT
после `minimum_phase_from_magnitude(base_mag)` где большая часть
бинов прижата к noise_floor — numerical artifact даёт сдвиг peak.
Half-window после IFFT пропускает off-centre без проверки.

## Что нужно сделать

### 1. Лог peak-index в трёх местах

В `src-tauri/src/fir/mod.rs` или `helpers.rs` (где выполняется IFFT
для Composite + linear_phase_main=false):

**(а) Сразу после initial IFFT (до half-window):**

```rust
// Diagnostic: peak position in raw impulse before windowing
let raw_peak_idx = impulse.iter().enumerate()
    .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
    .map(|(i, _)| i).unwrap_or(0);
let raw_peak_val = impulse[raw_peak_idx];
tracing::info!(
    "[DIAG-MP] post-IFFT peak: idx={} val={:.4e} taps={} (expect idx=0 for min-phase)",
    raw_peak_idx, raw_peak_val, impulse.len()
);
```

**(б) После half-window:**

```rust
let win_peak_idx = impulse.iter().enumerate()
    .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
    .map(|(i, _)| i).unwrap_or(0);
tracing::info!(
    "[DIAG-MP] post-half-window peak: idx={} val={:.4e}",
    win_peak_idx, impulse[win_peak_idx]
);
```

**(в) Внутри iter loop (если IFFT повторяется в iterative_refine):**

```rust
let iter_peak_idx = impulse.iter().enumerate()
    .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
    .map(|(i, _)| i).unwrap_or(0);
tracing::info!(
    "[DIAG-MP] iter={} post-IFFT peak: idx={} val={:.4e}",
    iter, iter_peak_idx, impulse[iter_peak_idx]
);
```

### 2. Лог fraction бинов выше noise_floor

Перед IFFT добавить:

```rust
let nf = config.noise_floor_db;
let active = base_mag.iter().filter(|&&db| db > nf + 0.1).count();
tracing::info!(
    "[DIAG-MP] base_mag stats: active_bins={}/{} ({:.1}%) above noise_floor={}",
    active, base_mag.len(), 100.0 * active as f64 / base_mag.len() as f64, nf
);
```

### 3. Запуск

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

Сообщить пользователю:
- Открыть текущий проект, sr=176.4k, 64K taps.
- Перейти на Band 1 (LP=200), вкладка Export → triggers FIR rebuild.
- Скопировать **все строки** `[DIAG-MP]` из терминала.
- Перейти на Band 3 (HP=2000), вкладка Export.
- Скопировать `[DIAG-MP]`.
- Прислать оба блока.

### 4. Что НЕ делать

- Не фиксить.
- Не bumping.
- Не коммитить.
- Не теоретизировать про windowing fix до получения чисел.

## Ожидаемое

- **Если raw_peak_idx ≈ 0** (например 0..50) на обоих полосах → IFFT
  даёт правильный min-phase, bug в других местах (windowing? meta-shift?).
- **Если raw_peak_idx >> 0** (например ~750 для Band 1 чтобы дать
  Pre-ring 4.25 ms на 176.4k) → bug в minimum_phase_from_magnitude
  / IFFT для sparse spectrum. Тогда фикс — либо threshold на active_bins,
  либо peak-shift к началу для compensate.
- Сравнение Band 1 (active_bins ~37) vs Band 3 (active_bins ~много)
  покажет зависимость от sparseness.

## Правила

- Без нарратива.
- Read-only diagnostic.
- Один step.

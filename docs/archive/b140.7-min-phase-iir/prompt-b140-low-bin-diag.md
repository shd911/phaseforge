# Промт для Code: диагностика low-freq linear FFT bins на разных sr

**Тип:** временный диагностический log. Без bump, без коммита.

## Контекст

Регрессия rolloff shift на 44.1/48 кГц после b140.5. Гипотеза:
linear FFT bins ниже log grid f_min (5 Hz) получают boundary clamp на
log[0] target_mag. На низких sr таких bins больше → больше DC energy →
shift apparent rolloff.

Нужны фактические значения linear FFT bins[0..10] target_mag для 48
vs 176 кГц на одинаковом проекте.

## Что нужно сделать

### 1. Найти точку interp log → linear FFT в Rust

```
grep -n "interp_log\|interpolate.*linear.*grid\|linear.*FFT" src-tauri/src/dsp src-tauri/src/fir
```

Скорее всего это:
- `src-tauri/src/dsp/impulse.rs:compute_impulse_response` (где target_mag/phase mapping на linear grid).
- Или `src-tauri/src/fir/mod.rs:generate_model_fir` после получения target.

Найти точное место.

### 2. Добавить debug log

После interp log → linear FFT grid:

```rust
// Diagnostic: log first 10 linear FFT bins to detect boundary clamp
tracing::info!(
    "[DIAG] target_mag bins 0..10: bin_width={:.3} Hz, vals={:?}",
    sr / fft_size as f64,
    &target_mag_linear[0..10.min(target_mag_linear.len())],
);
```

Точное имя переменной (`target_mag_linear`, `mag_full`, etc.) — на
усмотрение Code, главное вывести **значения первых 10 bins linear
spectrum** перед IFFT.

### 3. Запуск и сбор данных

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
- Открыть flat проект.
- Выбрать sr=48000, нажать Export.
- Скопировать строку `[DIAG]` из терминала.
- Переключить sr=176400, нажать Export.
- Скопировать строку `[DIAG]`.

Прислать оба `[DIAG]` блока.

### 4. Что НЕ делать

- Не фиксить код по гипотезе до получения evidence.
- Не bumping.
- Не коммитить.

## Ожидаемые данные

- На 48k bin width ≈ 0.73 Hz, bins 0..6 freq 0..4.4 Hz < 5 Hz log_min.
  Если все имеют похожее non-low значение (например -90 dB) — clamp подтверждён.
- На 176k bin width ≈ 2.69 Hz, только bin 1 < 5 Hz. Остальные >= 5.
  Bin 1 ~ -90 dB, bins 2+ должны drop сильно.

## Правила

- Без нарратива.
- Read-only diagnostic.

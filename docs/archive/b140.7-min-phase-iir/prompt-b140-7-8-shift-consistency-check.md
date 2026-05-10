# Промт для Code: b140.7.8 — verify shift consistency между pad и phase correction

**Тип:** read + small diag patch + bump. Без новых features. Проверить
что shift в padding и в phase delay correction — одно и то же значение.

## Step 0

```
cd /Users/olegryzhikov/phaseforge
git status
```

## Контекст

После b140.7.7 user сообщает что на 48k phase не достигает 0 на Nyquist
(должна, т.к. shift=N/2=32768 чётный → π·shift mod 2π = 0).

Гипотеза: shift в padding (= N/2 после b140.7.7) и shift в phase delay
correction — разные значения. Например padding использует фиксированный
N/2, а correction всё ещё `N/2 - raw_peak_idx`. Или наоборот.

Если они синхронизированы → дальше копаем глубже.

## Что нужно сделать

### 1. Bump до b140.7.8

- `tauri.conf.json` → b140.7.8.
- `lib.rs` → b140.7.8.
- `version.ts` → b140.7.8.

### 2. Прочесть iir_path.rs end-to-end (read-only)

Найти:
- Строку где applies padding shift (b140.7.6/7 logic).
- Строку где computes realized_phase с delay correction.
- Проверить что **то же самое** значение `shift` используется в обоих
  местах.

Если нашёл расхождение — исправить так чтобы оба места использовали
одну переменную `let shift = half;` (= N/2).

### 3. Diag log два значения

Добавить временно:
```rust
tracing::info!(
    "[CENTER DIAG] pad_shift={} phase_correction_shift={} N={}",
    pad_shift_value, phase_correction_shift_value, n
);
```

Где `pad_shift_value` — фактический shift применённый в padding,
`phase_correction_shift_value` — фактический shift применённый в delay
correction realized_phase.

Поместить **внутри** generate_min_phase_fir_iir.

### 4. Build + run

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

Сообщить user-у:
- Title bar = b140.7.8.
- Открыть Band 3 HP=2000 sr=48k → Export tab.
- В терминале:
  ```bash
  tail -100 /tmp/phaseforge-dev.log | grep "CENTER DIAG"
  ```
  Прислать строки.

### 5. Что покажет

Если pad_shift == phase_correction_shift == 32768 → код консистентен,
phase Nyquist должна быть 0 математически. Если user всё равно видит
не-ноль → bug в чём-то ещё (например в plot UI extrapolation или в
формуле correction).

Если pad_shift != phase_correction_shift → нашли несоответствие.
Зафиксировать одно значение для обоих.

### 6. End-of-prompt — в шаге 4.

## Что НЕ делать

- Не добавлять Hilbert / cepstral / архитектурные изменения.
- Не менять impulse generation logic.
- Только verify что pad shift == correction shift == N/2.

## Acceptance

- Title bar = b140.7.8.
- `[CENTER DIAG]` строка в логе.
- Если код консистентен — отчёт user-у с цифрами.
- Если расхождение — fix одной переменной.

## Правила

- Без нарратива.
- Один short report.

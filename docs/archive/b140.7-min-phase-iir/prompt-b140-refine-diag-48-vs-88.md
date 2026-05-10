# Промт для Code: refinement convergence 48k vs 88k

**Тип:** диагностический log. Без bump, без коммита.

## Контекст

Регрессия rolloff shift в FIR при sr=44.1/48 кГц. На 88+ кГц нет
проблемы (подтверждено пользователем). Bandpass HP=200 LR4 + LP=2000
LR4 на flat файле.

Предыдущая гипотеза (clamp низких linear FFT bins ниже log_min) **не
подтвердилась** — bins 0..6 на 48k имеют значение -416 dB (эффективно
silent), не могут влиять на passband 200-2000.

iterative_refine метрики:
- 48k: iter1 0.787 → iter3 0.545 dB (плохо сходится)
- 176k: iter1 0.528 → iter3 0.229 dB (нормально)

Граница между 48 и 88 → нужно сравнить refinement convergence на этих
двух sr на одинаковых freq точках passband.

## Что нужно сделать

### 1. Найти iterative_refine

```
grep -n "iterative_refine\|fn refine\|realized_mag\|excess_db" src-tauri/src/fir
```

Это `src-tauri/src/fir/helpers.rs:iterative_refine` или подобное.

### 2. Добавить per-iteration log на ключевых частотах

Внутри loop iterative_refine, после получения `realized_mag` на log
grid, найти индексы ближайшие к ключевым freq и вывести значения:

```rust
// Diagnostic: realized_mag at key passband frequencies for sr regression
let key_freqs = [100.0, 200.0, 400.0, 1000.0, 2000.0, 4000.0]; // HP*0.5, HP, HP*2, mid, LP, LP*2
let key_vals: Vec<(f64, f64, f64)> = key_freqs.iter().map(|&f| {
    // find nearest log_grid index
    let idx = log_freqs.iter().enumerate()
        .min_by(|(_, a), (_, b)| (a.ln() - f.ln()).abs().partial_cmp(&(b.ln() - f.ln()).abs()).unwrap())
        .map(|(i, _)| i).unwrap_or(0);
    (f, target_mag_db[idx], realized_mag_db[idx])
}).collect();
tracing::info!(
    "[DIAG-REFINE] iter={} sr={} key_freqs (f, target_db, realized_db): {:?}",
    iter, sr, key_vals
);
```

Точные имена переменных (`log_freqs`, `target_mag_db`, `realized_mag_db`,
`iter`, `sr`) — на усмотрение Code, главное чтобы вывод включал:
- iter number
- sr
- 6 пар (target_db, realized_db) на ключевых freq

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
- Открыть flat проект, HP=200 LR4 + LP=2000 LR4.
- sr=48000, нажать Export, скопировать **все строки** `[DIAG-REFINE]`.
- sr=88200, нажать Export, скопировать `[DIAG-REFINE]`.
- Прислать оба блока.

### 4. Что НЕ делать

- Не фиксить refinement до получения evidence.
- Не bumping.
- Не коммитить.
- Не теоретизировать про window leakage / FFT size до получения чисел.

## Ожидаемое

Сравнение realized_db в passband на 48k vs 88k покажет:
- Если на 48k realized отстаёт от target в районе HP/LP → сходимость
  refinement сама по себе sr-dependent.
- Если на обоих sr realized близок к target → bug не в refinement,
  искать дальше (FIR truncation, window applied к impulse, IFFT
  size relative to fft_size).

## Правила

- Без нарратива.
- Read-only diagnostic.
- Один step.

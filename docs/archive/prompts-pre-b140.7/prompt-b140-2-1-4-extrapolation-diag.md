# Промт для Code: диагностика per-band freq диапазонов

**Тип:** diagnostic. Без bump, без коммита.

## Контекст

В b140.2.1.3 New SUM показывает Band 5 (supertweeter) magnitude на
низких частотах (до 120 Гц), хотя реальные данные supertweeter
начинаются с ~200 Гц. Гипотеза: `resampleOntoGrid` экстраполирует
measurement за пределы исходного freq диапазона.

## Что нужно сделать

### 1. Расширить diff test (`tests/e2e_sum_real_project.rs`)

Добавить диагностический output:

```rust
// Per-band freq range и magnitude на сложных точках
for (i, b) in project.bands.iter().enumerate() {
    if let Some(m) = &b.measurement {
        let f_lo = m.freq.first().copied().unwrap_or(0.0);
        let f_hi = m.freq.last().copied().unwrap_or(0.0);
        eprintln!("Band {} ({}): measurement freq [{:.1}, {:.1}] Hz, {} points",
            i, b.name, f_lo, f_hi, m.freq.len());
    }
}

// Common grid evaluateSum
eprintln!("New common grid: [{:.1}, {:.1}] Hz, {} points",
    new_result.freq[0], new_result.freq.last().unwrap(), new_result.freq.len());

// Per-band resampled measurement (через port resampleOntoGrid)
// На частотах ВНЕ исходного диапазона — что возвращает?
for i in 0..bands.len() {
    let band_freq_hi = project.bands[i].measurement.as_ref().unwrap().freq.last().copied().unwrap_or(0.0);
    let band_freq_lo = project.bands[i].measurement.as_ref().unwrap().freq.first().copied().unwrap_or(0.0);

    // Sample resampled measurement на нескольких контрольных частотах
    for &test_f in &[5.0_f64, 50.0, 200.0, 1000.0, 10000.0, 30000.0] {
        let resampled = port_resample_onto_grid(&new_result.per_band[i], test_f);
        let inside = test_f >= band_freq_lo && test_f <= band_freq_hi;
        eprintln!("  Band {} resampled mag at {} Hz: {:.2} dB (in-range: {})",
            i, test_f, resampled, inside);
    }
}
```

### 2. Проверить TypeScript `resampleOntoGrid` логику

Прочитать в `src/lib/band-evaluator.ts` функцию `resampleOntoGrid`.
Понять:
- Что возвращает на freq < input.freq[0] (extrapolation метод)?
- Что возвращает на freq > input.freq[last]?
- Совпадает ли с `interpolate_log` в Rust?

Если `resampleOntoGrid` делает constant extrapolation (повтор крайних
значений) — это и есть баг. Должен возвращать **−200 dB** (или -∞)
для freq вне диапазона, чтобы coherent sum не учитывал phantom data.

### 3. Что прислать обратно

```
Per-band measurement freq ranges:
  Band 1 (Purifi NF):  [5.0, 22000.0] Hz, 950 pts
  Band 2 (8x merged):  [...]
  Band 3 (6.5M merged): [...]
  Band 4 (74-FF):       [200.0, 22000.0] Hz, 800 pts  ← начинается с 200
  Band 5 (25FF super):  [500.0, 22000.0] Hz, 750 pts  ← начинается с 500

New common grid: [5.0, 22000.0] Hz, 950 points

Band 5 resampled mag at 5 Hz: -65 dB (in-range: false) ← если -65 а не -200,
                                                          extrapolates неправильно
Band 5 resampled mag at 50 Hz: -65 dB (in-range: false)
Band 5 resampled mag at 500 Hz: -3 dB (in-range: true)
Band 5 resampled mag at 10000 Hz: 0 dB (in-range: true)
```

Это локализует расхождение: либо `resampleOntoGrid` экстраполирует, либо
`interpolate_log` (Rust) делает то же.

## Что НЕ делать

- Не правлять `resampleOntoGrid` пока не подтверждено что extrapolation
  is the bug.
- Не bumping версию.
- Не коммитить.

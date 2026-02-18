# 02 — Target Curve Engine

## Overview

Целевая кривая — это "идеал", к которому мы приводим АЧХ. Строится из комбинации компонентов.

## Components

### 1. Flat Reference

Базовая линия — горизонтальная прямая на заданном уровне (обычно 0 dB или средний SPL измерения).

```rust
fn flat_target(freq: &[f64], level_db: f64) -> Vec<f64> {
    vec![level_db; freq.len()]
}
```

### 2. High-Pass / Low-Pass Filters

Моделируют физические ограничения динамика / сабвуфера.

**Типы:**
- **Butterworth** — максимально плоская АЧХ в полосе пропускания
- **Bessel** — максимально линейная группа задержка (лучше для транзиентов)
- **Linkwitz-Riley** (LR2, LR4, LR8) — сумма двух каскадированных Butterworth = flat на crossover point

**Порядки:** 1st – 8th (6–48 dB/oct slope)

#### Magnitude Response

Для Butterworth N-го порядка:

```
H(f) = -10 * log10(1 + (f/fc)^(2*N))          // low-pass
H(f) = -10 * log10(1 + (fc/f)^(2*N))          // high-pass
```

Для Linkwitz-Riley: `H_LR(f) = 2 * H_Butterworth(f)` (удвоенный slope в dB).

Для Bessel: аналитическая формула через полиномы Бесселя, или табличная аппроксимация.

#### Super-Gaussian Approximation

Для более плавного перехода (без резонансного горба):

```
H(f) = exp(-0.5 * (f/fc)^(2*shape))
```

`shape` контролирует резкость: shape=1 → Gaussian, shape→∞ → brick wall.

### 3. Tilt

Наклон АЧХ — типичная коррекция для комнаты. Обычно -0.5 ... -1.5 dB/oct.

```rust
fn tilt(freq: &[f64], db_per_octave: f64, ref_freq: f64) -> Vec<f64> {
    freq.iter()
        .map(|&f| db_per_octave * (f / ref_freq).log2())
        .collect()
}
```

`ref_freq` обычно = 1 kHz (pivot point).

### 4. Shelving Filters

Low shelf и High shelf — для подъёма/спада басов или верхов.

```
H_low_shelf(f)  = gain_db / (1 + (f/fc)^2)     // simplified 1st order
H_high_shelf(f) = gain_db / (1 + (fc/f)^2)
```

Более точная реализация — через bilinear transform аналогового shelf прототипа с настраиваемым Q (slope).

### 5. House Curve Presets

Готовые профили:

| Preset | Description |
|--------|-------------|
| Flat | 0 dB everywhere |
| Harman | +bass shelf below 200Hz, slight HF rolloff |
| B&K / Brüel & Kjær | Classic room target with ~-1dB/oct tilt |
| X-curve (cinema) | -1.5dB/oct above 2kHz, -3dB/oct above 10kHz |
| Custom | User-defined via point editor |

---

## Data Model

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetCurve {
    pub reference_level_db: f64,
    pub tilt_db_per_octave: f64,
    pub tilt_ref_freq: f64,
    pub high_pass: Option<FilterConfig>,
    pub low_pass: Option<FilterConfig>,
    pub low_shelf: Option<ShelfConfig>,
    pub high_shelf: Option<ShelfConfig>,
    pub preset: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterConfig {
    pub filter_type: FilterType,  // Butterworth | Bessel | LR | SuperGaussian
    pub order: u8,                // 1..8
    pub freq_hz: f64,
    pub shape: Option<f64>,       // for SuperGaussian
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShelfConfig {
    pub freq_hz: f64,
    pub gain_db: f64,
    pub q: f64,
}
```

---

## Evaluation

```rust
/// Evaluate target curve at given frequencies
pub fn evaluate(target: &TargetCurve, freq: &[f64]) -> Vec<f64> {
    let mut result = vec![target.reference_level_db; freq.len()];
    
    // Add tilt
    add_tilt(&mut result, freq, target.tilt_db_per_octave, target.tilt_ref_freq);
    
    // Apply HP/LP
    if let Some(hp) = &target.high_pass {
        apply_highpass(&mut result, freq, hp);
    }
    if let Some(lp) = &target.low_pass {
        apply_lowpass(&mut result, freq, lp);
    }
    
    // Apply shelves
    if let Some(ls) = &target.low_shelf {
        apply_low_shelf(&mut result, freq, ls);
    }
    if let Some(hs) = &target.high_shelf {
        apply_high_shelf(&mut result, freq, hs);
    }
    
    result
}
```

---

## UI Interactions

- Слайдеры: HP freq, LP freq, order, tilt, shelf gain/freq/Q
- Preset dropdown
- Real-time update: каждое изменение слайдера → пересчёт → перерисовка
- Целевая кривая рисуется полупрозрачной линией поверх измерения
- "Error fill" — заливка между измерением и целью (зелёная = ок, красная = большое отклонение)

---

## Constraints

- При смене target → сбрасываются PEQ фильтры и FIR (пересчёт обязателен)
- Target всегда интерполируется на ту же сетку частот, что и измерение
- Reference level авто-подбирается: среднее измерения в диапазоне 200–2000 Hz

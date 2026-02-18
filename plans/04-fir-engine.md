# 04 — FIR Correction Engine

## Overview

FIR (Finite Impulse Response) — "хирургическая" коррекция. Берёт остаточную ошибку после PEQ и запекает точную коррекцию амплитуды + фазы в импульсный файл.

## Pipeline

```
Measurement + PEQ filters
    │
    ▼
Current Curve = Raw + Σ PEQ responses
    │
    ▼
Interpolate to Linear Frequency Grid (N/2+1 bins)
    │
    ▼
Correction Spectrum = Target - Current (per bin)
    │
    ▼
Boost/Cut Limiting
    │
    ▼
Phase Correction (strategy-dependent)
    │
    ▼
Complex Spectrum Assembly (Real + Imag)
    │
    ▼
Conjugate Symmetry (mirror for real output)
    │
    ▼
IFFT → Time Domain Impulse
    │
    ▼
Circular Shift (center the peak)
    │
    ▼
Window Function (Blackman / Kaiser / Tukey)
    │
    ▼
Normalize & Export WAV (f64 IEEE Float)
```

---

## Step 1: Linear Grid Interpolation

FFT требует равномерно распределённых точек в линейном пространстве.

```rust
pub struct FirConfig {
    pub tap_count: usize,          // 4096, 8192, 16384, 32768, 65536, 131072, 262144
    pub sample_rate: f64,          // 44100, 48000, 96000
    pub phase_strategy: PhaseStrategy,
    pub window: WindowType,
    pub max_boost_db: f64,         // default: 18.0
    pub noise_floor_db: f64,       // default: -60.0
}

let n_bins = config.tap_count / 2 + 1;  // unique frequency bins
let df = config.sample_rate / config.tap_count as f64;  // freq resolution
// Bins: 0, df, 2*df, ..., sample_rate/2
```

Для 65536 taps @ 48kHz: `df = 0.73 Hz`, `n_bins = 32769`.

---

## Step 2: Correction Spectrum

```rust
fn compute_correction(
    freq_bins: &[f64],
    current_mag_db: &[f64],
    target_mag_db: &[f64],
    config: &FirConfig,
) -> Vec<f64> {
    freq_bins.iter().enumerate().map(|(i, &f)| {
        let correction_db = target_mag_db[i] - current_mag_db[i];
        apply_limits(correction_db, target_mag_db[i], config)
    }).collect()
}
```

### Boost/Cut Limiting

```rust
fn apply_limits(correction_db: f64, target_db: f64, config: &FirConfig) -> f64 {
    if correction_db > 0.0 {
        // BOOST — aggressive limiting
        let limited = correction_db.min(config.max_boost_db);
        
        // Don't boost if target is below noise floor
        // (бессмысленно усиливать шум)
        if target_db < config.noise_floor_db {
            0.0
        } else {
            limited
        }
    } else {
        // CUT — minimal limiting (safe for amplifier)
        correction_db.max(-60.0)  // -60dB floor
    }
}
```

**Why asymmetric:** Boost → увеличивает мощность → может повредить динамик. Cut → уменьшает → безопасно.

---

## Step 3: Phase Correction

Три стратегии — определяют "характер" FIR фильтра.

### Strategy A: Linear Phase

```rust
fn linear_phase_correction(current_phase: &[f64]) -> Vec<f64> {
    // Simply invert the measured phase
    current_phase.iter().map(|&p| -p).collect()
}
```

**Result:** Все частоты приходят одновременно. Идеальные транзиенты.  
**Downside:** Pre-ringing (эхо до удара, не после). Заметно на ударных.

### Strategy B: Minimum Phase

```rust
fn minimum_phase_correction(correction_mag_db: &[f64], n_fft: usize) -> Vec<f64> {
    // Hilbert transform: derive phase from magnitude
    // 1. ln(|H|) → compute Hilbert transform → phase
    let log_mag: Vec<f64> = correction_mag_db.iter()
        .map(|&m| (10.0_f64.powf(m / 20.0)).ln())
        .collect();
    
    // 2. Hilbert transform via FFT
    let analytic = hilbert_transform(&log_mag, n_fft);
    
    // 3. Extract imaginary part = minimum phase
    analytic.iter().map(|c| -c.im).collect()  // negative for correction
}
```

**Result:** Нет pre-ringing. Фаза "минимально возможная" для данной амплитуды.  
**Downside:** Не корректирует excess group delay исходной системы.

### Strategy C: Mixed Phase (Best of Both)

```rust
pub struct MixedPhaseConfig {
    pub crossover_freq: f64,  // Hz, typically 200-500
    pub transition_width: f64, // octaves for smooth blend
}

fn mixed_phase_correction(
    current_phase: &[f64],
    correction_mag_db: &[f64],
    freq_bins: &[f64],
    config: &MixedPhaseConfig,
    n_fft: usize,
) -> Vec<f64> {
    let linear = linear_phase_correction(current_phase);
    let minimum = minimum_phase_correction(correction_mag_db, n_fft);
    
    // Crossfade: linear phase below crossover, minimum phase above
    freq_bins.iter().enumerate().map(|(i, &f)| {
        let blend = sigmoid_blend(f, config.crossover_freq, config.transition_width);
        // blend = 0 → linear phase, blend = 1 → minimum phase
        linear[i] * (1.0 - blend) + minimum[i] * blend
    }).collect()
}

fn sigmoid_blend(freq: f64, crossover: f64, width_octaves: f64) -> f64 {
    let octaves_from_xover = (freq / crossover).log2();
    let x = octaves_from_xover / width_octaves;
    1.0 / (1.0 + (-x * 5.0).exp())  // smooth sigmoid
}
```

**Rationale:**  
- НЧ (< 200Hz): linear phase. Pre-ringing на басах не слышно, зато фаза идеально ровная.
- ВЧ (> 500Hz): minimum phase. Pre-ringing на ВЧ заметно, minimum phase его убирает.
- Переход: плавный blend через sigmoid.

---

## Step 4: Complex Spectrum Assembly

```rust
fn assemble_spectrum(
    correction_mag_db: &[f64],
    correction_phase_rad: &[f64],
    n_fft: usize,
) -> Vec<Complex64> {
    let n_bins = n_fft / 2 + 1;
    
    // First half: positive frequencies
    let mut spectrum = Vec::with_capacity(n_fft);
    for i in 0..n_bins {
        let mag_linear = 10.0_f64.powf(correction_mag_db[i] / 20.0);
        let phase = correction_phase_rad[i];
        spectrum.push(Complex64::new(
            mag_linear * phase.cos(),
            mag_linear * phase.sin(),
        ));
    }
    
    // Second half: conjugate symmetry (required for real output)
    for i in 1..(n_fft / 2) {
        let j = n_fft / 2 - i;  // mirror index
        spectrum.push(spectrum[j].conj());
    }
    
    spectrum
}
```

---

## Step 5: IFFT

```rust
use rustfft::FftPlanner;

fn ifft(spectrum: &mut [Complex64]) {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(spectrum.len());
    fft.process(spectrum);
    
    // Normalize
    let n = spectrum.len() as f64;
    for s in spectrum.iter_mut() {
        *s /= n;
    }
}
```

**Optimization:** Использовать `realfft` крейт для real-valued output — вдвое меньше вычислений.

---

## Step 6: Circular Shift + Windowing

```rust
fn finalize_impulse(mut impulse: Vec<f64>, window: WindowType) -> Vec<f64> {
    let n = impulse.len();
    
    // 1. Circular shift: move peak to center
    let peak_idx = impulse.iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
        .unwrap().0;
    
    let shift = n / 2 - peak_idx;
    impulse.rotate_right(((shift % n as i64 + n as i64) % n as i64) as usize);
    
    // 2. Apply window
    let window_coeffs = generate_window(n, window);
    for (s, w) in impulse.iter_mut().zip(window_coeffs.iter()) {
        *s *= w;
    }
    
    impulse
}
```

### Window Types

| Window | Characteristics |
|--------|----------------|
| **Blackman** | Good sidelobe suppression (-58dB), moderate main lobe width. Default choice. |
| **Kaiser** (β=8-12) | Adjustable tradeoff. β=10 ≈ Blackman. β=14 → extreme sidelobe suppression. |
| **Tukey** (α=0.5) | Flat in center, tapered edges. Preserves more of the correction energy. |
| **Hann** | Simple, -31dB sidelobes. Lighter than Blackman. |

---

## Step 7: WAV Export

```rust
use hound::{WavWriter, WavSpec, SampleFormat};

fn export_wav(impulse: &[f64], path: &Path, sample_rate: u32, bit_depth: BitDepth) {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: match bit_depth {
            BitDepth::F32 => 32,
            BitDepth::F64 => 64,
        },
        sample_format: SampleFormat::Float,
    };
    
    let mut writer = WavWriter::create(path, spec).unwrap();
    
    match bit_depth {
        BitDepth::F32 => {
            for &s in impulse {
                writer.write_sample(s as f32).unwrap();
            }
        }
        BitDepth::F64 => {
            // hound doesn't support f64 natively — write raw bytes
            // ...custom implementation needed
        }
    }
    
    writer.finalize().unwrap();
}
```

**Note:** `hound` не поддерживает f64 из коробки. Для 64-bit float WAV — либо форкнуть hound, либо писать WAV header вручную (32 байта header + raw f64 data). Формат прост.

---

## Tap Count Guidelines

| Taps | Resolution @ 48kHz | Memory | Use Case |
|------|-------------------|--------|----------|
| 4096 | 11.7 Hz | 32 KB | Quick preview |
| 16384 | 2.9 Hz | 128 KB | Midrange correction |
| 65536 | 0.73 Hz | 512 KB | Full-range with bass |
| 131072 | 0.37 Hz | 1 MB | Precision bass correction |
| 262144 | 0.18 Hz | 2 MB | Maximum precision |

**Rule of thumb:** Нужно как минимум 2–3 периода на самой низкой корректируемой частоте. Для 20 Hz → `48000 / 20 * 3 = 7200` taps minimum → 8192 или 16384.

---

## Performance Targets

| Taps | Target Time (M1 Mac) |
|------|----------------------|
| 65536 | < 50ms |
| 131072 | < 100ms |
| 262144 | < 300ms |

`rustfft` на Apple Silicon должен уложиться с запасом.

---

## Pre-ringing Visualization

Для linear phase — показать пользователю pre-ringing:

1. Сгенерировать единичный импульс
2. Свернуть с FIR фильтром
3. Показать time-domain plot с отметкой "main impulse" и "pre-ring region"
4. Вычислить energy ratio: `pre_ring_energy / total_energy` в процентах

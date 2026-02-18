# 03 — PEQ Engine (Auto-Alignment)

## Overview

Параметрический EQ — "грубая" минимально-фазовая коррекция. Создаёт набор PK (peaking) фильтров для приближения АЧХ к целевой кривой.

## Pipeline

```
Raw Measurement
    │
    ▼
Variable Smoothing (psychoacoustic)
    │
    ▼
Error = Smoothed - Target
    │
    ▼
Greedy Iterative Fitting (up to 60 iterations)
    │
    ▼
Pruning (remove redundant filters)
    │
    ▼
[Optional] Joint L-BFGS Optimization
    │
    ▼
Final PEQ Filter Set
```

---

## Step 1: Variable Smoothing

```rust
pub fn variable_smoothing(freq: &[f64], mag: &[f64]) -> Vec<f64> {
    freq.iter().enumerate().map(|(i, &f)| {
        let fraction = if f < 100.0 {
            1.0 / 48.0   // detailed for room modes
        } else if f < 500.0 {
            1.0 / 12.0   // moderate
        } else {
            1.0 / 3.0    // coarse for HF
        };
        fractional_octave_smooth(freq, mag, i, fraction)
    }).collect()
}

fn fractional_octave_smooth(freq: &[f64], mag: &[f64], idx: usize, fraction: f64) -> f64 {
    let center = freq[idx];
    let k = (2.0_f64).powf(fraction / 2.0);
    let f_low = center / k;
    let f_high = center * k;
    
    // Weighted average of all points in [f_low, f_high]
    // Weight: triangular or rectangular
    // ...
}
```

---

## Step 2: Greedy Iterative Fitting

```rust
pub struct PeqBand {
    pub freq_hz: f64,
    pub gain_db: f64,
    pub q: f64,
    pub filter_type: BandType,  // PK, LS, HS
}

pub fn greedy_fit(
    freq: &[f64],
    smoothed: &[f64],
    target: &[f64],
    config: &AutoEqConfig,
) -> Vec<PeqBand> {
    let mut current = smoothed.to_vec();
    let mut filters = Vec::new();
    
    for _iter in 0..config.max_iterations {  // default: 60
        // 1. Compute error
        let error: Vec<f64> = current.iter()
            .zip(target.iter())
            .map(|(c, t)| c - t)
            .collect();
        
        // 2. Find peak error (with peak bias)
        let (peak_idx, peak_val) = find_peak_error(&error, config.peak_bias);
        
        // 3. Check convergence
        if peak_val.abs() < config.tolerance_db {
            break;
        }
        
        // 4. Create corrective filter
        let band = PeqBand {
            freq_hz: freq[peak_idx],
            gain_db: -peak_val,  // invert error
            q: heuristic_q(freq[peak_idx]),
            filter_type: BandType::PK,
        };
        
        // 5. Apply filter to current curve
        let filter_response = peaking_response(freq, &band);
        for (c, f) in current.iter_mut().zip(filter_response.iter()) {
            *c += f;
        }
        
        filters.push(band);
    }
    
    filters
}
```

### Peak Bias

Приоритет пиков над провалами — ключевая эвристика:

```rust
fn find_peak_error(error: &[f64], peak_bias: f64) -> (usize, f64) {
    // Weight positive errors (peaks) more than negative (dips)
    let weighted: Vec<f64> = error.iter().map(|&e| {
        if e > 0.0 { e * peak_bias } else { e }  // peak_bias = 1.5..2.0
    }).collect();
    
    // Find index of max absolute weighted error
    weighted.iter().enumerate()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
        .map(|(i, &v)| (i, error[i]))  // return original error value
        .unwrap()
}
```

**Rationale:**
- Подавление пика (cut) — акустически безопасно, не перегружает динамик
- Подъём провала (boost) — может перегрузить динамик, усилить искажения
- Провалы часто вызваны интерференцией (comb filtering) — неисправимы EQ

### Heuristic Q

```rust
fn heuristic_q(freq: f64) -> f64 {
    if freq < 80.0 {
        2.0    // wide — room modes are broad
    } else if freq < 300.0 {
        3.0    // moderate
    } else if freq < 1000.0 {
        4.0    // narrower
    } else {
        5.0    // narrow — HF corrections need precision
    }
}
```

---

## Step 3: Peaking Filter Response

Стандартная формула PK фильтра (RBJ Audio EQ Cookbook):

```rust
fn peaking_response(freq: &[f64], band: &PeqBand) -> Vec<f64> {
    freq.iter().map(|&f| {
        let w = f / band.freq_hz;
        let num = 1.0 + (band.gain_db / 20.0).powi(2) * w.powi(2) / band.q.powi(2);
        let den = (1.0 - w.powi(2)).powi(2) + w.powi(2) / band.q.powi(2);
        // Simplified magnitude-only approximation:
        // Exact: use biquad coefficient computation
        10.0 * (num / den).log10()
    }).collect()
}
```

**Better approach:** Вычислять через biquad коэффициенты (a0, a1, a2, b0, b1, b2) → transfer function `H(z)` → magnitude. Это точнее и нужно для экспорта.

```rust
pub struct BiquadCoeffs {
    pub b0: f64, pub b1: f64, pub b2: f64,
    pub a0: f64, pub a1: f64, pub a2: f64,
}

impl BiquadCoeffs {
    pub fn peaking(freq_hz: f64, gain_db: f64, q: f64, sample_rate: f64) -> Self {
        let a = 10.0_f64.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * freq_hz / sample_rate;
        let alpha = w0.sin() / (2.0 * q);
        
        BiquadCoeffs {
            b0: 1.0 + alpha * a,
            b1: -2.0 * w0.cos(),
            b2: 1.0 - alpha * a,
            a0: 1.0 + alpha / a,
            a1: -2.0 * w0.cos(),
            a2: 1.0 - alpha / a,
        }
    }
    
    pub fn magnitude_db(&self, freq_hz: f64, sample_rate: f64) -> f64 {
        let w = 2.0 * PI * freq_hz / sample_rate;
        let ejw = Complex64::new(w.cos(), w.sin());
        let ejw2 = Complex64::new((2.0*w).cos(), (2.0*w).sin());
        
        let num = self.b0 + self.b1 * ejw + self.b2 * ejw2;
        let den = self.a0 + self.a1 * ejw + self.a2 * ejw2;
        20.0 * (num / den).norm().log10()
    }
}
```

---

## Step 4: Pruning

```rust
pub fn prune_filters(
    freq: &[f64],
    raw: &[f64],
    target: &[f64],
    filters: &mut Vec<PeqBand>,
    tolerance_db: f64,
) {
    let mut i = 0;
    while i < filters.len() {
        // Try removing filter i
        let removed = filters.remove(i);
        
        // Compute resulting curve without this filter
        let curve = apply_all_filters(freq, raw, filters);
        let max_error = max_abs_error(&curve, target);
        
        if max_error <= tolerance_db * 1.2 {  // dynamic tolerance: slightly relaxed
            // Filter was redundant, keep it removed
            continue;
        } else {
            // Filter is needed, put it back
            filters.insert(i, removed);
            i += 1;
        }
    }
}
```

---

## Step 5: Joint Optimization (Optional, Advanced)

После greedy + pruning, все оставшиеся фильтры оптимизируются совместно.

**Variables:** Для каждого из N фильтров: `(freq, gain, q)` → 3N параметров.

**Objective:** `minimize Σ w(f) * (current(f) - target(f))²` (weighted MSE).

Weight `w(f)` — психоакустическая (больший вес на средних частотах 200–5000 Hz).

**Solver:** `argmin` крейт с L-BFGS backend.

```rust
use argmin::prelude::*;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;

struct PeqObjective { freq: Vec<f64>, raw: Vec<f64>, target: Vec<f64>, weights: Vec<f64> }

impl CostFunction for PeqObjective {
    type Param = Vec<f64>;  // [f1, g1, q1, f2, g2, q2, ...]
    type Output = f64;
    
    fn cost(&self, params: &Self::Param) -> Result<f64, Error> {
        let bands = unpack_bands(params);
        let curve = apply_all_filters(&self.freq, &self.raw, &bands);
        Ok(weighted_mse(&curve, &self.target, &self.weights))
    }
}
```

**Bounds:** Freq ±1 octave from greedy, Gain ±20dB, Q: 0.3–30.

---

## Export Formats

### Generic (JSON)
```json
[
  {"freq": 63.0, "gain": -4.2, "q": 2.1, "type": "PK"},
  {"freq": 125.0, "gain": -6.8, "q": 3.0, "type": "PK"}
]
```

### Equalizer APO
```
Filter 1: ON PK Fc 63.0 Hz Gain -4.2 dB Q 2.1
Filter 2: ON PK Fc 125.0 Hz Gain -6.8 dB Q 3.0
```

### miniDSP
CSV export compatible with miniDSP plugin import.

### Roon DSP
JSON format for Roon's parametric EQ.

---

## Config

```rust
pub struct AutoEqConfig {
    pub max_iterations: usize,     // default: 60
    pub tolerance_db: f64,         // default: 1.0
    pub peak_bias: f64,            // default: 1.5
    pub max_boost_db: f64,         // default: 6.0
    pub max_cut_db: f64,           // default: 18.0
    pub freq_range: (f64, f64),    // default: (20.0, 20000.0)
    pub enable_joint_opt: bool,    // default: false
}
```

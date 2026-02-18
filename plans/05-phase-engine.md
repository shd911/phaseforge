# 05 — Phase Engine & Advanced Features

## Phase Analysis

### Group Delay

Групповая задержка — производная фазы по частоте. Показывает, на сколько опаздывает каждая частотная компонента.

```rust
pub fn group_delay(freq: &[f64], phase_rad: &[f64]) -> Vec<f64> {
    let mut gd = Vec::with_capacity(freq.len());
    
    for i in 0..freq.len() {
        let dphi = if i == 0 {
            phase_rad[1] - phase_rad[0]
        } else if i == freq.len() - 1 {
            phase_rad[i] - phase_rad[i - 1]
        } else {
            (phase_rad[i + 1] - phase_rad[i - 1]) / 2.0
        };
        
        let dw = if i == 0 {
            2.0 * PI * (freq[1] - freq[0])
        } else if i == freq.len() - 1 {
            2.0 * PI * (freq[i] - freq[i - 1])
        } else {
            2.0 * PI * (freq[i + 1] - freq[i - 1]) / 2.0
        };
        
        gd.push(-dphi / dw);  // seconds
    }
    
    gd
}
```

### Excess Group Delay

Это то, что реально слышно. Минимально-фазовая система имеет "естественную" групповую задержку от своей АЧХ. Excess = total - minimum_phase.

```rust
pub fn excess_group_delay(
    freq: &[f64],
    total_phase: &[f64],
    magnitude_db: &[f64],
    n_fft: usize,
) -> Vec<f64> {
    let min_phase = compute_minimum_phase(magnitude_db, n_fft);
    let total_gd = group_delay(freq, total_phase);
    let min_gd = group_delay(freq, &min_phase);
    
    total_gd.iter().zip(min_gd.iter())
        .map(|(t, m)| t - m)
        .collect()
}
```

### Minimum Phase via Hilbert Transform

```rust
pub fn compute_minimum_phase(magnitude_db: &[f64], n_fft: usize) -> Vec<f64> {
    // 1. Log magnitude
    let log_mag: Vec<f64> = magnitude_db.iter()
        .map(|&m| {
            let linear = 10.0_f64.powf(m / 20.0).max(1e-10);
            linear.ln()
        })
        .collect();
    
    // 2. Extend to full FFT size (mirror)
    let mut full = vec![0.0; n_fft];
    for i in 0..log_mag.len().min(n_fft / 2 + 1) {
        full[i] = log_mag[i];
        if i > 0 && i < n_fft / 2 {
            full[n_fft - i] = log_mag[i];
        }
    }
    
    // 3. FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut complex: Vec<Complex64> = full.iter()
        .map(|&r| Complex64::new(r, 0.0))
        .collect();
    fft.process(&mut complex);
    
    // 4. Apply causal window (Hilbert multiplier)
    complex[0] *= Complex64::new(1.0, 0.0);  // DC: keep
    for i in 1..n_fft / 2 {
        complex[i] *= Complex64::new(2.0, 0.0);  // positive freq: ×2
    }
    // Nyquist: keep
    for i in (n_fft / 2 + 1)..n_fft {
        complex[i] = Complex64::new(0.0, 0.0);  // negative freq: zero
    }
    
    // 5. IFFT
    let ifft = planner.plan_fft_inverse(n_fft);
    ifft.process(&mut complex);
    let n = n_fft as f64;
    
    // 6. Minimum phase = imaginary part of result
    complex.iter().take(n_fft / 2 + 1)
        .map(|c| c.im / n)
        .collect()
}
```

---

## Excess Group Delay Targeting

Вместо коррекции абсолютной фазы — корректируем только excess group delay. Это сохраняет "естественную" минимально-фазовую реакцию динамика и убирает только аномалии (порт, кроссовер, комнатные отражения).

```rust
pub enum PhaseTarget {
    /// Zero phase (linear phase) — all frequencies arrive simultaneously
    ZeroPhase,
    
    /// Zero excess group delay — preserve minimum-phase behavior
    ZeroExcessGroupDelay,
    
    /// Custom target group delay curve
    CustomGroupDelay(Vec<(f64, f64)>),  // (freq_hz, delay_ms)
}
```

---

## Multi-Point Averaging

### Problem

Одна точка измерения = одна позиция головы. Коррекция может быть идеальной в одной точке и ужасной в 10 см рядом.

### Solution

Усреднение нескольких измерений с разных позиций:

```rust
pub struct MultiPointConfig {
    pub measurements: Vec<Measurement>,
    pub weights: Vec<f64>,            // per-measurement weight
    pub method: AveragingMethod,
}

pub enum AveragingMethod {
    /// Simple energy average (magnitude only)
    EnergyAverage,
    
    /// Complex average (preserves phase information)
    ComplexAverage,
    
    /// Spatially-weighted: center position gets highest weight
    SpatiallyWeighted { center_idx: usize, falloff: f64 },
}
```

### Energy Average (Magnitude Only)

```rust
fn energy_average(measurements: &[Measurement], weights: &[f64]) -> Vec<f64> {
    let n_freq = measurements[0].magnitude.len();
    let total_weight: f64 = weights.iter().sum();
    
    (0..n_freq).map(|i| {
        let sum: f64 = measurements.iter().zip(weights.iter())
            .map(|(m, &w)| {
                let linear = 10.0_f64.powf(m.magnitude[i] / 10.0);
                linear * w
            })
            .sum();
        10.0 * (sum / total_weight).log10()
    }).collect()
}
```

### Spatially Weighted

Позиция центра = вес 1.0. Каждые 10 см от центра → вес уменьшается по Gaussian.

```rust
fn spatial_weights(positions: &[(f64, f64, f64)], center_idx: usize, sigma: f64) -> Vec<f64> {
    let center = positions[center_idx];
    positions.iter().map(|p| {
        let dist = ((p.0 - center.0).powi(2) + 
                    (p.1 - center.1).powi(2) + 
                    (p.2 - center.2).powi(2)).sqrt();
        (-dist.powi(2) / (2.0 * sigma.powi(2))).exp()
    }).collect()
}
```

---

## A/B Comparison

### Realtime Preview (Optional, если возможно через Tauri)

Если подключить системный audio output:

```
Input audio → Convolution with FIR → Output
              ↕ Toggle
Input audio → Bypass → Output
```

Но это сложно и не в приоритете. Проще:

### Visual A/B

- Переключатель "Before / After" на графике
- "Before" = raw measurement
- "After" = measurement + PEQ + FIR
- Overlay mode: оба одновременно, разные цвета
- Разница (error): отдельный plot внизу

---

## Batch Processing

```rust
pub struct BatchJob {
    pub input_dir: PathBuf,
    pub output_dir: PathBuf,
    pub target: TargetCurve,
    pub peq_config: AutoEqConfig,
    pub fir_config: FirConfig,
    pub naming: NamingConvention,  // e.g., "{input_name}_corrected.wav"
}

pub async fn run_batch(job: BatchJob, progress: impl Fn(f64)) -> Result<BatchResult> {
    let files = find_measurement_files(&job.input_dir)?;
    let total = files.len() as f64;
    
    for (i, file) in files.iter().enumerate() {
        let measurement = import_measurement(file)?;
        let target = evaluate(&job.target, &measurement.freq);
        let peq = greedy_fit(&measurement.freq, &measurement.magnitude, &target, &job.peq_config);
        let fir = generate_fir(&measurement, &peq, &target, &job.fir_config)?;
        
        let output_name = format_name(&job.naming, file);
        export_wav(&fir, &job.output_dir.join(output_name), job.fir_config.sample_rate as u32, BitDepth::F64)?;
        
        progress((i + 1) as f64 / total);
    }
    
    Ok(BatchResult { processed: files.len() })
}
```

---

## Project Save/Load

```rust
#[derive(Serialize, Deserialize)]
pub struct Project {
    pub version: String,
    pub measurements: Vec<Measurement>,
    pub target: TargetCurve,
    pub peq_bands: Vec<PeqBand>,
    pub fir_config: FirConfig,
    pub multi_point: Option<MultiPointConfig>,
    pub notes: String,
}

// Extension: .phaseforge (JSON internally)
```

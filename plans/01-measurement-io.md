# 01 — Measurement I/O

## Supported Formats

### REW `.txt` (Primary)

REW exports frequency response as tab-separated text:

```
* Freq(Hz)  SPL(dB)  Phase(degrees)
20.000      65.3     -45.2
20.125      65.8     -44.9
...
20000.000   72.1     +120.3
```

**Parsing rules:**
- Lines starting with `*` → header/comment, skip
- Empty lines → skip
- Остальное: `split_whitespace()`, parse 3 floats
- Freq должна быть монотонно возрастающей (валидация)
- Phase может быть wrapped (±180°) или unwrapped — детектим автоматически

### `.frd` (Frequency Response Data)

Стандартный формат для акустических симуляторов:

```
20.00  65.3  -45.2
20.13  65.8  -44.9
```

Тот же формат, но без заголовков. Иногда 2 колонки (freq + mag, без фазы).

### `.mdat` (REW project)

ZIP-архив с XML внутри. Содержит множество измерений. Парсинг сложнее — отложим на M5 если нужно.

### ARTA `.pir` / `.ref`

Бинарный формат. Low priority, но структура документирована.

---

## Internal Data Model

```rust
/// Single frequency response measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Measurement {
    pub name: String,
    pub source_path: Option<PathBuf>,
    pub sample_rate: Option<f64>,       // if known from source
    pub freq: Vec<f64>,                 // Hz, sorted ascending
    pub magnitude: Vec<f64>,            // dB SPL
    pub phase: Option<Vec<f64>>,        // degrees, unwrapped
    pub metadata: MeasurementMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementMetadata {
    pub date: Option<String>,
    pub mic: Option<String>,
    pub notes: Option<String>,
    pub smoothing: Option<f64>,         // octave fraction applied
}
```

---

## Interpolation

Два режима интерполяции нужны для разных этапов pipeline:

### Log-spaced Grid (для отображения и PEQ)

Равномерная сетка в log-пространстве. Типично 1024 точки от 10 Hz до 24 kHz.

```rust
fn interpolate_log_grid(m: &Measurement, n_points: usize, f_min: f64, f_max: f64) 
    -> (Vec<f64>, Vec<f64>, Option<Vec<f64>>)
```

Метод: cubic spline в log(freq) пространстве.

### Linear Grid (для FFT/FIR)

Равномерная сетка от 0 Hz до Nyquist. Размер = `tap_count / 2 + 1` точек.

```rust
fn interpolate_linear_grid(m: &Measurement, n_bins: usize, sample_rate: f64) 
    -> (Vec<f64>, Vec<f64>, Option<Vec<f64>>)
```

**Edge cases:**
- DC (0 Hz): экстраполяция от первой точки, magnitude = first point, phase = 0
- Nyquist: экстраполяция или zero, в зависимости от target curve

---

## Phase Unwrapping

REW и большинство инструментов экспортируют wrapped phase (±180°). Для FIR коррекции нужна unwrapped.

```rust
fn unwrap_phase(wrapped: &[f64]) -> Vec<f64> {
    let mut unwrapped = vec![wrapped[0]];
    for i in 1..wrapped.len() {
        let mut diff = wrapped[i] - wrapped[i - 1];
        // Normalize to ±180
        while diff > 180.0 { diff -= 360.0; }
        while diff < -180.0 { diff += 360.0; }
        unwrapped.push(unwrapped[i - 1] + diff);
    }
    unwrapped
}
```

---

## Smoothing (Psychoacoustic)

Variable fractional-octave smoothing — ключ к правильной коррекции:

| Band | Fraction | Rationale |
|------|----------|-----------|
| < 100 Hz | 1/48 oct | Комнатные моды — узкие, нужно ловить точно |
| 100–500 Hz | 1/12 oct | Переходная зона |
| > 500 Hz | 1/3 oct | Ухо не слышит узких пиков на ВЧ, избыточная коррекция вредна |

**Алгоритм:** Для каждой точки `f[i]` считаем среднее по окну `[f/k, f*k]` где `k = 2^(fraction/2)`.

Smoothing применяется только для PEQ auto-alignment, НЕ для FIR (FIR работает с raw данными).

---

## Tauri IPC Commands

```rust
#[tauri::command]
async fn import_measurement(path: String) -> Result<Measurement, String> { ... }

#[tauri::command]
async fn get_smoothed(measurement: Measurement, smoothing: SmoothingConfig) 
    -> Result<Measurement, String> { ... }
```

---

## Validation Checklist

- [ ] REW txt с 3 колонками (freq, mag, phase)
- [ ] REW txt с 2 колонками (freq, mag — no phase)
- [ ] .frd с 3 колонками
- [ ] .frd с 2 колонками
- [ ] Файл с wrapped phase → корректный unwrap
- [ ] Файл с пустыми строками / комментариями
- [ ] Файл с нестандартным разделителем (запятые vs табы)
- [ ] Частоты не по порядку → ошибка
- [ ] Пустой файл → ошибка
- [ ] Огромный файл (500k точек) → < 100ms парсинг

---

## Test Data

Собрать коллекцию из 10+ реальных файлов:
- REW measurement room + nearfield
- REW с разными настройками smoothing
- .frd из VituixCAD
- .frd из XSim
- Edge cases: один бин, Nyquist = 48kHz vs 96kHz

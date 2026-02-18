# 07 — Export Formats & Pipeline

## FIR WAV Export

### Formats

| Format | Bits | Use Case |
|--------|------|----------|
| IEEE Float 64-bit | 64 | Maximum precision, chaining multiple filters |
| IEEE Float 32-bit | 32 | Universal compatibility (Roon, JRiver, Brutefir) |
| PCM 24-bit | 24 | Legacy convolvers |
| PCM 16-bit | 16 | Не рекомендуется, но поддержка для совместимости |

### WAV Header (64-bit float)

`hound` не поддерживает f64 нативно. Реализуем вручную:

```rust
fn write_wav_f64(path: &Path, samples: &[f64], sample_rate: u32) -> Result<()> {
    let mut file = File::create(path)?;
    let data_size = (samples.len() * 8) as u32;
    let file_size = 36 + data_size;
    
    // RIFF header
    file.write_all(b"RIFF")?;
    file.write_all(&file_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;
    
    // fmt chunk
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?;        // chunk size
    file.write_all(&3u16.to_le_bytes())?;          // format: IEEE float
    file.write_all(&1u16.to_le_bytes())?;          // channels
    file.write_all(&sample_rate.to_le_bytes())?;   // sample rate
    let byte_rate = sample_rate * 8;                // bytes/sec
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&8u16.to_le_bytes())?;          // block align
    file.write_all(&64u16.to_le_bytes())?;         // bits per sample
    
    // data chunk
    file.write_all(b"data")?;
    file.write_all(&data_size.to_le_bytes())?;
    for &s in samples {
        file.write_all(&s.to_le_bytes())?;
    }
    
    Ok(())
}
```

### Sample Rate Matching

FIR impulse ОБЯЗАН иметь тот же sample rate, что и воспроизводимый аудио поток.

Поддержка: 44100, 48000, 88200, 96000, 176400, 192000 Hz.

Опция: генерация нескольких WAV файлов для разных sample rates за один клик.

---

## PEQ Export Formats

### Equalizer APO (`.txt`)

```
Preamp: -3.2 dB
Filter 1: ON PK Fc 63.0 Hz Gain -4.2 dB Q 2.10
Filter 2: ON PK Fc 125.0 Hz Gain -6.8 dB Q 3.00
Filter 3: ON PK Fc 250.0 Hz Gain +2.1 dB Q 4.00
Filter 4: ON LS Fc 80.0 Hz Gain +3.0 dB Q 0.71
Filter 5: ON HS Fc 8000.0 Hz Gain -1.5 dB Q 0.71
```

Preamp = negative of max positive gain (headroom protection).

### miniDSP (`.csv` / `.xml`)

```csv
# miniDSP parametric EQ
# Channel: 1
PEQ1,ON,63.0,-4.2,2.10
PEQ2,ON,125.0,-6.8,3.00
PEQ3,ON,250.0,2.1,4.00
```

### Roon DSP (`.json`)

```json
{
  "type": "parametric_eq",
  "global_gain": -3.2,
  "bands": [
    {"type": "Peak", "frequency": 63.0, "gain": -4.2, "q": 2.1},
    {"type": "Peak", "frequency": 125.0, "gain": -6.8, "q": 3.0}
  ]
}
```

### CamillaDSP (`.yml`)

```yaml
filters:
  peq1:
    type: Biquad
    parameters:
      type: Peaking
      freq: 63.0
      gain: -4.2
      q: 2.1
  peq2:
    type: Biquad
    parameters:
      type: Peaking
      freq: 125.0
      gain: -6.8
      q: 3.0
```

### Generic JSON

Наш внутренний формат — superset. Другие экспортируются из него.

---

## Project File (`.phaseforge`)

### Format

JSON (gzipped optional). Extension: `.phaseforge`.

```json
{
  "version": "1.0.0",
  "created": "2025-01-15T10:30:00Z",
  "modified": "2025-01-15T14:22:00Z",
  
  "measurements": [
    {
      "name": "Left Speaker - Position 1",
      "source_path": "/Users/user/rew/left_pos1.txt",
      "sample_rate": 48000,
      "freq": [20.0, 20.5, ...],
      "magnitude": [62.3, 62.8, ...],
      "phase": [-45.2, -44.9, ...]
    }
  ],
  
  "target": {
    "reference_level_db": 0.0,
    "tilt_db_per_octave": -0.5,
    "high_pass": { "type": "LR4", "freq_hz": 20.0 },
    "low_pass": null,
    "low_shelf": { "freq_hz": 80.0, "gain_db": 3.0, "q": 0.707 }
  },
  
  "peq_bands": [
    { "freq_hz": 63.0, "gain_db": -4.2, "q": 2.1, "type": "PK", "enabled": true },
    { "freq_hz": 125.0, "gain_db": -6.8, "q": 3.0, "type": "PK", "enabled": true }
  ],
  
  "fir_config": {
    "tap_count": 65536,
    "sample_rate": 48000,
    "phase_strategy": "mixed",
    "mixed_crossover_hz": 300,
    "window": "blackman",
    "max_boost_db": 18.0
  },
  
  "notes": "Living room system, measured 2025-01-15"
}
```

### Backward Compatibility

`version` field. При загрузке старого формата — migration functions.

---

## Export Pipeline (UI Flow)

```
User clicks [Export ▼]
    │
    ├── "FIR WAV..."
    │   → File dialog (save)
    │   → Options: sample rate, bit depth, tap count
    │   → Progress bar → Done
    │
    ├── "PEQ Config..."
    │   → Format picker: APO / miniDSP / Roon / CamillaDSP / JSON
    │   → File dialog (save)
    │   → Done
    │
    ├── "Batch FIR..."
    │   → Folder picker (input measurements)
    │   → Folder picker (output WAVs)
    │   → Config confirmation
    │   → Progress bar (per file) → Summary
    │
    └── "Save Project"
        → File dialog (save .phaseforge)
        → Done
```

---

## Headroom / Preamp Gain

Любой EQ с boost'ами требует preamp attenuation чтобы избежать clipping.

```rust
fn compute_preamp(bands: &[PeqBand], fir_correction_db: &[f64]) -> f64 {
    // Worst-case: all boosts stack at same frequency
    // Practical: compute max of combined response
    let max_boost = combined_response_max(bands, fir_correction_db);
    if max_boost > 0.0 {
        -max_boost - 0.5  // 0.5 dB safety margin
    } else {
        0.0
    }
}
```

Preamp включается во все экспорт-форматы.

---

## Validation Before Export

- [ ] FIR tap count ≥ minimum для заданной нижней частоты
- [ ] Sample rate matches user's audio chain
- [ ] Preamp computed and non-positive
- [ ] No NaN/Inf в impulse data
- [ ] WAV file size < 100MB warning
- [ ] Phase strategy warning для linear phase + fast transient material

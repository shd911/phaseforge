# Промт для Code: b140.7.12 — REPhase reference comparison tests

**Тип:** test infrastructure. Bump до b140.7.12. Без правок DSP до
получения результатов тестов.

## Step 0

```
cd /Users/olegryzhikov/phaseforge
git status
ls -la test-fixtures/rephase/
```

Должны быть 4 файла: `44100.wav`, `48000.wav`, `88200.wav`, `176400.wav`.

## Контекст

User предоставил reference WAVs из REPhase для сравнения.

**REPhase config**:
- Mode: Minimum-Phase Filters
- High-pass Linkwitz-Riley **48 dB/oct** (= standard LR8)
- Frequency: 2000 Hz
- taps: 65536
- centering: middle
- windowing: hann
- format: 64 bits IEEE mono

**PhaseForge equivalent** для теста:
- HP filter type=LinkwitzRiley, **order=4** (UI label "LR4" = 48 dB/oct in PhaseForge convention)
- freq_hz=2000
- linear_phase=false (Min-Phase)
- taps=65536
- sr matching REPhase file

Оба должны давать аналогично шаблон HP=2000 48 dB/oct min-phase.

## Что нужно сделать

### 1. Bump до b140.7.12

- `tauri.conf.json` → b140.7.12.
- `lib.rs` → b140.7.12.
- `version.ts` → b140.7.12.

### 2. Добавить test-fixtures/ в .gitignore (если нет)

```
echo "test-fixtures/" >> .gitignore  # если не там
```

### 3. Helper для загрузки REPhase WAV

В test module `src-tauri/src/fir/iir_path.rs` или новый
`src-tauri/tests/rephase_compare.rs`:

```rust
use std::path::Path;
use std::fs::File;
use std::io::Read;

/// Load Float64 WAV file (RIFF parser, supports format_code=3 + bps=64).
fn load_wav_f64(path: &Path) -> Result<(u32, Vec<f64>), String> {
    let mut file = File::open(path).map_err(|e| format!("open: {}", e))?;
    let mut data = Vec::new();
    file.read_to_end(&mut data).map_err(|e| format!("read: {}", e))?;
    
    // Parse RIFF header — skip to fmt chunk for sr, then data chunk for samples.
    // Standard 44-byte header for canonical WAV; for safety scan chunks.
    // ... implementation ...
    
    todo!("parse RIFF, return (sample_rate, samples_f64)")
}
```

(Парсер уже есть в `lib.rs::export_fir_wav` readback diag — можно
переиспользовать или вынести в shared util.)

### 4. Тесты сравнения

```rust
#[test]
fn rephase_match_hp_lr8_2000_sr_44100() {
    rephase_compare(44100);
}
#[test]
fn rephase_match_hp_lr8_2000_sr_48000() {
    rephase_compare(48000);
}
#[test]
fn rephase_match_hp_lr8_2000_sr_88200() {
    rephase_compare(88200);
}
#[test]
fn rephase_match_hp_lr8_2000_sr_176400() {
    rephase_compare(176400);
}

fn rephase_compare(sr: u32) {
    let path = format!("test-fixtures/rephase/{}.wav", sr);
    let (rephase_sr, rephase_samples) = load_wav_f64(Path::new(&path))
        .expect(&format!("Failed to load {}", path));
    assert_eq!(rephase_sr, sr, "Sample rate mismatch");
    
    // Generate PhaseForge IIR path output for same config
    let cfg = IirPathInput {
        hp: Some(&FilterConfig {
            filter_type: FilterType::LinkwitzRiley,
            order: 4,  // → 48 dB/oct в PhaseForge convention
            freq_hz: 2000.0,
            linear_phase: false,
            ..Default::default()
        }),
        lp: None,
        peq: &[],
        sr: sr as f64,
        taps: 65536,
        window: WindowKind::Blackman,
        max_boost_db: 6.0,
        noise_floor_db: -150.0,
        ..Default::default()
    };
    let pf_out = generate_min_phase_fir_iir(&cfg).expect("PhaseForge gen");
    let pf_samples = &pf_out.impulse;
    
    // Both should be 65536 samples — assert
    assert_eq!(rephase_samples.len(), 65536, "REPhase samples count");
    assert_eq!(pf_samples.len(), 65536, "PhaseForge samples count");
    
    // FFT both
    let n = 65536;
    let rephase_spec = real_fft(&rephase_samples);
    let pf_spec = real_fft(pf_samples);
    
    // Compare at key freqs in passband (where filter is well-defined)
    let key_freqs = [500.0, 1000.0, 2000.0, 3000.0, 5000.0, 10000.0];
    let mut max_mag_diff = 0.0_f64;
    let mut max_phase_diff = 0.0_f64;
    for f in key_freqs {
        if f >= sr as f64 / 2.0 - 100.0 { continue; }  // skip near Nyquist
        let bin = (f * n as f64 / sr as f64).round() as usize;
        let r_mag = 20.0 * rephase_spec[bin].norm().log10();
        let p_mag = 20.0 * pf_spec[bin].norm().log10();
        let r_phase = rephase_spec[bin].arg().to_degrees();
        let p_phase = pf_spec[bin].arg().to_degrees();
        
        let mag_diff = (r_mag - p_mag).abs();
        // Phase diff with wrap handling
        let mut phase_diff = (r_phase - p_phase).abs();
        if phase_diff > 180.0 { phase_diff = 360.0 - phase_diff; }
        
        max_mag_diff = max_mag_diff.max(mag_diff);
        max_phase_diff = max_phase_diff.max(phase_diff);
        
        eprintln!(
            "sr={} f={:>6} Hz: REPhase mag={:+8.2} dB phase={:+7.1}° | \
             PF mag={:+8.2} dB phase={:+7.1}° | Δmag={:.2} Δphase={:.1}°",
            sr, f, r_mag, r_phase, p_mag, p_phase, mag_diff, phase_diff
        );
    }
    
    // Tolerance — set reasonable based on first run
    assert!(max_mag_diff < 1.0,
        "sr={}: max mag diff {:.2} dB > 1 dB", sr, max_mag_diff);
    assert!(max_phase_diff < 10.0,
        "sr={}: max phase diff {:.1}° > 10°", sr, max_phase_diff);
}
```

### 5. Run tests

```
cd src-tauri && cargo test rephase_match 2>&1 | tail -50
```

**Если все 4 PASS** — PhaseForge матчит REPhase в пределах допусков.
Bug закрыт, commit.

**Если что-то FAIL** — print eprintln вывод покажет:
- На каких freq расхождение
- Mag vs phase diff
- Какой sr хуже

Без правок DSP делать на этом этапе. Прислать результаты, user решает
fix direction по конкретным числам.

### 6. Build + sanity (без UI verify пока тесты не PASS)

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

### 7. Commit (только если все 4 rephase_match PASS)

```
git add -A
git commit -m "$(cat <<'EOF'
test: REPhase reference comparison for HP min-phase IIR (b140.7.12)

Adds 4 tests comparing PhaseForge IIR path output against REPhase
reference WAVs at sr={44.1, 48, 88.2, 176.4k}. Reference WAVs in
test-fixtures/rephase/ (gitignored) generated with REPhase Min-Phase
HP=2000 Linkwitz-Riley 48 dB/oct, taps=65536, centered, hann.

Tests assert magnitude and phase agreement at key passband freqs.
Establishes objective acceptance for HP IIR path correctness.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

## Что НЕ делать

- Не патчить DSP до получения результатов rephase_match.
- Не loosen tolerance без user одобрения.
- Не commit-ить если есть FAIL.

## Acceptance

- 4 теста rephase_match PASS на всех sr.
- max_mag_diff < 1 dB.
- max_phase_diff < 10° в passband.

## Что прислать

Если PASS — короткий report: "4/4 PASS, max mag/phase diffs per sr".

Если FAIL — eprintln вывод по каждому freq для каждого sr. Без
гипотез, только числа.

## Правила

- Без нарратива.
- Read-only диагностика DSP — тесты, не fixes.

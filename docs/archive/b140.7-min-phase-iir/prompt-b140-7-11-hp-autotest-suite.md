# Промт для Code: b140.7.11 — automated test suite для HP IIR path

**Тип:** test infrastructure. Bump до b140.7.11. Цель — поймать
phase/impulse/WAV bugs **до** UI verify, чтобы не гонять циклы.

## Step 0

```
cd /Users/olegryzhikov/phaseforge
git status
```

## Контекст

10+ промтов на phase/impulse fix для HP=2000 sr=48k без объективного
acceptance. UI verify subjective, тесты Code-а не покрывают тот case.

Нужна автоматическая testovая батарея которая:
1. Берёт production config user-а (HP=2000 LR4 sr=48k taps=65536).
2. Прогоняет через `generate_min_phase_fir_iir` (IIR path).
3. FFT-ит **WAV impulse** (то что реально попадает в файл).
4. Проверяет конкретные freq points против analytical reference.
5. Проверяет peak position в WAV.
6. Проверяет phase Nyquist behavior.
7. PASS/FAIL без UI наблюдения.

## Что нужно сделать

### 1. Bump до b140.7.11

- `tauri.conf.json` → b140.7.11.
- `lib.rs` → b140.7.11.
- `version.ts` → b140.7.11.

### 2. Новый тест-файл `src-tauri/src/fir/iir_path.rs` под `#[cfg(test)]`

Группа тестов имитирующих user's REW workflow:

```rust
#[test]
fn hp_lr4_2000_sr_48k_wav_matches_analytical() {
    // User's exact config:
    let cfg = IirPathInput {
        hp: Some(&FilterConfig {
            type_: FilterType::LinkwitzRiley,
            order: 4,
            freq_hz: 2000.0,
            linear_phase: false,
        }),
        lp: None,
        peq: &[],
        sr: 48000.0,
        taps: 65536,
        window: WindowKind::Blackman,
        max_boost_db: 6.0,
        noise_floor_db: -150.0,
    };
    let out = generate_min_phase_fir_iir(&cfg).unwrap();

    // FFT того что в WAV — это что REW реально читает
    let wav_impulse = &out.impulse;
    let fft_mag = compute_fft_magnitude_db(wav_impulse, cfg.sr);
    let fft_phase = compute_fft_phase_deg(wav_impulse, cfg.sr);

    // Analytical reference: LR8 HP @ 2000 (PhaseForge convention)
    // Expected slope: 48 dB/oct, -6 dB at corner

    // Magnitude checks at key freqs (tolerance ±0.5 dB)
    let mag_tests = [
        (100.0,  -208.0, 5.0),   // deep stopband, looser tolerance
        (200.0,  -160.0, 5.0),
        (500.0,  -97.0,  3.0),
        (1000.0, -48.0,  2.0),
        (2000.0, -6.0,   0.5),   // corner, strict
        (5000.0, 0.0,    0.5),   // passband
        (10000.0, 0.0,   0.5),
    ];
    for (freq, expected_db, tol) in mag_tests {
        let bin = (freq * cfg.taps as f64 / cfg.sr) as usize;
        let actual = fft_mag[bin];
        assert!(
            (actual - expected_db).abs() < tol,
            "Mag at {} Hz: expected {} dB, got {:.2} dB (tol {:.1})",
            freq, expected_db, actual, tol
        );
    }

    // Phase at Nyquist must be ≈0° (or ±180° wrap, but consistent)
    let nyq_bin = cfg.taps / 2;
    let nyq_phase = fft_phase[nyq_bin];
    let nyq_phase_normalized = ((nyq_phase + 180.0) % 360.0 - 180.0).abs();
    assert!(
        nyq_phase_normalized < 5.0 || (nyq_phase_normalized - 180.0).abs() < 5.0,
        "Phase at Nyquist {} Hz: {:.1}° (expected ≈0° or ±180°)",
        cfg.sr / 2.0, nyq_phase
    );

    // Phase at passband freq (5 kHz) — должна быть близка к analytical
    // For LR8 HP at f >> fc: phase ≈ 0
    let pb_bin = (5000.0 * cfg.taps as f64 / cfg.sr) as usize;
    let pb_phase = fft_phase[pb_bin];
    // Tolerance: linear-phase term from N/2 padding adds ramp;
    // raw FFT(raw_impulse) gives analytical, but WAV is centered.
    // For analytical model match check use raw FFT, не wav FFT.
    
    // Peak position в wav_impulse
    let peak_idx = wav_impulse.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
        .map(|(i, _)| i).unwrap();
    let half = cfg.taps / 2;
    assert!(
        (peak_idx as i64 - half as i64).abs() < 10,
        "WAV peak idx {} not near N/2={} (REW expects centered)", 
        peak_idx, half
    );
}

#[test]
fn hp_lr4_2000_sr_44k1_wav_matches_analytical() {
    // То же что выше, sr=44100. User тестирует обе.
    // Same checks scaled.
}

#[test]
fn hp_lr4_2000_sr_88k2_wav_baseline() {
    // sr=88200 — baseline где user сказал "OK".
    // Same checks. Если этот fail — проблема не sr-specific.
}

#[test]
fn hp_lr4_2000_sr_176k4_wav_baseline() {
    // sr=176400 — baseline где user сказал "OK".
    // Same checks.
}
```

### 3. Тест UI plot output (raw FFT path)

```rust
#[test]
fn hp_lr4_2000_sr_48k_ui_plot_phase_matches_model() {
    // Берёт IirPathOutput.realized_phase (что UI plot отображает).
    // Сравнивает с analytical filter phase напрямую (через target::evaluate).
    // Tolerance: ≤ 5° peak в passband (100..20k Hz, без deep stopband).

    let cfg = /* same as above */;
    let out = generate_min_phase_fir_iir(&cfg).unwrap();
    
    // out.realized_phase — что в UI plot
    // Compute analytical reference:
    let analytical_phase = compute_analytical_filter_phase(
        &cfg.hp.unwrap(), &out.log_freq, cfg.sr
    );
    
    for (i, freq) in out.log_freq.iter().enumerate() {
        if *freq < 100.0 || *freq > 20000.0 { continue; }
        let diff = (out.realized_phase[i] - analytical_phase[i]).to_degrees().abs();
        assert!(diff < 5.0,
            "UI plot phase at {} Hz: realized {:.1}°, analytical {:.1}°, diff {:.2}°",
            freq, out.realized_phase[i].to_degrees(), analytical_phase[i].to_degrees(), diff
        );
    }
}
```

### 4. Helper functions

Если ещё нет — добавить в test module:
```rust
fn compute_fft_magnitude_db(samples: &[f64], sr: f64) -> Vec<f64> { ... }
fn compute_fft_phase_deg(samples: &[f64], sr: f64) -> Vec<f64> { ... }
fn compute_analytical_filter_phase(filter: &FilterConfig, freqs: &[f64], sr: f64) -> Vec<f64> { ... }
```

### 5. Run tests

```
cd src-tauri && cargo test --lib hp_lr4 2>&1 | tail -30
```

**Если что-то FAIL** — STOP, report details:
- Какой sr.
- Какая freq.
- Expected vs actual (mag dB or phase deg).
- НЕ патчить пока user не одобрит fix direction.

**Если все PASS** — fix b140.7.10 действительно работает на HP cases.
Тогда commit.

### 6. Build + UI sanity check

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

Title bar = b140.7.11. Если cargo тесты PASS на HP cases — user
проверит REW визуально для confirmation.

### 7. Commit (только если все HP tests PASS)

```
git add -A
git commit -m "$(cat <<'EOF'
test: HP IIR path acceptance suite (b140.7.11)

Adds automated tests covering user's exact production scenarios:
HP=2000 LR4 across sr={44.1k, 48k, 88.2k, 176.4k}, 65536 taps,
Blackman window. Tests check WAV impulse FFT magnitude vs
analytical reference at key frequencies, peak position centered
near N/2, phase Nyquist behavior, and UI plot realized_phase
match with analytical filter phase.

Catches regressions before UI verify cycles.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

## Что НЕ делать

- Не патчить DSP, пока не запущены тесты и не известны результаты.
- Не commit-ить до PASS всех HP tests.
- Не убирать LP tests которые уже есть (они dependent regression check).

## Acceptance

- 4 новых HP tests на разных sr (44.1, 48, 88.2, 176.4) PASS.
- 1 UI plot phase test PASS.
- Existing 185 cargo tests PASS.
- 104 vitest PASS.

## Если HP test fail на 48k

Тогда **диагностика по конкретным числам теста** — не теоретизировать:
- Какой freq fail-нул?
- Phase или magnitude?
- Discrepancy в degrees / dB?
- Сравнить с 176k где OK — что отличается в выходе?

После этого делать целенаправленный фикс не от UI впечатления, а от
конкретных автоматических чисел.

## Правила

- Без нарратива.
- Один report после run tests.

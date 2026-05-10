# Промт для Code: b140.7 IIR-based min-phase FIR pipeline

**Тип:** architectural rebuild. Новый параллельный путь, FFT path
не трогаем. Bump до b140.7, коммит после all PASS + UI verify.

## Step 0

```
cd /Users/olegryzhikov/phaseforge
git status
```

Должно быть clean (после rollback variant B). Если modified — STOP.

## Цель

Min-Phase FIR через IIR cascade вместо FFT/IFFT cepstral. Аналитический
filter design (LR/BU/Bessel) → biquad cascade → bilinear transform
→ apply impulse → truncate to N taps. Peak-at-0 by construction
(analog filter с poles в LHP = min-phase физически).

**Scope b140.7**: только Min-Phase user choice (linear_phase_main=false)
с не-Gaussian filters. Linear-phase / Composite / Gaussian / Custom
остаются на FFT pipeline (там нет той проблемы).

## Implementation phases (внутри одного промта)

### Phase 0 — bump

- `src-tauri/tauri.conf.json` → title `"PhaseForge — b140.7"`.
- `src-tauri/src/lib.rs` → b140.7 startup log.
- `src/lib/version.ts` (если есть) → b140.7.

### Phase 1 — создать модуль `src-tauri/src/fir/iir_path.rs`

Структуры:

```rust
/// Digital biquad section (Direct Form I).
#[derive(Clone, Debug)]
pub struct DigitalBiquad {
    pub b0: f64, pub b1: f64, pub b2: f64,
    pub a1: f64, pub a2: f64, // a0 normalized to 1
}

impl DigitalBiquad {
    /// Apply biquad to one sample, updating state.
    pub fn process(&self, x: f64, state: &mut [f64; 4]) -> f64 {
        // state = [x_n-1, x_n-2, y_n-1, y_n-2]
        let y = self.b0 * x + self.b1 * state[0] + self.b2 * state[1]
                          - self.a1 * state[2] - self.a2 * state[3];
        state[1] = state[0]; state[0] = x;
        state[3] = state[2]; state[2] = y;
        y
    }
}
```

Functions для построения analog biquad коэффициентов (s-domain) и
bilinear transform к digital:

```rust
/// Analog biquad: H(s) = (b0*s² + b1*s + b2) / (s² + a1*s + a2)
struct AnalogBiquad { b0: f64, b1: f64, b2: f64, a1: f64, a2: f64 }

/// Pre-warp Fc для bilinear: Fc_warped = (sr/π) * tan(π * fc / sr)
fn prewarp(fc: f64, sr: f64) -> f64 {
    (sr / std::f64::consts::PI) * (std::f64::consts::PI * fc / sr).tan()
}

/// Bilinear transform analog biquad → digital biquad.
/// s = (2/T) * (z-1)/(z+1), T = 1/sr.
fn bilinear(analog: &AnalogBiquad, fc: f64, sr: f64) -> DigitalBiquad {
    let fc_w = prewarp(fc, sr);
    let omega = 2.0 * std::f64::consts::PI * fc_w;
    // Substitute s = 2*sr*(1-z⁻¹)/(1+z⁻¹), expand, normalize a0.
    // ... стандартная формула, см. e.g. RBJ Audio EQ Cookbook ...
    todo!("implement standard bilinear")
}
```

Сборка cascade per filter type:

```rust
/// LR-N HP/LP as cascade of 2 BU(N/2) sections (LR = BU squared).
/// E.g. LR4 = 2x BU2 = 2 biquads cascade.
pub fn lr_section_analog(order: usize, fc: f64, is_hp: bool) -> Vec<AnalogBiquad>;

pub fn butterworth_section_analog(order: usize, fc: f64, is_hp: bool) -> Vec<AnalogBiquad>;

pub fn bessel_section_analog(order: usize, fc: f64, is_hp: bool) -> Vec<AnalogBiquad>;
```

PEQ biquads — переиспользовать существующие из проекта (grep
`compute_peq` или `PeqBand`).

Cascade evaluation:

```rust
/// Apply impulse δ(n) to cascade of digital biquads, return N taps of impulse.
pub fn cascade_impulse(biquads: &[DigitalBiquad], n_taps: usize) -> Vec<f64> {
    let mut impulse = vec![0.0; n_taps];
    let mut states: Vec<[f64; 4]> = vec![[0.0; 4]; biquads.len()];
    for n in 0..n_taps {
        let mut x = if n == 0 { 1.0 } else { 0.0 };
        for (bq, state) in biquads.iter().zip(states.iter_mut()) {
            x = bq.process(x, state);
        }
        impulse[n] = x;
    }
    impulse
}
```

Main entry:

```rust
pub struct IirPathInput<'a> {
    pub hp: Option<&'a FilterConfig>,
    pub lp: Option<&'a FilterConfig>,
    pub peq: &'a [PeqBand],
    pub sr: f64,
    pub taps: usize,
    pub window: WindowKind,
    pub max_boost_db: f64,
    pub noise_floor_db: f64,
}

pub struct IirPathOutput {
    pub impulse: Vec<f64>,
    pub time_ms: Vec<f64>,
    pub realized_mag: Vec<f64>,    // dB on log grid (caller resamples)
    pub realized_phase: Vec<f64>,  // wrapped radians
    pub norm_db: f64,
    pub causality: f64,
    pub log_freq: Vec<f64>,        // 5..40k log grid 512 pts
}

pub fn generate_min_phase_fir_iir(input: &IirPathInput) -> Result<IirPathOutput, String>;
```

Внутри `generate_min_phase_fir_iir`:
1. Build cascade: HP analog biquads → bilinear → digital. Same для LP. Append PEQ digital biquads (analytical, sample-rate independent).
2. Apply impulse → impulse[N] (raw, before window).
3. Optional window (Blackman half-window для energy preservation; impulse causal → правая часть от peak ≈ 0).
4. Normalize: passband peak → 0 dB. Найти peak в impulse FFT в passband range, scale impulse.
5. Compute FFT(impulse) → realized_mag/phase on linear FFT grid → resample на 5..40k log grid 512 pts через ту же `resampleOntoGrid` логику что в FFT path.
6. Compute causality: `sum(impulse[0..peak].abs²) / sum(impulse.abs²)` для consistency с FFT path metric (peak at 0 → causality≈100%).

### Phase 2 — routing в `generate_model_fir`

В `src-tauri/src/fir/mod.rs::generate_model_fir`:

```rust
// Determine if IIR path is applicable
let iir_applicable = !config.linear_phase_main
    && config.phase_mode == PhaseMode::Composite
    && hp_is_iir_realizable(&hp_config)  // helper: not Gaussian
    && lp_is_iir_realizable(&lp_config)
    && config.subsonic_cutoff_hz.is_none(); // IIR path b140.7 не покрывает subsonic

if iir_applicable {
    let iir_input = IirPathInput { /* ... */ };
    let iir_out = crate::fir::iir_path::generate_min_phase_fir_iir(&iir_input)?;
    // Convert IirPathOutput → ModelFirResult (existing return type)
    return Ok(/* ... */);
}

// Existing FFT path остаётся для:
// - linear_phase_main=true (Linear-Phase / Composite linear)
// - Gaussian filters
// - Subsonic protect (until Phase 3)
// - Custom measured targets
```

`hp_is_iir_realizable` / `lp_is_iir_realizable` — return false для
Gaussian, true для LR / Butterworth / Bessel / Custom-rational.

### Phase 3 — тесты

Cargo tests в `src-tauri/src/fir/iir_path.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iir_lr4_lp_200_peak_at_zero_sr_48k() {
        // Input: LP=LR4@200, sr=48k, taps=65536
        let out = generate_min_phase_fir_iir(/* ... */).unwrap();
        let peak_idx = out.impulse.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(i, _)| i).unwrap();
        assert!(peak_idx <= 5, "LR4 LP=200: peak idx={} > 5", peak_idx);
        // Causality should be > 99%
        assert!(out.causality > 0.99, "causality={:.3}", out.causality);
    }

    #[test]
    fn iir_lr4_hp_2000_peak_at_zero_sr_48k() { /* ... */ }

    #[test]
    fn iir_bp_200_2000_peak_at_zero_sr_48k() { /* ... */ }

    #[test]
    fn iir_lr4_lp_200_realized_mag_matches_target_in_passband() {
        // Build analytical LR4 reference via target::evaluate, compare
        // realized_mag from IIR path against reference in passband.
        // Tolerance: ≤ 0.5 dB RMS, ≤ 1.0 dB peak.
    }

    #[test]
    fn iir_bilinear_unit_test_lr2_dc_gain() {
        // Trivial: LR2 LP at fc=1000, sr=48k. DC gain = 1 (0 dB).
        // Apply impulse, sum coefficients = DC gain.
    }
}
```

Регрессия: запустить **все** existing cargo тесты. Должно быть **179 PASS**
(IIR path активируется только при routing условиях, FFT path
unchanged).

### Phase 4 — UI verify

После сборки запустить dev:

```
cd /Users/olegryzhikov/phaseforge && nohup cargo tauri dev > /tmp/phaseforge-dev.log 2>&1 &
```

Открыть test проект (flat), Min-Phase mode (Lin-φ off):
- sr=48k, Band 1 LP=200 LR4 → Export. **Pre-ring=0.00 ms, Causal≥99%**.
- sr=48k, Band 2 BP 200-2000 LR4 → то же.
- sr=48k, Band 3 HP=2000 LR4 → то же.
- sr=176.4k, повторить.

В логе должна быть строка `[IIR PATH] LR4 LP=200 ... peak_idx=N` (добавить
log на entry в iir_path module для observability).

Если **что-то fail** (Pre-ring > 0.5 ms, Causal < 95%, mag err > 0.5 dB)
— STOP, прислать логи + screenshot. Не commit.

### Phase 5 — Commit (только если all PASS)

```
git add -A
git commit -m "$(cat <<'EOF'
feat: IIR-based min-phase FIR pipeline (b140.7)

Replace FFT/cepstral min-phase reconstruction with analytical IIR
cascade for non-Gaussian Min-Phase user choice. Pipeline:
analog filter design → bilinear → digital biquads → cascade impulse
→ truncate to N taps. Peak-at-0 by construction (analog poles in
LHP = min-phase physically).

Scope: LR/BU/Bessel HP+LP and PEQ biquads, Min-Phase user choice
(linear_phase_main=false), no subsonic protect, no Composite mode.
FFT path retained for Linear-Phase, Composite with subsonic,
Gaussian, and Custom measured targets.

Cause: cepstral on sparse linear FFT grid (large noise_floor regions)
created high-quefrency artifact → constant group delay → impulse
peak shifted from idx=0 (Pre-ring 4.25 ms on Band 1 LP=200, REW
phase mismatch with model).

Cepstral floor and post-IFFT shift attempted earlier (b140.7 tries 1-2)
broke either functional invariants or magnitude fidelity. Variant
analytical-phase-passthrough (try 3) was DSP-incorrect (analog phase
≠ discrete-Hilbert). IIR cascade is the natural representation for
min-phase from analytical filter specs.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

### Phase 6 — End-of-prompt automation

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

## Что НЕ делать

- Не менять FFT path code (existing `composite_phase_inner`,
  `iterative_refine` для FFT путей).
- Не покрывать Gaussian / Subsonic в этом промте — Phase 2 после b140.7.
- Не покрывать Custom measured targets — для них FFT cepstral
  остаётся стандартом.
- Не делать UI toggle между IIR и FFT — routing автоматически по
  config.
- Не коммитить если any test fail или UI Pre-ring > 0.5 ms.

## Acceptance

- 179+ cargo PASS (existing) + новые IIR tests PASS.
- Pre-ring=0.00 ms, Causal≥99% на LP=200, BP 200-2000, HP=2000
  (Min-Phase mode, sr=48k и sr=176.4k).
- Realized_mag совпадает с target в passband ≤ 0.5 dB RMS.
- В REW: экспортированный WAV — phase совпадает с model (constant
  group delay отсутствует).
- 104+ vitest PASS (без изменений в TS pipeline для IIR path —
  Rust сам routes).

## STOP triggers

При fail на любом phase — STOP, report, без auto-rollback. User
ревьюит и решает: исправить inline, доделать, или rollback всего.

## Правила

- Без нарратива.
- Phases выполнять последовательно.
- Между phases — короткий status report (cargo PASS count, etc.).
- Если Bilinear formula нечёткая — найти reference (RBJ EQ Cookbook,
  Bristow-Johnson) и cite в комментарии.

# Промт для Code: b140.7 implementation — variant (B) analytical phase pipeline

**Тип:** implementation. Bump до b140.7, коммит после all PASS + UI verify.

## Step 0 — clean state

```
cd /Users/olegryzhikov/phaseforge
git status
git diff --stat
```

Должно быть clean. Если modified — STOP.

## Решения по плану (final)

- **Default `use_provided_phase = true`** в b140.7 (флаг сохранён для
  опт-аута / debug). Bug в production важнее backward compat.
- **Новое IPC поле `modelPhaseLinear: number[]`** длины n_bins —
  отдельно от legacy `modelPhase` (log grid).
- **DC point**: `modelPhaseLinear[0] = 0` для f=0.
- **Length validation в Rust**: если `use_provided_phase=true` и длина
  ≠ n_bins → error, не silent passthrough.

## Implementation order

### 1. Bump (early — чтобы видеть в startup)

- `src-tauri/tauri.conf.json` — title → `"PhaseForge — b140.7"`.
- `src-tauri/src/lib.rs` — startup log → `b140.7`.
- `src/lib/version.ts` (если есть) — соответственно.

### 2. Golden snapshots capture (b140.6 baseline)

Перед изменением DSP:

```
git stash # сохранить bump временно
```

(или работать на отдельном branch capture/golden-b140.6 если удобно)

Создать `src-tauri/tests/golden_b140_6.rs`:

```rust
//! Golden snapshots captured BEFORE variant (B) implementation.
//! Hashes of impulse[0..1024], realized_mag, realized_phase для
//! 8 фикстур из docs/variant-b-plan.md.

use blake3::Hasher;

fn hash_f64_slice(data: &[f64]) -> String {
    let mut h = Hasher::new();
    for v in data { h.update(&v.to_le_bytes()); }
    h.finalize().to_hex().to_string()
}

#[test]
fn snapshot_b140_6_F1_lp_only_48k() {
    // Build FIR for: sr=48000, LP=LR4@200, no HP, no PEQ, no subsonic
    let result = generate_fir_for_fixture(/* ... */);
    let h_imp = hash_f64_slice(&result.impulse[0..1024]);
    let h_mag = hash_f64_slice(&result.realized_mag);
    let h_ph = hash_f64_slice(&result.realized_phase);
    
    // Print for capture (test passes by default), assert in next phase
    println!("F1 b140.6: imp={} mag={} ph={}", h_imp, h_mag, h_ph);
}

// ... F2..F8 аналогично, см. план Fixtures table
```

Запустить:
```
cd src-tauri && cargo test golden_b140_6 -- --nocapture | tee /tmp/golden-b140-6.txt
```

Скопировать вывод hashes в **константы** в новом
`src-tauri/tests/golden_b140_6_hashes.rs`:

```rust
pub const F1_LP_48K_IMPULSE: &str = "<hash>";
pub const F1_LP_48K_MAG: &str = "<hash>";
// ... и т.д.
```

`git stash pop` (вернуть bump).

### 3. Rust изменения

#### 3.1 `src-tauri/src/fir/types.rs` (или где `FirConfig`)

Добавить поле:
```rust
#[serde(default = "default_use_provided_phase")]
pub use_provided_phase: bool,
```

```rust
fn default_use_provided_phase() -> bool { true }
```

Добавить параметр в сигнатуру `generate_model_fir` (или в command IPC):
```rust
pub model_phase_linear: Option<Vec<f64>>,
```

#### 3.2 `src-tauri/src/fir/mod.rs::generate_model_fir`

Заменить блок выбора `target_phase_rad` (около line 593-638):

```rust
let target_phase_rad = if config.use_provided_phase {
    let lin = model_phase_linear.as_ref()
        .ok_or_else(|| "use_provided_phase=true requires model_phase_linear")?;
    if lin.len() != n_bins {
        return Err(format!(
            "model_phase_linear length {} != n_bins {}", lin.len(), n_bins
        ));
    }
    lin.clone()
} else if effective_linear {
    vec![0.0; n_bins]
} else if config.phase_mode == PhaseMode::Composite {
    crate::fir::helpers::compose_target_phase(/* existing args */)
} else {
    // existing MinimumPhase / LinearPhase paths
};
```

`peq_phase_rad` (line 642-648): когда `use_provided_phase=true` →
`vec![0.0; n_bins]` (PEQ уже в model_phase_linear).

`phase_rad = target_phase_rad + peq_phase_rad` остаётся.

#### 3.3 `src-tauri/src/fir/helpers.rs::iterative_refine`

В iter loop (line 120-128, 190-217): добавить условие
`if !config.use_provided_phase { recompute_phase(...) }`. При
`use_provided_phase=true` — phase не пересчитывается, остаётся
переданной. Magnitude correction через damped error остаётся.

### 4. TS изменения

#### 4.1 `src/lib/band-evaluator.ts`

После line 404 (`firCombinedPhase`), добавить resample:

```ts
function unwrapPhase(p: number[]): number[] {
  const out = [...p];
  for (let i = 1; i < out.length; i++) {
    let d = out[i] - out[i - 1];
    while (d > Math.PI) { out[i] -= 2 * Math.PI; d = out[i] - out[i - 1]; }
    while (d < -Math.PI) { out[i] += 2 * Math.PI; d = out[i] - out[i - 1]; }
  }
  return out;
}

function wrapPhase(p: number[]): number[] {
  return p.map(v => {
    let w = v;
    while (w > Math.PI) w -= 2 * Math.PI;
    while (w < -Math.PI) w += 2 * Math.PI;
    return w;
  });
}

function resampleLogToLinear(
  logFreq: number[], logVal: number[], nBins: number, sr: number
): number[] {
  const out = new Array<number>(nBins).fill(0);
  out[0] = 0; // DC
  for (let k = 1; k < nBins; k++) {
    const f = (k * sr) / (2 * (nBins - 1)); // f_k = k * sr / n_fft, n_fft = 2*(n_bins-1)
    if (f <= logFreq[0]) { out[k] = logVal[0]; continue; }
    if (f >= logFreq[logFreq.length - 1]) {
      out[k] = logVal[logVal.length - 1]; continue;
    }
    let lo = 0, hi = logFreq.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (logFreq[mid] <= f) lo = mid; else hi = mid;
    }
    const lf = Math.log(logFreq[lo]), hf = Math.log(logFreq[hi]);
    const t = (Math.log(f) - lf) / (hf - lf);
    out[k] = logVal[lo] + t * (logVal[hi] - logVal[lo]);
  }
  return out;
}
```

Применить:
```ts
const nBins = Math.floor(cfg.taps / 2) + 1;
const phaseUnwrapped = unwrapPhase(firCombinedPhase);
const phaseLinearUnwrapped = resampleLogToLinear(
  firFreq, phaseUnwrapped, nBins, cfg.sampleRate
);
const modelPhaseLinear = wrapPhase(phaseLinearUnwrapped);
modelPhaseLinear[0] = 0; // explicit DC
```

#### 4.2 IPC payload

В вызове `invoke("generate_model_fir", ...)` (line 406-430):
- Добавить ключ `modelPhaseLinear`.
- Добавить в config `use_provided_phase: true`.

### 5. Тесты

#### 5.1 Cargo

Новый тест в `src-tauri/src/dsp/phase.rs` или `fir/mod.rs`:

```rust
#[test]
fn min_phase_impulse_peaks_at_zero_variant_b() {
    for (label, hp_hz, lp_hz, expected_peak_max) in [
        ("LP=200", None::<f64>, Some(200.0_f64), 5),
        ("BP 200-2000", Some(200.0), Some(2000.0), 5),
        ("HP=2000", Some(2000.0), None, 5),
    ] {
        // Build analytical phase on log grid + linear (mimicking TS resample)
        // Pass with use_provided_phase=true
        // Assert impulse peak idx <= expected_peak_max
    }
}
```

Регрессия: запустить все `golden_b140_6_*` тесты — при
`use_provided_phase=false` всё bit-exact.

При `use_provided_phase=true` (новый default) golden_b140_6 могут
отличаться — это **ожидаемо**. Для variant-b сравнения создать
`golden_b140_7_variant_b/` с новыми snapshots.

#### 5.2 Vitest

Новый тест `evaluateBandFull` resample correctness:
- Build trivial log grid + known phase.
- Resample → linear.
- Assert правильные значения на ключевых freq.

### 6. Run all tests

```
cd src-tauri && cargo test --lib
cd /Users/olegryzhikov/phaseforge && npm run test
```

Ожидание:
- Cargo: ≥ 179 + новые тесты PASS. golden_b140_6 при flag=false bit-exact.
- Vitest: 104 + новые PASS.

Если что-то fail вне ожидаемого — STOP, проанализировать.

### 7. UI verify

После сборки запустить:
```
cd /Users/olegryzhikov/phaseforge && nohup cargo tauri dev > /tmp/phaseforge-dev.log 2>&1 &
```

Открыть проект, sr=48k и sr=176.4k:
- Band 1 LP=200 → Export → Pre-ring=0.00, Causal≥99%.
- Band 2 BP 200-2000 → Export → то же.
- Band 3 HP=2000 → Export → то же.
- Курсор в passband: Model° совпадает с FIR° (≤ 0.5° RMS).

Прислать скрин Export для подтверждения.

### 8. Commit

```
git add -A
git commit -m "$(cat <<'EOF'
feat: variant (B) analytical phase pipeline (b140.7)

Replace cepstral min-phase reconstruction in Rust composite_phase_inner
with passthrough of TS-computed analytical phase. TS reconstructTargetPhase
already produces correct phase on log grid for SPL view; resample
log→linear FFT grid with unwrap/interp/wrap and pass via new
modelPhaseLinear IPC field.

Cause of REW phase mismatch in min-phase FIR exports: cepstral on
sparse linear FFT grid (large noise_floor regions) created
high-quefrency artifact → constant group delay → impulse peak shifted
from idx=0 (Pre-ring 4.25 ms on Band 1 LP=200).

Fix: bypass cepstral entirely. Analytical phase is exact for known
filter forms (LR4, BU, Bessel, Custom) and Hilbert-reconstructed
on smooth log grid for Gaussian min-phase + subsonic.

iterative_refine: phase fixed (analytical truth), magnitude-only
refinement. Eliminates Band 3 (HP=2000) divergence as derivative.

Opt-out via FirConfig.use_provided_phase=false (default true).

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

### 9. End-of-prompt

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

- Не менять `reconstructTargetPhase` (TS) — она работает.
- Не менять magnitude pipeline (b140.6 fix остаётся).
- Не убирать legacy paths (`use_provided_phase=false`) — они служат
  опт-аутом и регрессионной защитой.
- Не делать default=false — bug в production важнее compat.

## Acceptance

- Pre-ring=0.00 ms на всех 3 полосах × 2 sr.
- Causal ≥ 99%.
- Model° совпадает с FIR° на Export.
- iterative_refine на Band 3 не расходится.
- ≥ 179 cargo + 104 vitest PASS.
- Commit с co-author.

## Правила

- Без нарратива.
- Все шаги по порядку, не пропускать.
- Если шаг fail — STOP, report, без авто-rollback.

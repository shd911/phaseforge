# Min-Phase FIR end-to-end trace (b140.6)

Read-only investigation. Full path from user phase choice → IFFT → returned
impulse, for a Min-Phase user choice with subsonic OFF and ON.

## TL;DR

The TS frontend always sends `phase_mode: "Composite"` with a derived
`linear_phase_main` flag. Min-Phase user choice on every filter →
`linear_phase_main = false` → Rust path goes through the "min-phase"
branch (no center shift, half-window only). On paper the impulse should
be causal with peak at `samples[0]`. Two suspicious places where a
linear-phase / constant-delay artefact could leak in are flagged at the
end — but the read-only trace alone does not pin a single root cause.

## 1. TS side: what gets sent to Rust

`src/lib/band-evaluator.ts:350-424` — only call site for
`generate_model_fir`.

Phase mode is hardcoded:
- `src/lib/band-evaluator.ts:421` — `phase_mode: "Composite"`.

`linear_phase_main` is derived from per-filter `linear_phase`:
- `src/lib/band-evaluator.ts:365-368`
  ```ts
  const isUserLin = (f: FilterConfig | null | undefined) =>
    !f || f.linear_phase === true;
  const linearMain =
    isUserLin(band.target.high_pass) && isUserLin(band.target.low_pass);
  ```
  — note `!f` returns `true` when the filter is null. So a band with
  only an LP set evaluates `isUserLin(hp=null)=true` regardless of user
  intent.

Subsonic comes from a separate flag:
- `src/lib/band-evaluator.ts:370` —
  `const subsonicCutoff = hasActiveSubsonicProtect(hp) ? hp!.freq_hz / 8 : null;`

`firFreq` is built and extended (b140.5) before `generate_model_fir`:
- `src/lib/band-evaluator.ts:375-392` — log grid 5..fMaxFir then
  `appendNoiseFloorTail` to Nyquist.
- `src/lib/band-evaluator.ts:411-414` — `freq`, `targetMag`, `peqMag`,
  `modelPhase` shipped to Rust.

The TS-computed `firCombinedPhase` (`firTargetPhase + firPeqPhase`,
`src/lib/band-evaluator.ts:404`) becomes `modelPhase`. **Composite mode
ignores it** — Rust recomputes a composite phase internally from the
magnitudes (see §3 below).

## 2. Rust dispatch on `phase_mode`

`src-tauri/src/fir/mod.rs:564-579` — chooses `effective_linear`:
```rust
let effective_linear = match config.phase_mode {
    PhaseMode::LinearPhase => true,
    PhaseMode::MixedPhase => config.gaussian_min_phase_filters.is_empty(),
    PhaseMode::MinimumPhase | PhaseMode::HybridPhase => false,
    PhaseMode::Composite => config.linear_phase_main,
};
```

For TS-sent `Composite` + `linear_phase_main=false`:
`effective_linear = false`.

Phase for the IFFT, `src-tauri/src/fir/mod.rs:593-638`:
- Composite branch (lines 593-610) calls
  `crate::fir::helpers::compose_target_phase(&lin_mag, &lin_peq, n_fft,
  n_bins, sample_rate, linear_phase_main, subsonic_cutoff_hz,
  noise_floor_db)`.
- Other branches (lines 611-638) handle LinearPhase / MixedPhase /
  MinimumPhase / HybridPhase.

PEQ phase is **not added again** for Composite:
- `src-tauri/src/fir/mod.rs:642-648` —
  `if Composite { vec![0.0; n_bins] } else if has_peq { Hilbert(lin_peq) }`.

Final IFFT phase:
- `src-tauri/src/fir/mod.rs:650-653` —
  `phase_rad = target_phase_rad + peq_phase_rad`.

## 3. `compose_target_phase` / `composite_phase_inner`

`src-tauri/src/fir/helpers.rs:455-495` (`composite_phase_inner`) — three
independent Hilbert sources:

```rust
let base_mag: Vec<f64> = (0..n)
    .map(|k| (total_mag_db[k] - subsonic_mag_db[k] - peq_mag_db[k]).max(noise_floor_db))
    .collect();
let base_phase = if linear_phase_main {
    vec![0.0_f64; n]
} else {
    minimum_phase_from_magnitude(&base_mag, n_fft)
};
let peq_phase = …; // Hilbert(peq_mag_db) or zeros
let subsonic_phase = …; // Hilbert(subsonic_mag_db) or zeros
(0..n).map(|k| base_phase[k] + peq_phase[k] + subsonic_phase[k]).collect()
```

`linear_phase_main=false` → `base_phase = Hilbert(target − subsonic − peq)`,
i.e. the main-filter min-phase only. Sum of three min-phase contributions
is itself min-phase (sum of causal cepstral parts stays causal).

`compose_target_phase` (`src-tauri/src/fir/helpers.rs:501-522`) is the
public entry; it calls `subsonic_mag_db_lin(n_bins, sample_rate,
cutoff_hz)` to build `subsonic_mag_db` then forwards to
`composite_phase_inner`.

## 4. Initial impulse: IFFT + windowing

`src-tauri/src/fir/mod.rs:656-690`:
```rust
let mut spectrum = assemble_complex_spectrum(&lin_mag, &phase_rad, n_fft);
engine.fft_inverse(&mut spectrum);
let norm = 1.0 / n_fft as f64;
let mut impulse: Vec<f64> = spectrum.iter().map(|c| c.re * norm).collect();

if effective_linear {
    circular_shift_to_center(&mut impulse);
    let window = generate_window(n_fft, &config.window);
    for (i, w) in window.iter().enumerate() { impulse[i] *= w; }
} else if config.phase_mode == PhaseMode::MixedPhase && … {
    // peak-centered full window
} else {
    let half_win = generate_half_window(n_fft, &config.window);
    for (i, w) in half_win.iter().enumerate() { impulse[i] *= w; }
}
```

For Composite + `linear_phase_main=false`:
- `effective_linear=false` (line 578).
- Not MixedPhase.
- Hits the `else` arm at lines 684-690 → **half-window only, no shift**.

`generate_half_window`
(`src-tauri/src/fir/windowing.rs:13-18`) returns the right half of a
length-`2n` symmetric window — `half_win[0]=peak (1.0)`, decays to 0 at
`half_win[n-1]`. So a causal impulse with peak at `samples[0]` keeps its
peak.

## 5. Iterative refine

`src-tauri/src/fir/helpers.rs:79-272` (`iterative_refine`).

Linear-phase predicate inside the loop:
- `src-tauri/src/fir/helpers.rs:113-114` —
  `is_linear_phase = LinearPhase || (Composite && linear_phase_main)`.

Per-iteration phase recompute (b139.3.4 + b140.1):
- `src-tauri/src/fir/helpers.rs:120-128` — `recompute_min_phase`,
  `recompute_composite`, `subsonic_mag_fixed`.
- `src-tauri/src/fir/helpers.rs:190-217` — at the end of each iter,
  if Composite, `iter_phase = composite_phase_inner(refined_db, sub,
  peq_mag_db, n_fft, config.linear_phase_main, noise_floor_db)`.
- `src-tauri/src/fir/helpers.rs:219-220` —
  `assemble_complex_spectrum(&refined_db, &iter_phase, n_fft)` then
  `engine.fft_inverse`.

Per-iteration windowing:
- `src-tauri/src/fir/helpers.rs:226-228` — `if is_linear_phase
  { circular_shift_to_center(impulse); }`. **Skipped** when
  `linear_phase_main=false`.
- `src-tauri/src/fir/helpers.rs:250-263` — Composite branch: full window
  if `linear_phase_main`, else half window. Matches the initial
  windowing in `generate_model_fir`.

## 6. After iter loop: realized phase + normalisation

`src-tauri/src/fir/mod.rs:704-769`:
- Line 717: `let delay_samples = if effective_linear { (n_fft / 2) as f64
  } else { 0.0 };` — for **display only** (subtracted from realised
  phase to show "excess" phase). Does not modify `impulse`.
- Line 752-753: `realized_mag` / `realized_phase` interpolated back onto
  the caller-side `freq` (b140.6 then resamples again in TS).
- Line 760-766: `norm_db = max(realized_mag)`, `impulse *= 10^(-norm_db/20)`.
  Pure amplitude scale — no time shift.

`time_ms` is naive sample index × `1000/sr`, `src-tauri/src/fir/mod.rs:774-775`:
```rust
let dt_ms = 1000.0 / config.sample_rate;
let time_ms: Vec<f64> = (0..n_fft).map(|i| i as f64 * dt_ms).collect();
```
So `impulse[i]` lives at `time_ms[i] = i / sr`. A min-phase impulse
should peak at `i=0` → `time_ms[0]=0`.

## 7. Subsonic ON case (Min-Phase user + subsonic_protect)

Same pipeline. The only differences from §3 / §4:
- `subsonic_cutoff_hz = hp.freq_hz / 8` (TS `band-evaluator.ts:370`).
- `subsonic_mag_db_lin(n_bins, sample_rate, Some(cutoff))` returns a
  Butterworth-8 HP magnitude on the linear FFT grid; otherwise zeros.
  Reconstructed phase = `Hilbert(base_mag) + Hilbert(subsonic_mag) +
  Hilbert(peq_mag)`. Each Hilbert source is min-phase, sum is min-phase.
- Initial windowing: still the half-window branch (linear_phase_main is
  unchanged).

## 8. Where a linear-phase / constant-delay artefact could leak in

The trace shows a consistent "min-phase" path for `linear_phase_main=false`
(no center shift, half-window only, recomputed composite phase per iter).
Two callouts that the read-only trace cannot fully clear:

1. **`linearMain` is product-AND, not user-AND.** TS:
   `src/lib/band-evaluator.ts:365-368`
   ```ts
   const isUserLin = (f) => !f || f.linear_phase === true;
   const linearMain = isUserLin(hp) && isUserLin(lp);
   ```
   If the user has set `linear_phase=true` on **either** filter (HP-only,
   LP-only, or both), `linearMain=true` and the FIR centres + full-windows.
   For Band 1 (LP=200 LR4): `linearMain = isUserLin(null) && isUserLin(lp)
   = true && lp.linear_phase`. So the eventual linear-phase output is
   gated entirely by the LP filter's `linear_phase` flag. Worth verifying
   the actual `linear_phase` on the user's project — Pre-ring 4.25 ms +
   Causality 51% are the textbook signature of `effective_linear=true`,
   which only happens when the LP's flag is `true`. Band 3 (HP=2000 LR4)
   showing Pre-ring 0.06 ms + Causality 78% is consistent with that band
   having `linear_phase=false` on its HP.

2. **Initial impulse before iter 0.** `src-tauri/src/fir/mod.rs:684-690`
   only half-windows; it does **not** verify the IFFT result is actually
   causal. If `compose_target_phase` produced a phase whose IFFT places
   the peak away from `samples[0]` (e.g. due to numerical artefacts in
   `minimum_phase_from_magnitude` when most of `base_mag` is clamped to
   `noise_floor_db`), the half-window passes the off-centre impulse
   through unchanged. The iter loop then refines around that off-centre
   shape (no `circular_shift_to_center` for non-linear paths,
   `helpers.rs:226-228`). Worth printing the peak index of `impulse`
   right after `engine.fft_inverse` at `mod.rs:659-662` for the failing
   case to rule this in or out.

## 9. End-to-end summary

```
TS evaluateBandFull (band-evaluator.ts:350-424)
  ├─ phase_mode = "Composite"                    [line 421]
  ├─ linear_phase_main = isUserLin(hp) && isUserLin(lp)   [line 367-368]
  └─ subsonic_cutoff_hz = hp.freq_hz / 8 if active        [line 370]
       │
       ▼
Rust generate_model_fir (mod.rs:535-790)
  ├─ effective_linear = linear_phase_main         [line 578]
  ├─ target_phase_rad = compose_target_phase(...) [line 593-610]
  │     │
  │     └─ composite_phase_inner (helpers.rs:455-495)
  │           ├─ base_mag = total - subsonic - peq
  │           ├─ base_phase = Hilbert(base_mag)   [linear_phase_main=false]
  │           ├─ peq_phase = Hilbert(peq_mag)
  │           └─ subsonic_phase = Hilbert(subsonic_mag)
  ├─ phase_rad = target_phase_rad + peq_phase_rad [line 650-653]
  ├─ assemble_complex_spectrum + IFFT             [line 656-662]
  ├─ effective_linear=false  → half-window only   [line 684-690]
  └─ iterative_refine                             [line 694-702]
        ├─ FFT impulse → realized                 [helpers.rs:139-142]
        ├─ refined_db update                      [helpers.rs:160-181]
        ├─ recompute_composite phase              [helpers.rs:194-216]
        ├─ assemble + IFFT                        [helpers.rs:219-223]
        ├─ no center shift (is_linear_phase=false)[helpers.rs:226-228]
        └─ Composite + !linear_phase_main → half-window [helpers.rs:258-263]

Returned: impulse[0..n_fft] expected peak at sample 0 (causal).
time_ms[i] = i / sr (mod.rs:774-775).
```

Subsonic ON adds a second Hilbert source inside `composite_phase_inner`
(`helpers.rs:486-490`); pipeline is otherwise identical.

# `minimum_phase_from_magnitude` — read-only trace vs cepstral reference

Investigation only. No code changes, no tests, no bump.

## 1. Current implementation

Single definition: `src-tauri/src/dsp/phase.rs:10-51` (re-exported via
`src-tauri/src/dsp/mod.rs:11`).

```rust
// src-tauri/src/dsp/phase.rs
 1: use num_complex::Complex64;
 2: use super::fft::FftEngine;
 3:
 4: /// Compute minimum phase from magnitude spectrum via Hilbert transform.
 5: ///
 6: /// Input: `mag_db` — magnitude in dB for positive frequency bins (DC to Nyquist).
 7: /// `n_fft` — FFT size (must be even, typically next power of 2 >= 2 * mag_db.len()).
 8: ///
 9: /// Returns: minimum phase in **radians** for positive frequency bins (same length as input or n_bins).
10: pub fn minimum_phase_from_magnitude(mag_db: &[f64], n_fft: usize) -> Vec<f64> {
11:     let n_bins = n_fft / 2 + 1;
12:     let ln10_over_20 = 10.0_f64.ln() / 20.0;
13:
14:     // Build ln_magnitude as a real signal of length n_fft
15:     let mut ln_mag_signal: Vec<Complex64> = Vec::with_capacity(n_fft);
16:
17:     for i in 0..n_bins {
18:         let ln_val = mag_db[i.min(mag_db.len() - 1)] * ln10_over_20;
19:         ln_mag_signal.push(Complex64::new(ln_val, 0.0));
20:     }
21:     // Mirror for negative frequencies
22:     for i in 1..(n_fft - n_bins + 1) {
23:         let idx = n_bins - 1 - i;
24:         ln_mag_signal.push(Complex64::new(ln_mag_signal[idx].re, 0.0));
25:     }
26:
27:     // FFT
28:     let mut engine = FftEngine::new();
29:     engine.fft_forward(&mut ln_mag_signal);
30:
31:     // Apply Hilbert window
32:     ln_mag_signal[0] *= Complex64::new(1.0, 0.0);
33:     for i in 1..n_fft / 2 {
34:         ln_mag_signal[i] *= Complex64::new(2.0, 0.0);
35:     }
36:     if n_fft > 1 {
37:         ln_mag_signal[n_fft / 2] *= Complex64::new(1.0, 0.0);
38:     }
39:     for i in (n_fft / 2 + 1)..n_fft {
40:         ln_mag_signal[i] = Complex64::new(0.0, 0.0);
41:     }
42:
43:     // IFFT
44:     engine.fft_inverse(&mut ln_mag_signal);
45:     let norm = 1.0 / n_fft as f64;
46:
47:     // Extract imaginary part = minimum phase (radians)
48:     (0..n_bins)
49:         .map(|i| -ln_mag_signal[i].im * norm)
50:         .collect()
51: }
```

### Step-by-step annotation (current)

| Lines | Domain | What it does |
|---|---|---|
| 12 | scalar | `ln10/20` to convert dB → natural-log magnitude. |
| 14-25 | freq | Build a real, conjugate-symmetric N-point spectrum whose value at bin k is `ln|H(k)|` (NOT `log\|H\|` of a time signal). Bins 0..n_bins are dB·ln10/20; bins n_bins..n_fft are the conjugate-symmetric mirror (real-valued, so just the same value). |
| 27-29 | spec → "time"-ish | FFT of `ln_mag_signal`. Note: `ln_mag_signal` is a function of frequency, not of time. Forward-FFT'ing it produces a sequence indexed by what would be "quefrency" — i.e. real cepstrum on this domain. |
| 31-41 | quefrency | Apply the Hilbert window: keep DC and Nyquist as-is, double the lower half (bins 1..N/2-1), zero the upper half (bins N/2+1..N-1). This is the standard analytic-signal mask. |
| 43-44 | back | Inverse FFT brings the windowed signal back to the frequency domain — but now complex: real part ≈ original `ln|H|`, imaginary part = its Hilbert transform. |
| 47-50 | freq | Return `−Im(z) / n_fft` for the n_bins positive-frequency entries. The minus sign yields the conventional minimum-phase sign. |

### What the reference cepstral algorithm prescribes

Hofmann/Smith style ("Mathematical Theory of Discrete Signal Processing",
Oppenheim §7.4), reformulated for the goal "given `|H(ω)|` build a
causal impulse `h[n]` whose magnitude response is `|H(ω)|`":

1. `log_mag` ← `ln|H(ω)|`, with a floor against `−∞` (e.g. `max(ln|H|, ln(1e-12))`). Length N over the full circle.
2. **Real cepstrum**: `c[n] = real( IFFT( log_mag ) )` — N-point IFFT into the time domain.
3. **Causal fold**: build `c_min[n]` of length N where  
  `c_min[0] = c[0]`,  
  `c_min[n] = 2·c[n]` for `n = 1 .. N/2 − 1`,  
  `c_min[N/2] = c[N/2]`,  
  `c_min[n] = 0` for `n = N/2 + 1 .. N − 1`.
4. **Complex cepstrum back to freq**: `log_H_min = FFT( c_min )`. By construction `Re(log_H_min) ≈ log_mag` and `Im(log_H_min)` is the minimum phase.
5. Either return `Im(log_H_min)` as the minimum phase, **or** form `H_min = exp(log_H_min)` and `h[n] = real( IFFT( H_min ) )` to get a causal impulse with peak at `n = 0`.

## 2. Diff: current implementation vs reference

The current code is operationally equivalent to the reference for steps
1–4 once you map the variable names — but the *direction* of the FFTs
is swapped. Reference does **IFFT first** (freq → cepstrum/time),
windows in the cepstrum domain, then **FFT** (cepstrum/time → freq) to
get `log_H_min`. Current does **FFT first** then **IFFT**.

That swap is mathematically reversible for a windowing operation that's
even-symmetric in its support, *except* for the `1 / N` normalisation —
which the current code applies once (line 45) on the IFFT result. Net
factor matches the reference up to a sign convention. The minus sign at
line 49 absorbs that sign.

So the function reproduces the cepstral result, but with the
forward/inverse FFT roles swapped relative to canonical references. The
rest of the differences are:

- **No magnitude floor** before `ln`. `mag_db` is consumed as-is. If a
  caller passes a clamp via `noise_floor_db` (every call site does —
  see §3), `ln(10^(noise_floor/20)) = noise_floor · ln10/20` is still a
  *finite* but extremely large negative number (e.g. `noise_floor =
  −150 dB → ln|H| ≈ −17.27`). That's not infinite, but it dominates the
  cepstrum: a flat sea of `−17.27` for the inactive bins contributes a
  large DC quefrency component plus high-quefrency "ripple" from the
  cliff between active and inactive bins. The Hilbert windowing
  preserves these high-quefrency components and they re-appear in
  `Im(log_H_min)` as a near-linear phase ramp — i.e. **constant group
  delay** = peak shifted away from `n = 0`. This matches the diagnostic
  finding: 7.2 % active bins → peak at idx 204; 94.2 % active → idx 2.
- **No magnitude floor *raise* for the cepstrum step**. The reference
  recipe usually clamps `|H|` to something like `−60 dB` for cepstral
  computation (numerical stability), even when the synthesised spectrum
  is allowed to fall to `−150 dB` after the fact. The current code uses
  whatever floor the caller passed, which here is `noise_floor_db =
  −150 dB`. Tighter floor would shrink the cliff and the resulting
  group-delay artefact, but won't *eliminate* it for sparse spectra.
- **No causal-fold invariant check**. Reference algorithm guarantees
  that the IFFT of `H_min = exp(log_H_min)` is causal (peak at n=0)
  *by the construction of the fold*. Current code skips the explicit
  `c_min[N/2+1..] = 0` step in time domain — it does the equivalent in
  the frequency domain (zero out bins `n_fft/2+1..n_fft` at lines
  39-41 *after* the FFT-then-window). For the windowing of the same
  function this is mathematically equivalent (DFT is unitary), but it
  removes a natural place to enforce the causality invariant *and* the
  floor against `−∞`.
- **Returns `n_bins` phase values**, not `N` — caller is then
  responsible for rebuilding the conjugate-symmetric spectrum
  (`assemble_complex_spectrum`, `helpers.rs:528-551`). Compatible with
  the reference: both return half-spectrum.

In short: the algorithm is the cepstral one, but it works on `mag_db`
directly without any near-zero floor, so for sparse spectra (≤10 %
active bins) the cliff between active and inactive log-magnitude
generates a quefrency-domain feature that aliases into a constant-group-
delay phase term — exactly what produces the observed peak shift.

## 3. Call sites

All call sites in this repo (Rust only — TS does not call this
directly; it goes through Tauri commands that wrap the same function).

| File:line | Caller | Magnitude input | Clamping before call |
|---|---|---|---|
| `src-tauri/src/dsp/phase.rs:10` | function definition | — | — |
| `src-tauri/src/dsp/mod.rs:11` | re-export | — | — |
| `src-tauri/src/lib.rs:264` | `compute_minimum_phase` Tauri command | `lin_mag` (resampled from caller's grid onto linear FFT grid). Frontend uses this for SPL phase reconstruction. | Frontend caller may clamp; backend doesn't add a floor. |
| `src-tauri/src/fir/mod.rs:99` | `generate_fir` (legacy path) for `PhaseMode::MinimumPhase` | `limited` (= `lin_correction`, narrowband-limited correction) | Limited via `limit_narrowband_boost` and clamped to `noise_floor_db` upstream. |
| `src-tauri/src/fir/mod.rs:101` | `generate_fir` for `PhaseMode::MixedPhase` | same `limited` | same |
| `src-tauri/src/fir/mod.rs:102` | `generate_fir` for `PhaseMode::HybridPhase` | same `limited` | same |
| `src-tauri/src/fir/mod.rs:106` | `generate_fir` for `PhaseMode::Composite` | same `limited` | same |
| `src-tauri/src/fir/mod.rs:287` | `generate_hybrid_fir` | `lin_correction` | as above |
| `src-tauri/src/fir/mod.rs:630` | `generate_model_fir` MixedPhase per-filter Gaussian | `filt_mag` (synthesised Gaussian filter mag) | `filt_mag.iter_mut().for_each(\|v\| *v = v.max(noise_floor_db))` immediately before the call (line 628). |
| `src-tauri/src/fir/mod.rs:637` | `generate_model_fir` else (MinimumPhase / HybridPhase fallback) | `lin_target` | clipped to `noise_floor_db .. max_boost_db` at line 547. |
| `src-tauri/src/fir/mod.rs:645` | `generate_model_fir` non-Composite PEQ phase | `lin_peq` | clipped to `−60.0 .. max_boost_db` at line 556. |
| `src-tauri/src/fir/mod.rs:1016` | unit test | synthesised `correction` | test fixture. |
| `src-tauri/src/fir/helpers.rs:203` | `iterative_refine` per-iter for MinimumPhase | `refined_db` (running magnitude after weighted-error correction) | `refined_db[k]` is clamped to `noise_floor_db .. max_boost_db` inside the loop (line 173-175). |
| `src-tauri/src/fir/helpers.rs:496` | `composite_phase_inner` main-component | `base_mag = total − subsonic − peq` | `.max(noise_floor_db)` at line 469. |
| `src-tauri/src/fir/helpers.rs:504` | `composite_phase_inner` PEQ component | `peq_mag_db` (caller-clamped) | passed through. |
| `src-tauri/src/fir/helpers.rs:511` | `composite_phase_inner` subsonic component | `subsonic_mag_db` (analytic Butterworth-8) | naturally bounded. |

Every production call site either:
- clamps to `config.noise_floor_db` (typically `−150 dB` from the export
  panel default), or
- does no extra clamping and inherits whatever the caller set.

None of them apply a tighter cepstrum floor before the Hilbert step.

## 4. Bonus — why iterative_refine diverges on Band 3 (HP=2000, sr=48k)

Diagnostic shows `iter=1 max_err=22.777 dB → iter=3 max_err=31.113 dB`,
peak idx=2 after initial IFFT (very close to causal but not exactly).

`src-tauri/src/fir/helpers.rs:137-272` — relevant iter steps:

1. Take current `impulse` → FFT → `spec` (`helpers.rs:139-142`).
2. For each linear bin, compute realised `mag_db` from `|spec[k]|`,
   compare to `target_correction_db[k]`, push the weighted, damped (×0.7)
   error into `refined_db[k]`, clamped to noise floor / max boost
   (`helpers.rs:160-181`).
3. For Composite mode: `iter_phase = composite_phase_inner(refined_db,
   subsonic, peq, n_fft, linear_phase_main, noise_floor_db)`
   (`helpers.rs:194-217`). This is **the same** `minimum_phase_from_magnitude`
   call as in §1 — recomputed every iteration on `refined_db`.
4. `assemble_complex_spectrum(refined_db, iter_phase, n_fft)` → IFFT →
   new `impulse` (`helpers.rs:219-223`).
5. For Composite + `linear_phase_main=false`: half-window only, no
   center shift (`helpers.rs:258-263`).

Hypothesis (single paragraph): the initial impulse already has its peak
at idx=2 instead of idx=0 because `minimum_phase_from_magnitude` on the
sparse `lin_mag` injected a small constant group-delay term (§2). The
half-window at iter 1 starts from value ≈ 1.0 at index 0 and drops
slowly, so it can't move the peak back to idx=0 — it preserves the
shape. After FFT, the realised magnitude reflects that slightly
delayed-peak time-domain shape, which differs from `target_correction_db`
by a small phase ripple. The error-correction step at line 162 attributes
that mag-domain ripple to magnitude error and bumps `refined_db` by 0.7
× the weighted residual. Then `composite_phase_inner` on the bumped
`refined_db` runs `minimum_phase_from_magnitude` again and produces a
new shifted impulse — *with a slightly different shift*, because the
sparseness pattern is now perturbed. The mag-error gain accumulates each
iteration instead of converging; with `freq_weighting=true` the
crossover region's weights amplify the mismatch, so `max_err` grows
22.8 → 31.1 dB. In short: the divergence is the consequence of the
phase reconstruction having a magnitude-dependent linear-delay artefact;
the refinement loop interprets the artefact as a magnitude error and
keeps trying to "correct" it.

---

End of read. No code changed in this report (the only edit elsewhere
this session is the `[DIAG ACTIVE]` startup marker added to
`src-tauri/src/lib.rs:557` per the CLAUDE.md rule, plus the previously
present `[DIAG-MP]` instrumentation in `fir/mod.rs` and `fir/helpers.rs`).

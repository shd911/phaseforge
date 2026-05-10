# Post-IFFT shift experiment — cargo test report

Experimental fix in place but **NOT** committed and **NOT** bumped.
`[DIAG ACTIVE] post-shift: rotate impulse peak to idx=0` is in
`src-tauri/src/lib.rs:557`.

Patch applied at three sites (initial + iterative_refine for both
MinimumPhase/MixedPhase/HybridPhase and Composite + !linear_phase_main):

- `src-tauri/src/fir/mod.rs:684-694` — initial assembly, before
  half-window.
- `src-tauri/src/fir/helpers.rs:244-251` — iter loop, MinimumPhase /
  MixedPhase / HybridPhase branch.
- `src-tauri/src/fir/helpers.rs:264-271` — iter loop, Composite +
  `!linear_phase_main` branch.

Linear-phase paths and MixedPhase + Gaussian (peak-centered) are
untouched.

## Cargo summary

```
cargo test --lib (full suite)
test result: 168 passed; 2 failed; 1 ignored; 0 measured
```

New test added: `dsp::phase::tests::min_phase_impulse_peaks_at_zero_for_sparse_spectrum`
— **PASSES** (peak idx ≤ 5 for LP=200 / BP / HP=2000 sparse cases).

## Failures

### 1. `fir::tests::test_fir_magnitude_matches_target_all_phase_modes` — **(b) functional invariant**

```
LinearPhase: maxErr=0.00dB@351Hz, RMS=0.00dB, norm_db=-0.01
MinimumPhase: maxErr=9.12dB@1186Hz, RMS=6.25dB, norm_db=0.18

panic: src/fir/mod.rs:1639
MinimumPhase: magnitude error 9.12 dB @ 1186 Hz exceeds 3 dB threshold.
This likely means half-window was applied to non-causal impulse.
```

LinearPhase still PASSES at 0 dB error. Only MinimumPhase is broken.
The panic message itself spells out the cause: "half-window was applied
to non-causal impulse". The post-shift moves the peak to idx 0, but the
**pre-peak** content (everything that was at indices `0 .. peak_idx`
before shifting) circularly wraps to the **end** of the array, where
the half-window value is ~0. That content is real impulse energy
(just delayed), and it's being multiplied by ≈ 0, which removes it
from the realised magnitude.

This is not a phase-domain artefact — the realized FFT of the
windowed-and-rotated impulse no longer matches the input target
magnitude in the passband. Functional regression, not a snapshot
update.

### 2. `fir::tests::test_4band_all_filter_types` — **(b) functional invariant**

```
=== 4-Band x 4-FilterType FIR Test ===
--- BW4 (order=4, linear=false) ---
  Sub | norm_db=-3.76 dB | wav=512 KB
  LowMid | norm_db=+0.04 dB | wav=512 KB

panic: src/fir/mod.rs:1315
BW4/MidHigh: realized_mag at 1286 Hz = -5.0 dB (expected ~0)
```

Same root cause: 4-band cascade test for BW4 (Butterworth order 4,
non-linear-phase = min-phase user choice). MidHigh band's realised
magnitude is 5 dB low at 1286 Hz, well inside its passband. The Sub
band has `norm_db = -3.76 dB`, suggesting a global level shift from
the same energy-loss mechanism.

Functional regression.

## Classification

| # | Test | Category | Notes |
|---|---|---|---|
| 1 | test_fir_magnitude_matches_target_all_phase_modes | **(b)** | Magnitude tolerance: MinimumPhase RMS 6.25 dB > 3 dB threshold. |
| 2 | test_4band_all_filter_types | **(b)** | Realised mag mismatch in BW4 passband. |

Both failures are category **(b)** — functional invariants (realised
magnitude must match target). No category (a) golden-snapshot or
category (c) phase-related failures.

## Why this version of the experiment doesn't merge

`impulse.rotate_left(peak_idx)` preserves DTFT magnitude on the
infinite circle, but combined with the half-window it doesn't:
half-window applies a length-`n` taper from full-1.0 at index 0 to
0 at index n-1. The rotation moves what was the **pre-peak** energy
(indices `0 .. peak_idx`) to the **end** of the array (indices
`n - peak_idx .. n`), where the half-window value is ≈ 0. That energy
is then erased.

For a true minimum-phase response the pre-peak content is essentially
zero so this doesn't matter. For our "min-phase with cepstral leak"
case the pre-peak content is non-trivial (it's where the impulse
actually starts) — discarding it corrupts the magnitude response.

In other words, the rotation is not the wrong direction of fix, but
rotation **alone** is insufficient. To preserve magnitude one of the
following has to also change:

- Apply a **peak-centred** full window instead of half-window (same as
  MixedPhase path at `mod.rs:672-683`). This keeps both sides of the
  peak. Trade-off: causality drops to ~50 % since pre-peak content is
  preserved.
- Re-derive a phase that genuinely places the peak at idx 0 (i.e. fix
  the cepstrum step itself: lifter, smoothing, or a different
  algorithm). The cepstral floor experiment from b140.7 attempt 1
  showed that simple clamping doesn't reach idx 0 without breaking
  passband magnitude either.
- Accept the off-zero peak and document the constant group delay as
  expected behaviour for sparse spectra.

## Recommendation

**STOP — do not merge this experiment.** Both failures are
category (b). Per the prompt rule, this requires a discussion before
proceeding to a fix variant.

Options to discuss:
1. Peak-centred full-window (MixedPhase-style path) for sparse
   spectra. Loses causality 100→50 % but preserves magnitude.
2. Lifter-based cepstrum smoothing.
3. Pre-smooth `lin_mag` only for the cepstrum step.
4. Status quo (b140.6) + document min-phase artefact for sparse LP.

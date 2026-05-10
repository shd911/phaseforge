# IIR path grid audit (b140.7.3)

Read-only audit. No DSP changes. Goal: locate any sr-dependent grid
logic in the IIR path that could produce divergent UI plot vs WAV
export behaviour at sr = 44.1/48 k.

## Summary first

The IIR path (`src-tauri/src/fir/iir_path.rs::generate_min_phase_fir_iir`)
does **not** apply any sr-dependent log-grid resampling to the
`impulse` it returns. The impulse is built directly in the time
domain via biquad cascade at the requested `sample_rate`, with
length `cfg.taps`. Both `[IIR PATH DIAG]` and `[EXPORT WAV DIAG]`
b140.7.1 traces confirmed bit-identical impulses in memory all the
way to `export_fir_wav` — so the impulse content REW reads matches
what UI plots.

The `freq` log grid that *is* sr-dependent (5 Hz..`fMaxFir`,
`fMaxFir = min(40 kHz, sr·0.95/2)`) is used **only** for two
output-only operations:

1. resampling `realized_mag` / `realized_phase` from the linear FFT
   grid onto the caller's log grid for the UI plot
   (`iir_path.rs:388-390`);
2. computing the normalisation `norm_db = max(realized_mag)` and
   scaling the impulse by `10^(-norm_db/20)` (`iir_path.rs:392-398`).

Step 2 *does* multiply the impulse by an sr-dependent constant — but
it's a **constant** (no shape distortion), and for the test cases
(LP=200 / BP / HP=2000 LR4) the realised passband peak is well
inside both the sr=48 k log grid (5..22800) and the sr=176.4 k log
grid (5..40000), so the resulting `norm_db` should differ between
sr's only by sub-dB amounts.

Conclusion: the audit doesn't find a smoking gun in the IIR path
itself for an sr=48 k specific REW failure. The hypothesis listed in
the prompt (sr-dependent grid that affects WAV but not UI plot)
**doesn't match the code**. A likelier root cause is in the WAV
encoder / REW reader format compatibility (see §6).

## 1. All sr-dependent grid points found

### IIR path (`src-tauri/src/fir/iir_path.rs`)

The IIR pipeline operates entirely in the time domain at the
caller's `sample_rate`. There is **no internal log grid**. Specific
sr-dependencies, all of which are physically correct (= the digital
filter response sampled at the exact caller sr):

| Site | Use |
|---|---|
| `prewarp(fc, sr)` `iir_path.rs:103` | bilinear pre-warp `(sr/π)·tan(π·fc/sr)`; physically required so the digital filter realises its corner exactly at `fc` for the given sr. |
| `lp_biquad_q` / `hp_biquad_q` `iir_path.rs:108-149` | digital biquad coefficients via bilinear; `K = 2·sr` enters every coefficient. Correct, sr-aware. |
| `lp_first_order` / `hp_first_order` `iir_path.rs:151-178` | same bilinear treatment for first-order sections. |
| `cascade_impulse` `iir_path.rs:301-310` | applies δ[n] to digital biquad cascade and reads `n_fft` samples. The output IS the digital filter impulse response sampled at sr — sr-correct by construction. |
| `apply_tail_taper` `iir_path.rs:438-449` | pure raised-cosine on last 5 % of samples. Index-based (`n / 20`), depends only on `n_fft`, not sr. No sr-specific behaviour. |
| `lin_freq` `iir_path.rs:388` | `(0..n_bins).map(|k| sr * k as f64 / n_fft as f64)` — linear FFT bin frequencies. Used for `interp_1d` resample of realised mag/phase onto caller's log grid. Output-only, doesn't touch `impulse`. |
| `dt_ms` / `time_ms` `iir_path.rs:400-401` | `1000.0 / sr` per sample. Output metadata for plot. Doesn't change `impulse`. |

### Caller side (`src/lib/band-evaluator.ts`)

| Site | Use |
|---|---|
| `fMaxFir = Math.min(40000, cfg.sampleRate / 2 * 0.95)` `band-evaluator.ts:374` | log grid upper bound. At sr=48 k → 22800; at sr=176.4 k → 40000. **This is the same b140.5/b140.6 sr split.** |
| `evaluate_target_standalone({ ..., fMin: 5, fMax: fMaxFir })` `band-evaluator.ts:377` | builds the log grid (`firFreqRaw`) used as `freq` parameter to BOTH the FFT path and the IIR path. |
| `appendNoiseFloorTail(firFreqRaw, …, cfg.sampleRate, …)` `band-evaluator.ts:386-389` | extends the log grid up to Nyquist·0.999 with a noise-floor mag tail. b140.5 fix for the FFT path's constant-clamp issue. |
| TS routing: `useIirPath` `band-evaluator.ts:411-417` | invokes `generate_model_fir_iir` with the post-tail `firFreq` as `freq`. |

The **same `firFreq`** flows into IIR path's `input.freq`. Inside
the IIR path it's used at `iir_path.rs:389-390` (resample mag/phase
onto log grid for plot) and at `iir_path.rs:394` (find norm_db on
that log grid).

### Other Rust sites

`src-tauri/src/fir/mod.rs:514` / `:1102` — log grid construction
inside the FFT path's IR / SUM-IR generators. Not on the WAV export
path; doesn't touch IIR.

`src-tauri/src/fir/mod.rs:1669-2156` — test fixtures only, all
`5..40000` or `5..fMaxFir` log grids for the FFT path tests.

## 2. UI plot vs WAV export — what's the same, what differs

### UI plot path

```
band-evaluator.ts evaluateBandFull
  → invoke generate_model_fir_iir
    → iir_path.rs::generate_min_phase_fir_iir
      → cascade_impulse                (impulse on time grid at sr)
      → apply_tail_taper               (5 % tail fade)
      → FFT(impulse)                   (linear FFT on n_bins bins)
      → interp_1d(lin_freq, ..., freq) (resample to log grid 5..fMaxFir)
      → norm_db = max(realized_mag)    (on log grid)
      → impulse *= 10^(-norm_db/20)    (scale impulse by constant)
      → realized_mag -= norm_db        (shift mag display)
    → returns FirModelResult { impulse, realized_mag, realized_phase, ... }
  ← evaluateBandFull packages into BandEvalResult.fir
plot consumer reads result.fir.realizedMag/realizedPhase on `freq`
```

### WAV export path

```
ControlPanel.tsx handleExport (or fir-export.ts exportBandWav)
  → generateBandImpulse → evaluateBandFull (SAME call as UI)
  → result.fir.impulse                  (already scaled by norm_linear)
  → invoke export_fir_wav { impulse, sampleRate, path }
    → lib.rs::export_fir_wav
      → fir::export_wav_f64(impulse, sample_rate, path)  (iir_path.rs:453)
        → 64-bit IEEE-float WAV, format=3, mono
```

### Differences

There are **none in the impulse data** — both paths consume
`result.fir.impulse` produced by the SAME `evaluateBandFull` call.
b140.7.1 diag confirmed: `[IIR PATH DIAG]` ≡ `[EXPORT WAV DIAG TS]`
≡ `[EXPORT WAV DIAG]` (same head, same peak_abs, same sum).

The only thing UI does that WAV doesn't is read `result.fir.realizedMag`
and `result.fir.realizedPhase` for the on-screen plot. REW reads the
WAV file, then computes its own FFT — which should produce the SAME
shape as PhaseForge's `realized_mag` (modulo display normalisation).

## 3. Why "broken at sr=48 k, OK at sr=88+" doesn't match the IIR
   pipeline structure

The b140.6 sr=48 k regression had a clear mechanism: TS produced a
log grid with sr-dependent `fMax` (22800 vs 40000), Rust returned
`realized_mag/phase` on that grid, but uPlot indexed positionally
against a different x-axis → ~½-octave visual shift. That bug lived
in the **plot path**, not the impulse / WAV path.

For b140.7 IIR:
- The impulse content is sr-correct (digital biquad cascade at the
  given sr).
- It is NOT resampled before WAV export.
- WAV header records the correct `sample_rate`.
- REW reads the WAV at the recorded sr and computes its own FFT.

For REW to show a notch instead of HP at sr=48 k while UI shows
correct HP, **either**:

(a) the impulse is actually wrong at sr=48 k and PhaseForge UI is
    masking the error somehow (e.g. plot resamples in a way that
    averages out a defect REW shows pointwise); or
(b) the WAV file is being misinterpreted by REW at sr=48 k specifically.

I cannot rule out (a) by reading the code alone — but the IIR
cascade math is the same at every sr, with bilinear pre-warping
that should keep coefficients accurate.

(b) is more suspicious because:
- `export_wav_f64` writes **64-bit IEEE float** (`format = 3,
  bits_per_sample = 64`, lib.rs:474-480). 64-bit float WAV is rare;
  many tools (REW included historically) only support 32-bit float.
- A reader that silently truncates 64-bit to 32-bit by reading 4
  bytes per sample at advertised stride 8 would interpret the data
  as garbage, producing notch-like artefacts.
- 32-bit float WAVs of the same impulse would be ~1 ulp of f32 less
  precise but otherwise identical in REW.

The "OK at 88+" pattern would still need a separate explanation under
(b) — perhaps REW treats 88.2 k WAVs through a different code path
that happens to interpret 64-bit float correctly while 48 k goes
through a legacy decoder. Speculative.

## 4. Hypothesis

Most likely: **`export_wav_f64` writes 64-bit IEEE float, which REW
either does not support or supports inconsistently across sr.** The
identical impulse data flows through, but the WAV container format
is incompatible with REW's expectations.

Secondary possibility: an sr-dependent precision issue in the IIR
biquad coefficients at low fc/sr ratios for extreme-Q sections. For
LR4 = 2× BU4, the highest-Q biquad is Q≈1.31. At sr=48 k, fc=2000:
pole magnitude in z-plane ≈ 0.91 (well inside unit circle). At
sr=176.4 k, |pole| ≈ 0.97 (closer to circle, longer ringing). Both
stable. Doesn't obviously favour 88+ over 48.

## 5. Proposed fix (pending user approval)

**Switch `export_wav_f64` to write 32-bit IEEE float instead of
64-bit.** REW universally supports 32-bit float WAV. The precision
loss is negligible for FIR coefficients (≪ 1 dB headroom error in
realised magnitude) and matches the convention used by REPhase, RePhase,
Acourate, FIRTRA, and all standard convolvers.

Concrete change (one site):

`src-tauri/src/fir/mod.rs:408` already has a `export_wav_f32` that
writes 32-bit float. Change `export_fir_wav` Tauri command in
`lib.rs:574` from `fir::export_wav_f64` to `fir::export_wav_f32`.

Risk: minimal. f32 has ~7 decimal digits precision; impulse peaks in
PhaseForge are normalised to 0 dBFS so values are in [-1, +1] — well
within f32 range. Truncation noise floor ≈ -149 dB.

Verification after the swap:
- Re-export the SAME failing test case (Band 3 HP=2000 sr=48 k).
- Open in REW. If frequency response now matches the model — confirmed.
- If still broken, hypothesis is wrong and we need (a) further diag
  on the actual impulse content at sr=48 k vs sr=176.4 k (e.g.
  external script that FFTs both WAVs and compares to model).

## 6. STOP

No DSP code changed. Bump applied (b140.7.1 → b140.7.3) per the
versioning rule, diag tracing from b140.7.1 left in place. Awaiting
user decision on whether to proceed with the f32 WAV fix or to
gather more evidence first.

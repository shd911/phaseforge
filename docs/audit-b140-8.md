# PhaseForge b140.8 — pre-release 7-vector audit

State: b140.7.14, 4 commits since b140.4 release tag.
Cargo: **180 lib + 4 REPhase + 10 other = 194 PASS / 0 FAIL**.
Vitest: **104/104 PASS**.

---

## 1. Architecture

**Routing matrix (`src/lib/band-evaluator.ts:411-417`):**

```
useIirPath = !linearMain
          && subsonicCutoff === null
          && isIirRealizable(hp)
          && isIirRealizable(lp)
isIirRealizable = LR / Butterworth / Custom (or null)
```

| Config | Path |
|---|---|
| Min-Phase main + non-Gaussian + no subsonic | **IIR** (`iir_path.rs`) |
| Linear-phase main | FFT cepstral (legacy) |
| Composite + subsonic | FFT cepstral |
| Gaussian (any side) | FFT cepstral |
| Bessel | FFT cepstral |
| Custom measured target | FFT cepstral |

Both paths share `FirModelResult` shape. Tauri commands:
`generate_model_fir` (legacy), `generate_model_fir_iir` (b140.7).

**Dead code check.** Existing FFT path (`PhaseMode::MinimumPhase`,
`MixedPhase`, `HybridPhase`) used by:
- `generate_fir` (`fir/mod.rs:99-106`) — old correction-spectrum entry,
  may still be called by `generate_hybrid_fir` and SUM IR paths.
- `iterative_refine` (`helpers.rs:131-244`) — reuses these enum variants.

These are NOT dead — they back the FFT cepstral pipeline that's still
the default for everything outside the IIR-applicable routing matrix.

**Finding A1 (minor):** `MixedPhase` and `HybridPhase` paths are not
exercised by any production TS caller — TS sends only `Composite`. They
remain as Rust API surface for future use / direct unit tests. No
action needed for b140.8.

---

## 2. DSP correctness

| Suite | Coverage |
|---|---|
| `rephase_compare.rs` (4 tests) | HP=2000 LR8 min-phase WAV vs REPhase reference, sr={44.1, 48, 88.2, 176.4k} — max Δmag 0.44 dB, Δphase 2.5° |
| `iir_path::tests` (8 tests) | LR4 LP/HP/BP centered-at-N/2, mag/phase match analytical, DC gain, UI plot guardrail (25°) |
| `band-evaluator.test.ts` snapshot (12) | targetMag/targetPhase/correctedMag/correctedPhase + extension fields for 6 fixtures |
| `band-evaluator-fir.test.ts` (5) | FIR identity, grid bounds + b140.5 tail, b140.6 resample |
| `evaluate-sum.test.ts` (32) | SUM target/corrected/IR aggregation across all b140.3.x features |
| `golden_pipeline.test.ts` (1 skipped) | Snapshot pipeline parity |
| FFT path golden hashes (`generate_fir_b139_*`) | bit-exact for legacy paths |

**Edge cases covered.** Gaussian + subsonic (FFT cepstral); PEQ-only
(through `evaluateBandFull`); no measurement (target-only).

**Finding A2 (informational):** PhaseForge LR convention =
`(BU-N)² = 2N effective order = 12N dB/oct`
(`target/mod.rs:487-490`, mirrored in IIR path
`iir_path.rs::build_filter_cascade::LinkwitzRiley` and in UI Slope
mapping). This is non-standard (textbook LR-N = N total order) but
internally consistent across DSP, REPhase reference matches, and UI
labels.

---

## 3. Backward compatibility

- Save schema (`filter.order` 1..8) unchanged — confirmed in
  `ControlPanel.tsx:298-318` (`withOverride` preserves all fields,
  `order` field is the canonical persistence).
- Slope dropdown derives display from `order`, writes back through
  `slopeToOrder` → `order` (no new fields).
- Existing `.pfproj` files load with the same `filter.order` values;
  UI auto-maps to slope.
- Tauri command `generate_model_fir` retains legacy signature.
- `generate_model_fir_iir` is additive — no breaking change.

**Finding A3 (informational):** UI labels in `formatFilterInfo`
(b140.7.13 fix) now show `LR48` for `order=4` (was `LR24`).
**Visible change for users**, but corrects a long-standing display bug.
Worth highlighting in release notes.

---

## 4. UI/UX consistency

- `FilterBlock` HP and LP both use the new Slope dropdown.
- `formatFilterInfo` (status row) uses the same `orderToSlope` helper.
- Manual PEQ panel does not have an "Order" field (PEQ uses Q + gain
  + freq), so no migration needed.
- `CrossoverDialog.tsx` — verified: also uses `order` directly (not
  Slope dropdown). For consistency could be migrated, but it's a
  spec-by-order dialog (user picking from canonical orders), not a
  user-pickable slope. Leave as-is.

**Finding A4 (followup, non-blocking):** `CrossoverDialog` could use
the same Slope dropdown for full consistency. Defer to a later UX
polish prompt.

---

## 5. Performance

- `iir_path.rs::cascade_impulse` — direct biquad application, O(N · S)
  where S = section count (≤ 16 for LR8 + LR8 + 8 PEQ). For taps=65536
  and 16 sections: 1M floating-point multiply-add ops. ~milliseconds
  on M1 (no FFT round-trip needed).
- FFT path same as before — N log N IFFT + iterative refine.
- WAV write: 524 KB Float64 sequential.
- TypeScript bundle size: no significant new deps — only inline
  helpers.

No regressions observed. IIR path is faster than FFT path for typical
crossover configs (no iterative_refine).

---

## 6. Documentation

- `CHANGELOG.md` — last entry b140.4. **Needs b140.8 entry.**
- `CLAUDE.md` — already updated:
  - Versioning rule (b140.7.1) in place.
  - End-of-prompt automation (b140.3.x) intact.
  - `Last reviewed:` outdated (2026-05-07). Should bump after release.
- `README.md` — not checked, may be stale on min-phase / IIR
  capabilities.
- `docs/release-notes-*.md` — present for b140.4. **Needs b140.8.**
- `docs/regression-checklist.md` — manual UI checklist; b140.7 IIR
  flow not yet listed.

**Finding A5:** Add a manual UI checklist item for "Min-Phase Export →
REW phase matches model" to `docs/regression-checklist.md`.
Non-blocking; can ship and add in a follow-up commit.

---

## 7. Test coverage

| Layer | Tests | Status |
|---|---|---|
| Unit (DSP primitives) | 180 cargo lib | PASS |
| Unit (IIR cascade math) | 8 in `iir_path::tests` | PASS |
| Integration (REPhase reference) | 4 in `rephase_compare.rs` | PASS |
| Integration (e2e_export) | 8 cargo | PASS |
| Integration (SUM pipeline) | 2 e2e_sum_real_project | PASS |
| Frontend snapshot | 91 (band-evaluator + fir + sum) | PASS |
| Frontend UI components | 13 (FilterBlock, ControlPanel etc.) | PASS |

Total **194 cargo + 104 vitest = 298 automated tests**.

**Gap A6 (informational):** No automated test for the user-facing
flow "select Min-Phase, change Slope from 24 to 48, export WAV,
verify file content". Covered indirectly by REPhase tests + UI
manual verify. Adding e2e Tauri test would harden but is not
blocking for b140.8.

---

## Summary

**Findings:**

- A1: MixedPhase / HybridPhase variants kept (no production caller, but Rust API). **Non-blocking.**
- A2: LR convention is `(BU-N)² = 12N dB/oct`. Internally consistent, REPhase-validated. **Document in release notes.**
- A3: UI labels now show correct slope (LR48 for order=4). **Visible change — release notes.**
- A4: `CrossoverDialog` could adopt Slope dropdown for consistency. **Defer.**
- A5: `regression-checklist.md` should add Min-Phase + REW item. **Defer.**
- A6: No e2e test for full Slope-change → WAV-export flow. **Defer.**

**Critical findings: 0.**

**GO for b140.8 release.**

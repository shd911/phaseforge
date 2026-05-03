# PhaseForge b137 — Project Lifecycle, Measurement Analysis & Q Envelope

Release umbrella covering b131 → b137. b130 was the last public build; this
one bundles the full project-lifecycle stack, post-import measurement
analysis, target-divergence detection, and a frequency-dependent Q ceiling.

## Project lifecycle

**b131 — Window close confirmation.** Cmd+Q / red traffic-light no longer
silently drop unsaved work. A backend `on_window_event` handler intercepts
`CloseRequested`, the frontend reuses the same Save / Don't Save / Cancel
dialog used by New / Open / Recent, then acks the close.

**b132 — Session undo/redo.** Light-state stack (PEQ bands, target,
alignment, FIR/export/PEQ params; measurement arrays excluded). Cmd+Z /
Shift+Cmd+Z work everywhere except inside text fields and during a PEQ
drag. Coalesce window absorbs slider repeat / wheel; explicit
begin/commitInteraction wraps PEQ marker drag and wheel for one entry per
gesture. Stack capped at 5 (both directions). Cleared on New / Open.
Apply preserves measurement / settings / firResult by band id, so existing
measurements survive undo.

**b133 — Named project snapshots.** `<project>/snapshots/<id>.pfproj`
copies with a description, plus `index.json`. inbox/target/export are not
duplicated, so a removed measurement comes back as null with a warning.
Manual save only — no auto snapshots. Restore reuses restoreState, leaves
`currentProjectPath` alone, marks isDirty=true and clears session undo.
Index writes are atomic (tmp+rename); snapshot files use `create_new` with
retry to avoid silent overwrites on same-second clicks. Versions menu
disabled on v1 (folder-less) projects.

**b134 — Shortcuts, tooltips, edge-case polish.** Centralized global
shortcuts in App.tsx with case-insensitive matching and a single
`isModalOpen` guard. New Cmd+Shift+V opens the Versions dialog (falls
back to a toast if the project has no folder). Restoring a project with
missing measurement files now surfaces one summary toast instead of
silent console warnings. Window title dirty marker switched to `•` for
consistency. Tooltips added to File menu / Undo / Redo / Versions row
actions per spec. Toast component with click-to-dismiss + cleared timers
on manual close.

## Measurement analysis

**b135 — Post-import measurement analysis.** Three Rust detectors
(noise floor low/high, LF window rolloff, HF cliff) exposed via
`analyze_measurement` Tauri command and unit-tested with synthetic
inputs. Noise-floor flag is gated by a 15 dB drop vs band median so a
flat response does not trip it; resonance suppression requires an
amplitude swing > 1.5 dB so legitimate slopes are not killed by
measurement ripple. Frontend `MeasurementAnalysisDialog` with
per-finding apply buttons; analysis runs after Import and Merge.
Recommendations apply via `pushHistory` so Cmd+Z reverts.
SettingsData carries `analysis` + `analysis_dismissed`; restoreState
shows the dialog at most once for the first non-dismissed band on
project load.

**b135.1 — "All clear" dialog on clean measurements.** runAnalysis
opens the dialog unconditionally; the empty-findings fallback renders a
centered ✓ + "Замер выглядит чисто" / "Анализ не выявил подозрительных
участков." Closing routes through `close(true)` so analysis_dismissed=true
is persisted.

**b135.2 — Single application model.** Per-finding apply button +
global "Применить все"; the latter wraps the loop in
`beginInteraction`/`commitInteraction` so the entire batch lands as
one undo entry. Module-level `_appliedIds` reset on each open so a
stale set from a previous band doesn't bleed in.

**b135.3 — Apply-all UX clarity.** Subtitle above the findings list
explains the two paths. Apply-all button reflects state:
"Применить все" → "Применить оставшиеся (N)" → "Все применены".
Buttons reordered to macOS HIG: Закрыть on the left, primary right.

## Target divergence + Q envelope

**b136 — PEQ stale on target change.** BandState carries
`peqOptimizedTarget` — a snapshot of HP/LP/exclusion at last successful
Optimize. `peqStale()` compares it to the current target (filter_type,
order, freq_hz, shape, q + zone bounds; reference_level and linear_phase
intentionally ignored). Snapshot is taken **before** the optimize await
so a concurrent target edit can't retroactively hide staleness. PEQ tab
gets an orange border + banner with Reoptimize / Clear; band tabs show a
small dot. FIR export gates on staleness with a confirm dialog. The
snapshot round-trips through `.pfproj`, b132 history, and b133 named
snapshots; legacy projects without the field stay non-stale.

**b137 — Frequency-dependent Q cap with warning markers.** New
`peq::q_envelope` module: `q_cap_at` and `q_warn_at` interpolate linearly
in log2(f) between 200 Hz (cap=12 / warn=8) and 2000 Hz (cap=4 / warn=3).
LMA `clamp_params`, greedy `estimate_q_from_peak_width` and
`merge_nearby_bands`, the post-greedy enforcer, the post-LMA enforcer,
and the seed-Q for added bands all use the envelope inside the
passband; above-LP keeps the stricter `Q_MAX_ABOVE_LP=2.5`. The LMA
cost term penalizes Q above `q_warn_at(freq)` instead of a flat Q>8
threshold so the gradient pressure matches the envelope. `Q_MAX=10.0`
stays as a fallback for spots without a frequency context. Frontend
mirrors `q_warn_at` in `lib/peq-quality.ts`; PEQ table shows ⚠
(popup-on-click only, no hover tooltip), graph dot gets a yellow
`#d97706` stroke when its band exceeds warn. Disabled bands are never
flagged. Old projects with Q=10 at 1 kHz keep their values and just
get the badge — nothing is rewritten on load.

## Notes

- Rust test suite: 140 (134 prior + 6 new q_envelope).
- Frontend tests: 105 passing, 5 skipped.
- Snapshot/index files use atomic `tmp+rename` writes.
- All round-trips (project save/load, b132 history, b133 named snapshots)
  carry the new fields; `#[serde(default)]` on every new optional field
  keeps b130 projects loadable.

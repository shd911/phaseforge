# Archive

Spent prompts and intermediate analysis docs from past sessions. Kept
for historical reference and prompt-effectiveness review. Not active
project documentation — see `docs/release-notes-*.md` and `CLAUDE.md`
for current state.

## Contents

### `b140.7-min-phase-iir/` (2026-05-09 → 2026-05-10)

14+ итераций fix цикла на min-phase FIR. От симптома "REW phase
mismatch на 44.1/48 kHz" через cepstral floor / post-shift / variant
B (analytical-phase-passthrough — fail) к финальному решению: IIR
analytical cascade + REPhase reference acceptance tests.

Включает: все `prompt-b140-7-*.md`, intermediate analysis
(`min-phase-*.md`, `iir-grid-audit-*.md`, `postshift-test-report.md`,
`variant-b-*.md`), и diagnostic prompts (`prompt-b140-low-bin-diag.md`,
`prompt-b140-minphase-*.md`, `prompt-b140-refine-diag-*.md`,
`prompt-b140-rollback-test.md`).

Outcome: см. `docs/release-notes-b140.8.md`.

### `prompts-pre-b140.7/`

Промты b131 → b140.6 (b138 subsonic-protect, b139 BandEvaluator
unification, b140.0–b140.6 Composite mode + SUM rebuild). Большая
часть — последовательная DSP миграция документированная в
release-notes b140.4 / b140.8.

### `specs-tz/`

Старые ТЗ-документы (project lifecycle, peq-stale, q-envelope,
subsonic-protect, measurement-analysis, total-rebuild, unified-eval).
Соответствующая функциональность в production. Хранятся для history
of design decisions.

## Why kept

Per CLAUDE.md "Prompt effectiveness tracking" — каждый промт и
ретроспектива дают input в pattern detection (cascade, dead-end
hypotheses, test-first vs UI-first acceptance). Удалять = терять
урок-сигнал для будущих сессий.

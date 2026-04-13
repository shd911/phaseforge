# Changelog

All notable changes to PhaseForge are documented in this file.

## b116 (2026-04-13)
### Fixed
- auto-align: alignment_delay now persists in saved projects (serde default)
- auto-align: geometric mean for XO center frequency (log-scale correct)
- auto-align: phase unwrapping after PEQ/XO complex sum (eliminates 360° corruption in cost function)
- auto-align: guard against degenerate frequency grids in interpolation
- FrequencyPlot: correct length guards in coherent sum (freq.length vs stale nPts)

### Notes
- b115 was a preliminary attempt; HP/LP swap in XO detection was reverted.

## [0.1.115] - 2026-04-13

### Changed
- Removed diagnostic console.log from auto-align and SUM phase compensation

## [0.1.114] - 2026-04-13

### Added
- Spacing design tokens (`--space-xxs` through `--space-3xl`) — unified padding/gap/margin across UI

### Changed
- Replaced 175 hardcoded spacing values in App.css with design tokens
- Replaced 8 inline style attributes in FrequencyPlot.tsx and ControlPanel.tsx

## [0.1.113] - 2026-04-13

### Fixed
- SUM view phase readability: frequency-dependent delay compensation using power-weighted average delay. Phase no longer wraps wildly after auto-align while magnitude sum remains coherent
- Eliminated phase display instability from IR peak detection on multi-way sums

## [0.1.109] - 2026-04-13

### Fixed
- Auto-align: interpolate all bands onto common frequency grid before cost function evaluation. Eliminates misalignment when bands have different freq grids from measurements

## [0.1.108] - 2026-04-13

### Changed
- Auto-align direction reversed: HF→LF sequential optimization (tweeter is reference, delay=0)
- Negative delay propagation: if lower band needs negative delay, propagate |delay| to upper bands as positive shift (keeps all delays ≥ 0)

## [0.1.107] - 2026-04-13

### Added
- SUM matrix: optimization status indicators (green dots next to band names showing PEQ state)
- Default phase visibility for Σ curves in legend
- Hover highlighting on SUM matrix rows and column headers

## [0.1.106] - 2026-04-13

### Changed
- Centralized curve colors into named constants (`PEQ_COLOR`, `CORRECTED_COLOR`, `STATUS_GOOD`, etc.) — eliminates 25+ hardcoded hex values
- Updated measurement color palette to brighter REW-style colors

## [0.1.105] - 2026-04-13

### Added
- Typography design tokens (`--fs-xxs` through `--fs-4xl`) — 10 size variables replacing 100+ hardcoded font-size values
- Unified button system with `.dlg-btn` variants (primary, danger, lg) + `.tb-btn-xs`/`.tb-btn-sm` size variants
- Spacing variables: `--radius-*`, `--btn-pad-*`

## [0.1.102] - 2026-04-12

### Added
- Auto-align delays feature: gradient descent optimization for coherent sum at crossover regions
- Precondition guards for auto-align (requires phase + XO config)

## [0.1.101] - 2026-04-12

### Added
- Per-band alignment delay + per-band phase in SUM view
- DELAY row on IR/Step tab in SUM view
- Mouse wheel control for delay inputs in SUM legend
- Hybrid mode

### Fixed
- IR/Step coherent sums with delay
- Legend checkbox rendering (transparent fill + checkmark)
- Adaptive Y-axis decimals
- Various IR/Step chart lifecycle issues

### Changed
- Split `peq/mod.rs` into submodules (types, biquad, lma, greedy)
- Extracted `evaluateBand` + `addGaussianMinPhase` from FrequencyPlot
- Replaced 14 `any` type annotations with proper types

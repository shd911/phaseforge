# Changelog

All notable changes to PhaseForge are documented in this file.

## b140.4 (2026-05-07) — Composite phase mode + unified SUM pipeline

Большой пакет архитектурных улучшений с предыдущего релиза b139.5.3.

### Composite phase mode для FIR экспорта (b140.1)

При включённом защитном subsonic фильтре + линейно-фазовом основном
Gaussian — экспорт работает в **составном режиме**: основной фильтр
сохраняет линейную фазу в полосе пропускания, защитный subsonic вносит
минимально-фазовую rotation только в инфразвуке. Это убрало проблему
из b138.x когда включение защитного делало весь FIR минимально-фазовым.

### Unified SUM pipeline (b140.3.x)

После каскада b140.2.x (11 итераций без паритета через копирование
legacy) — **полная перестройка с чистого листа** через `evaluateSum`.

- **Σ Target**: coherent sum target curves каждой полосы с polarity
  и alignment_delay.
- **Σ Corrected**: per-band normalize к собственному target в его
  passband, width-aware excess limiter (узкие резонансы сохраняются,
  широкие превышения > 0.1 dB лимитируются), global shift вниз при
  обнаружении wide bumps.
- **Σ Measurement**: extension через target shape для magnitude и
  Hilbert reconstruction для phase — корректное продление за пределы
  native freq диапазона полос.
- **SUM IR/Step**: через единый источник `evaluateSum.ir` на широкой
  freq grid (5 Гц – 40 кГц) — subsonic виден в impulse.

Переключатель **Legacy/New** на вкладке общей суммы — старый pipeline
сохранён для совместимости.

### Band IR/Step на широкой grid (b140.3.3, b140.3.4)

Target IR и Corrected IR для отдельной полосы строятся на standalone
grid 5–40 кГц (не на узком grid замера). Subsonic фильтр и другие
низкочастотные эффекты теперь видны в импульсе и step response.

### Extension в evaluateBandFull (b140.3.2)

Логика расширения measurement за пределы native freq диапазона
(через target shape + Hilbert phase) перенесена на per-band уровень.
На band view синтетическая часть отображается **бледным цветом** —
пользователь видит где реальный замер, где математическая модель.

### Полировка (b140.3.x)

- Crossover handles + drag-drop в New SUM (b140.3.1.6).
- Корректная связь impulse↔step после resample (cumsum derive вместо
  независимой интерполяции, b140.3.6).
- Стабильные labels для legend (без variable suffix), auto-align не
  сбрасывает visibility (b140.3.7).

### Под капотом

- Test harness fix для cargo iterative_refine convergence (thread-local
  vs Mutex pollution, b140.3.8).
- 179 cargo + 102 vitest тестов проходят.
- Глобальный аудит pipeline перед релизом (только legacy/UX inline
  invoke остались, все мигрировано).

### Известные ограничения

- Default SUM режим — Legacy. New работает но требует больше
  тестирования на разных проектах.
- TODO: sr=None при импорте .txt (с b135) — отдельная задача.

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

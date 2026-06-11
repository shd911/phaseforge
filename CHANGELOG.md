# Changelog

All notable changes to PhaseForge are documented in this file.

## b141.16 (2026-06-11) — единая WAV peak конвенция

- **Единая конвенция пика WAV** (b141.14): cepstral min-phase
  (Gaussian/Bessel/subsonic/custom) получил тот же адаптивный сдвиг
  `min(N/2, до хвоста −100 dB)`, что и IIR-путь — пик у N/2 на всех
  маршрутах, полосы синхронны в конвольвере. Кривые графика не
  сдвигаются (realized-анализ до сдвига). Генерация на 2N не
  понадобилась. Acceptance: `tests/wav_peak_convention.rs`.
- Toast о смешанных конвенциях (b141.8) удалён; вместо него точечное
  предупреждение при экспорте, если адаптивный сдвиг не дотянул пик до
  центра (хвост не уместился) — с величиной смещения в отсчётах.
- Метрика предзвона на вкладке Export считается от первого значимого
  отсчёта (−80 dB от пика): ведущие нули центрирования — латентность,
  не предзвон.
- **Fix** (b141.15): в ветке смешанной фазы с Gaussian-фильтрами
  центрирующий поворот не вычитался из realized-фазы (расхождение
  ~7700° с min-phase эталоном). Ветка недостижима из UI с b141.2 —
  закрыта тестом паритета фазы.
- 7-векторный аудит пакета: bugs/security/perf/data — чисто; стейл
  docstring и недокументированные решения (без taper, пороги
  −100/−80 dB, circular wrap) задокументированы.

## b141.13 (2026-06-11) — аудит-фиксы, кэш DSP, конвенции IR/Step, UI

Накопленный пакет с релиза b141.1: 8-векторный аудит и волны фиксов
(b141.2–b141.6), затем кэш, конвенции и UI (b141.7–b141.13).

### Корректность DSP
- **CRITICAL — PEQ sample rate** (b141.5): оптимизатор и график считали
  биквады всегда на 48 kHz; при экспорте 96k+ расхождение с WAV до
  3.5 dB на ВЧ. Везде прокинут реальный sample rate экспорта.
- **Wrap-aware интерполяция фазы** (b141.9): линейный лерп завёрнутой
  фазы поперёк ±180° травил бин у LR-кроссовера — узкий провал −3 dB и
  выброс фазы в Σ. Кратчайшая дуга во всех 6 местах пересэмпла.
- **Пост-LMA polish** (b141.8): после merge/retain — ре-оптимизация
  (overshoot до 3–4 dB устранён); Q-penalty перенесена в residuals
  (видна градиенту); convergence объявляется только на принятом шаге.
- **WAV IIR-пути** (b141.8): адаптивный сдвиг `min(N/2, до хвоста)` —
  на малых taps с НЧ/высоким Q фиксированный N/2 резал до −37 dB
  энергии хвоста (до 2.7 dB ошибки в WAV против графика).
- **GD**: Σ target/corrected в режиме «Сумма» (раньше не считались);
  фаза разворачивается перед дифференцированием в band-режиме —
  убраны выбросы в тысячи мс (b141.12/13).

### Конвенции IR/Step (b141.10)
- Σ Corrected взвешивает полосы как SPL-сумма: нормировка к таргету в
  пассбанде (живое исполнение — микрофон + аттенюация по полосам).
- **Delay: положительное = позже** — паритет с HQPlayer; единая точка
  конвенции, auto-align согласован, старые проекты мигрируются
  автоматически с сохранением относительных задержек.

### Производительность
- **Кэш BandEvalResult** (b141.7): display-тогглы (фаза/вкладка/SUM)
  больше не гоняют DSP-пересчёт — LRU по содержимому полосы.
- Инкрементальный LMA Jacobian ~20× (b141.6); все Tauri-команды
  async — UI не виснет (b141.5); −50% IPC payload FIR (b141.6).

### UI (b141.11–13)
- Легенды: штриховые кривые обозначены образцом линии (было —
  нечитаемые пунктирные квадратики), синхронизация GD-легенды,
  чипы курсора с лимитом «+N», крупнее AUTO, контраст muted-текста.
- Export скрыт в режиме «Сумма»; −570 строк мёртвых вкладок панели;
  русификация остатков; тултипы причин disabled-кнопок.
- **Зона предзвона**: частотно-зависимая — 2 периода нижнего
  linear-phase кроссовера (было 1.5 периода минимального HP проекта:
  20 Hz → ложные 75 ms); при отсутствии linear-phase — отключена;
  видимая граница зоны.
- Предупреждение при экспорте WAV со смешанными peak-конвенциями
  полос (IIR N/2 vs cepstral peak-at-0) — рассинхрон в конвольвере.

### Инфраструктура
- CSP восстановлен + общий гард путей записи (b141.6).
- Golden FIR baseline платформо-зависимый (libm macOS≠Linux), CI
  actions v5 (b141.13); release-readiness suite — стандартный гейт.
- Тесты: cargo 15 сьютов, vitest 220, hard_failures=0.

## b140.8 (2026-05-10) — IIR-based min-phase + REPhase parity

Полный список изменений: `docs/release-notes-b140.8.md`. Ключевое:

- **IIR-based Min-Phase FIR**: новый pipeline для не-Gaussian
  filters (LR/Butterworth/Custom + PEQ). Решает REW phase mismatch
  на 44.1/48 kHz WAV экспортах, который был регрессией b140.4–b140.6.
- **REPhase reference parity**: 4 автоматических теста сравнивают
  PhaseForge IIR output vs REPhase reference WAVs. Worst case
  max Δmag 0.44 dB / Δphase 2.5° (sr=44.1k); best case 0.03 dB /
  0.2° (sr=176.4k).
- **UI: Slope dropdown** вместо Order numeric. LR4 теперь корректно
  отображается как `48 dB/oct` (было `LR24`).
- **FIR grid extensions** (b140.5/b140.6): noise-floor tail до
  Nyquist + resample realized_mag/phase на eval grid — закрывают
  display регрессии на 44.1/48 kHz.
- 194 cargo + 104 vitest = 298 автоматических тестов PASS.

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

# Variant (B) Plan: analytical phase pipeline

## Текущая архитектура (b140.6, code-cited)

### TS строит analytical phase

`src/lib/band-evaluator.ts:122-153` `reconstructTargetPhase(freq, basePhase, hp, lp)`:

- `basePhase` приходит из IPC `evaluate_target` (`src-tauri/src/target/mod.rs:77-113`,
  `target::evaluate`). Для **не-Gaussian** фильтров (LR4 / Butterworth / Bessel / Custom)
  `target::evaluate` возвращает **аналитическую** фазу через `apply_filter`
  (`target/mod.rs:198-218`) — `if cfg.linear_phase { return; }`, иначе
  `filter_lp_response` / `filter_hp_response` (analytical p_deg). Для Gaussian
  фаза = 0 (`target/mod.rs:202-205`).
- `reconstructTargetPhase` ДОБАВЛЯЕТ к `basePhase`:
  - `isGaussianMinPhase(hp)` → Hilbert(`gaussianFilterMagDb` + `subsonicMagDb` если активен): `band-evaluator.ts:130-137`. Покрывает Gaussian min-phase HP вместе с subsonic.
  - `else if hasActiveSubsonicProtect(hp) && hp.linear_phase===true` → Hilbert(subsonic only): `band-evaluator.ts:138-143`. Покрывает linear-phase Gaussian HP + subsonic.
  - `isGaussianMinPhase(lp)` → Hilbert(gaussianFilterMagDb): `band-evaluator.ts:146-150`.

Subsonic protect — Gaussian-only по design (`target/mod.rs:163-166`:
`subsonic_active = !is_lowpass && Gaussian && subsonic_protect && fc > 40`),
поэтому subsonic phase coverage в TS pipeline **полная** для всех конфигов
где subsonic вообще активен.

### TS строит firCombinedPhase

`src/lib/band-evaluator.ts:373-404`:

- `firFreqRaw` (log grid 5..fMaxFir, 512 pts): `evaluate_target_standalone` IPC,
  line 375-378.
- `firTargetPhaseRaw = reconstructTargetPhase(firFreqRaw, firResp.phase, hp, lp)` — line 379-381.
- `appendNoiseFloorTail` extends grid → `firFreq` (≈544 pts, 5 Hz..Nyquist·0.999),
  phase tail = 0: line 386-392.
- `firPeqMag/firPeqPhase` от `compute_peq_complex(freq=firFreq, bands)` — line 397-403.
  PEQ — biquads, фаза analytical (Rust DSP).
- `firCombinedPhase = firTargetPhase[i] + firPeqPhase[i]`: line 404. **Log grid,
  длина = firFreq.length.**

### Что передаётся в Rust

`src/lib/band-evaluator.ts:406-430`:

- `freq: firFreq` (log grid).
- `targetMag: firTargetMag`.
- `peqMag: firPeqMag`.
- `modelPhase: firCombinedPhase` (log grid, target+peq+subsonic+Gaussian min-phase).

### Rust в Composite + min-phase main path игнорирует modelPhase

`src-tauri/src/fir/mod.rs:564-579` — `effective_linear = linear_phase_main`
для Composite.

`src-tauri/src/fir/mod.rs:565`: `let max_phase_abs = model_phase.iter()...` —
**model_phase используется только для логирования max abs**, нигде дальше
не входит в путь к IFFT.

`src-tauri/src/fir/mod.rs:593-610`: для Composite — `target_phase_rad =
compose_target_phase(...)` (`fir/helpers.rs:501-522` → `composite_phase_inner`
455-495). Это **повторно** вычисляет фазу через cepstral Hilbert трёх
независимых источников (main + peq + subsonic) на линейном FFT гриде —
**полностью игнорируя переданный modelPhase**.

`src-tauri/src/fir/mod.rs:642-648`: для Composite — `peq_phase_rad =
vec![0; n_bins]` (не пересчитывается, т.к. peq уже учтён в composite).

`src-tauri/src/fir/mod.rs:650-653`: `phase_rad = target_phase_rad +
peq_phase_rad` — фаза для IFFT. **modelPhase сюда не входит.**

`src-tauri/src/fir/mod.rs:692-702`: `iterative_refine(..., &phase_rad, ...)`.
В iter loop (`fir/helpers.rs:132`, `135`, `142`, `194-217`) для Composite
вызывается `composite_phase_inner` каждую итерацию через флаг
`recompute_composite` — фаза **пересчитывается** на обновлённой `refined_db`.

**Вывод**: переданный `modelPhase` фактически — dead code в Composite пути.
Variant (B) заменяет cepstral путь на passthrough TS-вычисленной фазы.

## Целевая архитектура

### TS

1. `reconstructTargetPhase` уже даёт корректную analytical фазу для всех
   filter types — **не менять**.
2. PEQ phase из `compute_peq_complex` уже корректна — **не менять**.
3. Resample log→linear FFT grid выполняется в TS (один раз, со всеми
   нюансами unwrap'а):
   - Unwrap `firCombinedPhase` (continuous, без 360° скачков).
   - Linear interp на n_bins = `cfg.taps / 2 + 1` точек linear grid.
   - Re-wrap в `[-π, π]`.
4. Передавать в Rust **уже на linear FFT grid**: новое поле IPC
   `modelPhaseLinear: number[]` длины `n_bins` (отдельно от существующего
   `modelPhase` чтобы не ломать legacy paths).

### Rust

5. Новый config флаг `use_provided_phase: bool` ортогональный
   `phase_mode`. Когда `true`:
   - В `generate_model_fir`: `target_phase_rad = model_phase_linear.to_vec()`
     (без вызова `compose_target_phase`).
   - `peq_phase_rad = vec![0; n_bins]` (peq уже в model_phase_linear).
6. В `iterative_refine`:
   - Игнорировать `recompute_composite` когда `use_provided_phase=true`.
   - `iter_phase` остаётся равной passed phase — magnitude-only refinement.
7. Windowing path (linear/half) и `effective_linear` остаются как есть —
   зависят от `linear_phase_main`, не от phase source.

## Изменения по файлам

### 1. `src/lib/band-evaluator.ts`

- В блоке b140.5 после `appendNoiseFloorTail` (line 386-392) и сборки
  `firCombinedPhase` (line 404) — добавить resample на linear FFT grid
  длины `cfg.taps / 2 + 1` (n_bins). Алгоритм:
  - `unwrap(firCombinedPhase)` — линейная разворачивающая последовательность.
  - Linear interp в log-freq на n_bins точек где `f_k = k * sr / n_fft`.
  - DC bin (k=0): экстраполяция или `phase[0] = phase[k=1]` (TS unwrapped).
  - Re-wrap результата в `[-π, π]`.
- Добавить новое поле IPC `modelPhaseLinear: number[]`.
- Передавать `use_provided_phase: true` в config.

### 2. `src-tauri/src/fir/types.rs` (или где живёт `FirConfig`)

- Добавить поле `pub use_provided_phase: bool` (default `false` для
  обратной совместимости).
- Добавить параметр `model_phase_linear: Option<Vec<f64>>` в сигнатуру
  `generate_model_fir`. Tauri command — соответственно.

### 3. `src-tauri/src/fir/mod.rs`

- `generate_model_fir`, line 593-638 (выбор `target_phase_rad`):
  ```rust
  let target_phase_rad = if config.use_provided_phase {
      let p = model_phase_linear
          .expect("use_provided_phase=true requires model_phase_linear");
      assert_eq!(p.len(), n_bins);
      p
  } else if config.phase_mode == PhaseMode::Composite {
      crate::fir::helpers::compose_target_phase(...)  // existing
  } else if effective_linear { ... }
  ...
  ```
- Line 642-648 (peq_phase_rad): when `use_provided_phase` →
  `vec![0; n_bins]` (PEQ уже в model_phase_linear).

### 4. `src-tauri/src/fir/helpers.rs`

- `iterative_refine`, line 120-128 в районе флагов `recompute_*`:
  ```rust
  let recompute_composite = matches!(config.phase_mode, PhaseMode::Composite)
      && !config.use_provided_phase;
  ```
  И аналогично для `recompute_min_phase` если применимо.
- Line 142-145 (iter_phase init): когда `use_provided_phase`, `iter_phase
  = phase_rad.to_vec()` сразу.
- Line 190-217 (per-iter recompute): пропустить блок когда
  `use_provided_phase=true`.

### 5. Тесты

#### Cargo

- **Новый тест** `min_phase_impulse_peaks_at_zero_for_sparse_spectrum_variant_b`:
  build sparse magnitude (LP=200 / BP / HP=2000), вычислить analytical
  phase test-side (Butterworth response), вызвать `generate_model_fir`
  с `use_provided_phase=true`. Acceptance: peak idx ≤ 5 на всех трёх
  кейсах.
- **Регрессия golden snapshots**: `generate_fir_b139_golden_lr4_baseline_impulse_hash`
  и аналоги (`generate_fir_b139_3_*`, `fir_composite_*`) — при
  `use_provided_phase=false` все остаются bit-exact (default false →
  default путь не меняется).
- **Параллельный** `test_fir_magnitude_matches_target_all_phase_modes` с
  `use_provided_phase=true` — realized_mag в passband ≤ 1 dB RMS / ≤ 3 dB peak.
- `iterative_refine_converges_with_min_phase_subsonic` — параллельный с
  `use_provided_phase=true` ожидаем PASS без расхождения (фаза static).

#### Vitest

- Новый тест `evaluateBandFull` FIR path: проверить что
  `modelPhaseLinear.length === n_bins`, что unwrap устойчив на boundary
  (известный синусоидальный сигнал с linear ramp phase).

## Golden snapshots для regression check

Capture текущего b140.6 поведения ДО refactor для bit-exact compare:

| Fixture | sr | HP | LP | linear_phase | subsonic | Что хешировать |
|---|---|---|---|---|---|---|
| F1: LP-only | 48000 | – | LR4 200 | false | – | impulse[0..1024] hash, realized_mag hash, realized_phase hash |
| F2: BP | 48000 | LR4 200 | LR4 2000 | false | – | то же |
| F3: HP-only | 48000 | LR4 2000 | – | false | – | то же |
| F4: F1 @ 176.4k | 176400 | – | LR4 200 | false | – | то же |
| F5: F2 @ 176.4k | 176400 | LR4 200 | LR4 2000 | false | – | то же |
| F6: F3 @ 176.4k | 176400 | LR4 2000 | – | false | – | то же |
| F7: Gaussian min + subsonic | 48000 | Gaussian 632 (M=1, sub on) | – | false | on | то же |
| F8: Gaussian linear + subsonic | 48000 | Gaussian 632 (M=1, sub on) | – | true | on | то же |

Storage: `src-tauri/tests/golden_b140_6/` снапшоты (Rust constants или
JSON). После variant (B) — new snapshots с `use_provided_phase=true`
сохраняются параллельно, не заменяя b140.6. Diff документируется в
implementation report.

## Risk analysis

1. **Refinement convergence без phase update**. b140.6 уже имеет
   расходимость на Band 3 (`iterative_refine_converges_with_min_phase_subsonic`
   FAIL после composite-split в b140.1). При фиксированной phase
   расходимость должна исчезнуть (нет phase-mag перемешивания), но если
   target_mag сам по себе вылетает из коридора — damped error correction
   может расходиться. **Mitigation**: max_err threshold check после iter 1,
   early-exit если `max_err < 0.05 dB` (уже есть в `helpers.rs:180-183`).

2. **Phase resample log→linear unwrap**. Linear FFT bins начинаются от
   DC (k=0, f=0). Log grid не содержит f=0 → нужна экстраполяция (или
   `phase[0] = phase[k=1]`). Boundary at high freq: log grid после b140.5
   tail доходит до Nyquist·0.999 (`appendNoiseFloorTail` фиксирует
   phase=0 на tail), unwrap должен быть стабилен. **Mitigation**: unit
   test что resample(log) ≈ resample(linear) на тестовых сигналах с
   известной фазой (linear phase ramp, min-phase Butterworth).

3. **Subsonic phase coverage**. TS `reconstructTargetPhase`
   (`band-evaluator.ts:130-150`) покрывает Gaussian HP min-phase, Gaussian
   linear + subsonic, Gaussian LP min-phase. Не-Gaussian filters получают
   фазу из `target::evaluate` (analytical). `subsonic_protect` в Rust —
   только для Gaussian (`target/mod.rs:163-166`), поэтому non-Gaussian +
   subsonic невозможен по design. **Vacuously covered.**

4. **Composite + linear_phase_main=true**. После variant (B) этот путь
   тоже использует passed phase. TS `firCombinedPhase` для linear-phase
   main = только subsonic Hilbert (если subsonic включён) + peq Hilbert.
   Без subsonic и peq — phase ≈ 0 везде. Ожидаемо, не ломает linear-phase
   FIR (центрированный sinc-like impulse).

5. **Backward compatibility**. `use_provided_phase=false` по default
   сохраняет старый путь. Если внешние caller'ы (CI fixtures, scripts,
   `lib.rs` IPC `compute_minimum_phase`) полагаются на текущее поведение
   — не сломается. Только TS `evaluateBandFull` начинает посылать `true`.

6. **n_bins vs taps mismatch**. `n_bins = taps/2 + 1`. TS должен знать
   `taps` чтобы построить linear grid правильного размера. Берётся из
   `cfg.taps` который уже есть в TS — без новых connections.

## Acceptance после implementation

- `Pre-ring=0.00 ms`, `Causal ≥99%` на всех трёх production полосах
  (LP=200, BP 200-2000, HP=2000) при sr=48k и sr=176.4k.
- На Export plot: Model° кривая совпадает с FIR° в passband ≤ 0.5° RMS
  и ≤ 2° peak.
- В REW: импортированный экспортированный WAV — фаза совпадает с
  моделью; constant group delay (linear-phase rotation) отсутствует.
- iterative_refine на Band 3 (HP=2000 sr=48k): `max_err` не растёт
  iter→iter (early-exit при `< 0.05 dB`).
- Cargo: ≥ 179 PASS (b140.6 baseline) + новые tests на variant (B). При
  `use_provided_phase=false` все существующие тесты bit-exact.
- Vitest: 104 PASS (b140.6 baseline) + новые tests на TS-side resample.
- Golden snapshots: при `use_provided_phase=false` bit-exact match с
  pre-refactor b140.6. При `use_provided_phase=true` сохранены новые
  snapshots; diff документирован в report.

---

**STOP — план готов, ждём ревью user-а перед implementation.**

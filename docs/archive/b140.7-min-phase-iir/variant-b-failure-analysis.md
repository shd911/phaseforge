# Variant (B) failure — analytical phase pipeline (rolled back)

## Что было сделано

b140.7 implementation (откачено через `git checkout`):

- TS `evaluateBandFull` строит `firCombinedPhase` (analytical from
  `reconstructTargetPhase` + PEQ Hilbert) на log grid 5..fMaxFir.
- TS resamples log → linear FFT grid: `unwrap → interp(log→linear) →
  wrap`, DC = 0, длина = `taps/2 + 1`.
- IPC payload: новое поле `modelPhaseLinear`, `config.use_provided_phase
  = true`.
- Rust `FirConfig.use_provided_phase: bool` (default `false` для
  backward-compat existing cargo tests).
- Rust `generate_model_fir` принимает `model_phase_linear:
  Option<&[f64]>`, при `use_provided_phase=true` использует его как
  `target_phase_rad` напрямую, минуя `compose_target_phase` cepstral.
- Rust `iterative_refine`: `recompute_min_phase` / `recompute_composite`
  устанавливаются в `false` когда `use_provided_phase=true` —
  magnitude-only refinement без пересчёта phase.
- 21 caller-site `generate_model_fir(...)` обновлены вставкой `None,`
  argument; 14 `FirConfig {…}` literals — добавлен
  `use_provided_phase: false`.

## Тестовый результат на production (sr=48k, Band 1 LP=200 LR4)

Согласно user evidence (UI verify b140.7):

- Pre-ring: **243.65 ms** (b140.6 baseline: 4.25 ms — **REGRESS**).
- Causal: **46 %** (b140.6: 51 %, нет улучшения).
- Mag err: **18.69 dB** (b140.6: 0.01 dB — **BROKEN passband**).
- GD ripple: **562.53 ms**.
- Phase warp визуально огромный.

In-Rust unit test `min_phase_impulse_peaks_at_zero_variant_b` тоже
FAIL: peak idx 227 для LP=200 LR4 (требовал ≤ 5).

## Math root cause

Для discrete IFFT(`mag · exp(j·phase)`) → real-valued *causal* impulse
с peak-at-0 необходимо: **phase = Hilbert(log_mag) на ТОЙ ЖЕ
linear FFT grid**. Любая другая фаза, даже формально "min-phase" в
другом домене, даёт complex impulse → real part теряет imaginary
energy → magnitude corrupts на round-trip.

Analog continuous phase, возвращаемая `target::evaluate` (LR4 = 2 ×
Butterworth_phase из rational TF в s-плоскости), **≠** discrete
Hilbert(log_mag) на uniform-bin DFT grid. Они совпадают в continuous
limit, но при фиксированной discrete sampling дают разный impulse.
Ресэмпл log → linear с unwrap/wrap не превращает analog phase в
discrete-Hilbert-equivalent — это математический факт, не
implementation bug.

В Rust unit test wrap не должен влиять (cos/sin 2π-периодичны), и
действительно peak idx 227 близок к peak idx 204 от cepstral пути в
b140.6 — оба артефакта одного происхождения.

## Урок

SPL phase, которую `reconstructTargetPhase` строит — это **analog**
phase, plotted as is для дисплея. Визуальная корректность на
SPL-вкладке не значит, что её можно использовать как генератор impulse
через DFT IFFT.

Для FFT-based FIR generation cepstral на linear grid — **обязателен**;
вопрос только в смягчении его artifacts (lifter / floor / smoothing).

## Альтернативы для будущей работы

1. **Lifter (cepstrum windowing)**: умножать `c_min[n]` на гладкое
   окно вроде `1/(1+(n/n_lifter)^2)` перед обратной FFT в
   `minimum_phase_from_magnitude`. Подавляет high-quefrency leak от
   sharp magnitude transitions. Минимально инвазивно.

2. **Pre-smooth `lin_mag`** (1/12 oct fractional smoothing) только
   перед cepstrum step. `assemble_complex_spectrum` использует
   оригинальный `lin_mag` — magnitude precision не страдает, blurs
   только cliff в cepstrum input.

3. **Analog IIR → bilinear transform → truncated FIR** (REPhase
   архитектура). Большой refactor: заменяет FFT-based FIR generation
   на IIR-based time-domain. Гарантирует causality по построению.

4. **Time-domain min-phase reconstruction** через ARMA spectral
   factorization. Тоже большой refactor.

Нет приоритета без дополнительной диагностики и обсуждения.

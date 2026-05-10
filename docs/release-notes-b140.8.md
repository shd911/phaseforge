# PhaseForge b140.8 — IIR-based min-phase + REPhase parity

Большой блок DSP / UX улучшений с предыдущего релиза b140.4.

## IIR-based Min-Phase FIR pipeline (b140.7)

Новый pipeline для не-Gaussian Min-Phase user choice: analytical
filter design → bilinear transform → digital biquad cascade →
truncated FIR. Peak-at-0 by construction для Linkwitz-Riley /
Butterworth / Custom HP+LP и PEQ peaking biquads. Linear-Phase /
Composite + subsonic / Gaussian / Bessel / Custom measured targets —
остаются на старом FFT cepstral path.

Решает регрессию b140.4–b140.6: REW phase mismatch на min-phase WAV
экспорт, особенно для sr=44.1/48 kHz, где cepstral на sparse linear
FFT grid создавал constant group-delay artefact.

Под капотом:
- `iir_path.rs` модуль (~830 lines): biquad + bilinear math, Q tables
  для Butterworth, RBJ peaking/shelf формулы, cascade impulse
  generator.
- Новый Tauri command `generate_model_fir_iir`.
- Routing в `band-evaluator.ts` автоматически выбирает path по
  filter type / phase mode / subsonic.
- Plot data computed from raw cascade FFT (analytical truth, no
  phase corrections); WAV impulse padded with N/2 leading zeros for
  REW peak-detection compatibility.

## REPhase reference compatibility (b140.7.12)

4 cargo тестов сравнивают PhaseForge IIR output против REPhase
reference WAVs (`test-fixtures/rephase/`, gitignored).

| sr | max Δmag | max Δphase |
|---|---|---|
| 44 100 Hz | 0.44 dB | 2.5° |
| 48 000 Hz | 0.37 dB | 2.1° |
| 88 200 Hz | 0.11 dB | 0.6° |
| 176 400 Hz | 0.03 dB | 0.2° |

Tolerance: max Δmag < 1 dB, max Δphase < 10° в passband. PhaseForge
матчит REPhase реализацию HP=2000 LR8 min-phase, taps=65536, Hann
window, Float64 mono — на всех production sample rates.

Тесты skip cleanly when fixtures absent (CI / clean checkouts).

## UI: Order → Slope dropdown (b140.7.13)

HP/LP filter UI теперь показывает `Slope: X dB/oct` dropdown вместо
`Order: N` numeric. Значения по типу:

- Linkwitz-Riley: 12, 24, 36, 48, 60, 72, 84, 96 dB/oct (orders 1–8 × 12).
- Butterworth / Bessel / Custom: 6, 12, 18, 24, 30, 36, 42, 48 dB/oct
  (orders 1–8 × 6).
- Gaussian: использует `M shape` параметр, slope dropdown скрыт.

Save format JSON не меняется (`filter.order` персистентно). Existing
projects auto-mapped при load.

**Visible change для существующих проектов:** статусные labels теперь
показывают `LR48` для `order=4` (было `LR24`). Это исправление
давнего display-бага: PhaseForge LR convention внутренне реализует
`(Butterworth-N)² = 12N dB/oct`, что соответствует фактическому
DSP slope.

## FIR grid resample (b140.6)

`realized_mag/phase` resampled на eval freq grid внутри
`evaluateBandFull`. Решает sr-dependent rolloff shift на 44.1/48 kHz
WAV экспортах.

## FIR grid extension (b140.5)

FIR grid extended до Nyquist с noise-floor tail. Защищает от Rust
constant-clamp на linear FFT bins выше fMaxFir (был сдвиг ½ октавы
на rolloff на низких sample rates).

## Под капотом

- 194 cargo тестов (180 lib + 4 REPhase + 10 integration) PASS.
- 104 vitest PASS.
- `[DIAG ACTIVE]` startup marker convention для diagnostic patches.
- Versioning rule: каждый промт / fix bumps version (`b140.7 →
  b140.7.1 → b140.7.14`) для visual confirmation в title bar.

## Известные ограничения

- IIR path не покрывает: Gaussian (использует FFT cepstral),
  Composite + subsonic protect (legacy FFT path), Custom measured
  targets с импортированными измерениями.
- LP min-phase impulse имеет natural group delay (~5 ms на LP=200
  sr=48 k для 8-го порядка) — это физическая характеристика
  фильтра, не bug. UI Pre-ring/Causal метрики отражают эту
  естественную задержку.
- Bilinear discretization vs analog reference имеет небольшое
  frequency-dependent отклонение (≤ 2.5° в passband для LR8 vs
  REPhase). REPhase tests дают авторитетную acceptance — внутренний
  guardrail test толерантен до 25°.

## Migration

Существующие `.pfproj` файлы загружаются без изменений. UI
переключение на Slope dropdown прозрачно. Default phase mode не
меняется.

## Acceptance

- 4/4 REPhase reference tests PASS.
- 194 cargo + 104 vitest tests PASS.
- Manual REW workflow: HP=2000 LR8 sr=48 k Min-Phase Export → REW
  phase matches model.

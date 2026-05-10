# Промт для Code: b139.4a — составной режим FIR в Rust

**Тип:** новый режим в Rust + расширение BandEvaluator. Без UI миграции
(b139.4b сделает её отдельно). Bump до 0.1.0-b139.4a.

## Контекст

Текущее поведение b138.4: при включённом subsonic_protect генератор
FIR auto-demote в MinimumPhase mode для всего FIR, включая основной
фильтр. Это нарушает выбор пользователя по чекбоксу linear_phase.

Цель: новый режим экспорта который **уважает** linear_phase
основного фильтра (Gaussian), но subsonic_protect ВСЕГДА
минимально-фазовый, наложен поверх.

Подход: phase composition — split target magnitude на base part и
subsonic part, реконструировать phase для каждой части по своим
правилам, сложить.

## Acceptance матрица

| linear_phase | subsonic | Фаза в полосе пропускания | Фаза в зоне subsonic (5–40 Hz) |
|---|---|---|---|
| true  | OFF | 0 | 0 |
| true  | ON  | 0 | min-phase Butterworth (≈ -720°) |
| false | OFF | min-phase Gaussian | ≈ 0 (за rolloff Gaussian) |
| false | ON  | min-phase Gaussian | min-phase Gaussian + Butterworth |

Каждая клетка — отдельный cargo тест.

## Что нужно сделать

### 1. Новый PhaseMode в Rust

В `src-tauri/src/fir/types.rs` (или mod.rs) добавить:

```rust
pub enum PhaseMode {
    LinearPhase,
    MinimumPhase,
    Hybrid,           // существующий
    Composite,        // НОВЫЙ: linear-phase main + min-phase subsonic
}
```

### 2. Реализация Composite режима

В `src-tauri/src/fir/mod.rs` или helpers.rs — функция которая
получает на вход:
- `freq` (grid)
- `base_mag_db` — magnitude без subsonic (Gaussian + LP + tilt + etc)
- `subsonic_mag_db` — только subsonic attenuation (0 dB above cutoff,
  Butterworth-8 rolloff below). 0 если subsonic выключен.
- `linear_phase_main: bool` — выбор пользователя по чекбоксу.
- остальные FIR параметры (taps, sr, window, и т.д.)

Логика:

```rust
fn compose_phase_response(
    freq: &[f64],
    base_mag_db: &[f64],
    subsonic_mag_db: &[f64],
    linear_phase_main: bool,
    n_fft: usize,
) -> Vec<f64> {
    // 1. Phase основного фильтра
    let base_phase = if linear_phase_main {
        vec![0.0; freq.len()]
    } else {
        minimum_phase_from_magnitude(base_mag_db, n_fft)
    };

    // 2. Phase subsonic — всегда min-phase, ТОЛЬКО от subsonic part
    let subsonic_phase = minimum_phase_from_magnitude(subsonic_mag_db, n_fft);

    // 3. Сумма
    base_phase.iter().zip(subsonic_phase.iter())
        .map(|(b, s)| b + s)
        .collect()
}
```

`generate_model_fir` в Composite режиме:
1. Вычисляет составную phase через `compose_phase_response`.
2. total_mag = base_mag + subsonic_mag.
3. Собирает complex spectrum, IFFT, window → impulse.
4. **iterative_refine** в Composite режиме: при пересчёте phase на
   каждой итерации — снова вызывает `compose_phase_response`
   (иначе divergence как в b139.3.4).

### 3. Сигнатура: как передать subsonic_mag_db

Новые поля в `FirConfig`:

```rust
pub struct FirConfig {
    // ...existing
    pub linear_phase_main: bool,           // НОВОЕ: чекбокс пользователя
    pub subsonic_cutoff_hz: Option<f64>,   // НОВОЕ: Some(freq/8) если subsonic ON, None иначе
}
```

`generate_model_fir` использует `subsonic_cutoff_hz`:
- Если `Some(fc)` → внутри функции вычисляет `subsonic_mag_db` через
  Butterworth-8 формулу (та же что в `target/mod.rs:apply_filter`):
  `(f_sub/f)^16` → magnitude.
- target_mag (приходит от caller) ВКЛЮЧАЕТ subsonic attenuation
  (target evaluation уже применила subsonic).
- Чтобы получить base_mag_db, нужно **вычесть** subsonic_mag_db из
  target_mag: `base_mag_db = target_mag - subsonic_mag_db`.

Это требование: caller передаёт `target_mag` с учётом subsonic, и
указывает `subsonic_cutoff_hz` чтобы Rust мог разделить.

Альтернатива: caller передаёт base_mag и subsonic отдельно. Менее
эргономично, больше изменений API.

Выбор подхода — на усмотрение Code в зависимости от существующих
сигнатур. Главное: внутри `generate_model_fir` две magnitude должны
быть доступны раздельно для phase composition.

### 4. Расширение BandEvaluator (frontend)

В `src/lib/band-evaluator.ts`, в FIR ветке:

```typescript
const hp = band.target.high_pass;
const isLinearMain = hp?.linear_phase === true;
const subsonicActive = hasActiveSubsonicProtect(hp);
const subsonicCutoff = subsonicActive ? hp!.freq_hz / 8 : null;

const phaseMode = subsonicActive ? "Composite" : (isLinearMain ? "LinearPhase" : "MinimumPhase");

const fir = await invoke("generate_model_fir", {
  freq, targetMag, peqMag, modelPhase: combinedTargetPhase,
  config: {
    // ...
    phase_mode: phaseMode,
    linear_phase_main: isLinearMain,
    subsonic_cutoff_hz: subsonicCutoff,
  },
});
```

Удалить старую логику `isLin` / demotion.

### 5. Cargo тесты — 4 случая acceptance матрицы

```rust
#[test]
fn fir_composite_lin_main_no_subsonic() {
    // linear=true, subsonic=OFF → phase = 0 везде → FIR symmetric (linear-phase)
}

#[test]
fn fir_composite_lin_main_with_subsonic() {
    // linear=true, subsonic=ON → phase = 0 в полосе, min-phase в инфразвуке
    // Проверить через FFT(impulse): phase[freq=1000Hz] ≈ 0,
    // phase[freq=10Hz] ≠ 0 и соответствует min-phase Butterworth-8.
}

#[test]
fn fir_composite_min_main_no_subsonic() {
    // linear=false, subsonic=OFF → phase = min-phase Gaussian
}

#[test]
fn fir_composite_min_main_with_subsonic() {
    // linear=false, subsonic=ON → phase = min-phase Gaussian + min-phase subsonic
}
```

Метрики на phase:
- Phase в полосе пропускания (1k–10k Hz): для linear=true → ≤ 0.5° abs.
- Phase в зоне subsonic (5–40 Hz): для subsonic ON → ≥ 100° rotation
  накопительно (для 8th order сумма ≈ 720°).
- Magnitude всегда совпадает с target в пределах 0.5 dB после
  iterative_refine.

### 6. Регрессия

Все existing 158 cargo тестов — PASS. Особенно:
- Golden hash `3a56a4dab45f0fb1` без drift.
- `iterative_refine_converges_with_min_phase_subsonic` — PASS
  (сценарий b139.3.4).
- `fir_identity_for_flat_input_no_filters` — PASS.

Vitest 136 — PASS. BandEvaluator расширение покрыть тестом что новый
`phaseMode` корректно прокидывается в invoke.

### 7. Что НЕ делать в этом этапе

- Не мигрировать FrequencyPlot.tsx (это b139.4b).
- Не удалять старый код (ни `addGaussianMinPhase`, ни inline
  pipeline). Они ещё используются.
- Не трогать существующие LinearPhase / MinimumPhase / Hybrid режимы
  (только добавить Composite).

### 8. Bump

- `src-tauri/tauri.conf.json` → b139.4a.
- `src-tauri/src/lib.rs` startup-лог.
- skill `build-version`.

## Что прислать обратно

```
cargo test:
  fir_composite_lin_main_no_subsonic: PASS
  fir_composite_lin_main_with_subsonic: PASS
  fir_composite_min_main_no_subsonic: PASS
  fir_composite_min_main_with_subsonic: PASS
  iterative_refine_converges_with_min_phase_subsonic: PASS (regression)
  generate_fir_b139_golden_lr4_baseline_impulse_hash: PASS
  все existing: PASS

vitest: PASS count
```

При FAIL — eprintln output, без слепых правок.

## Правила

- Один коммит: `feat: composite phase mode for FIR (b139.4a)`
  + Co-Authored-By.
- 7-vector review.
- Без нарратива.

# Промт для Code: b140.1 — расщепление Composite на 3 источника фазы

ТЗ целиком: `docs/TZ-b140-total-rebuild.md`.
Текущий билд: 0.1.0-b140.0 → bump до 0.1.0-b140.1.

## Контекст

Пользователь импортирует экспортированный WAV в REW и видит **плоскую
фазу везде**, даже когда в проекте есть PEQ полосы (биквады, физически
min-phase). На вкладке экспорта PhaseForge phase отображается
правильно, но в импульсе она теряется.

Корень: `composite_phase_inner` в Rust получает **только**
`total_mag` (target+peq+subsonic) и `subsonic_mag`. Реконструирует:
- `base_mag = total_mag - subsonic_mag` (вычитание)
- `base_phase = if linear { 0 } else { Hilbert(base_mag) }`
- `subsonic_phase = Hilbert(subsonic_mag)`
- `total_phase = base_phase + subsonic_phase`

При `linear_phase_main = true` `base_phase = 0` — это правильно для
основного фильтра, но **PEQ contribution тоже находится в base_mag**
и его фаза тоже обнуляется. PEQ должен **всегда** иметь min-phase
независимо от выбора пользователя для main filter.

## E2E тест — pre-fix расширение

E2E b140.0 проверка показала `linear_no_subsonic_with_peq` как PASS
с phase_err=0.03°. Это значит **expected_phase функция тоже** теряет
PEQ phase. Тест self-consistent (real==expected==0), но оба
неправильны.

### Шаг 0: усилить acceptance — assert PEQ rotation

Перед фиксом расширить тест в `src-tauri/tests/e2e_export.rs`:

```rust
/// Дополнительный тест: при наличии PEQ полос с Q≥2 и gain≥3 dB
/// **realized phase** на peak frequency полосы должна иметь rotation
/// ≥ 20° (для значимых PEQ). Это ловит «PEQ phase teryaetsya».
#[test]
fn e2e_peq_phase_present_in_realized() {
    let configs: Vec<&ExportConfig> = acceptance_configs().iter()
        .filter(|c| !c.peq_bands.is_empty())
        .collect();

    for cfg in configs {
        let impulse = run_export_pipeline(cfg, 48000.0);
        let (real_mag, real_phase) = realized_response_compensated(&impulse, ..., cfg.linear_phase_main);

        // Для каждой PEQ полосы — phase rotation на peak частоте
        // должна быть значимой (Hilbert от gain dB полосы):
        for band in &cfg.peq_bands {
            let idx = freq.iter().position(|&f| f >= band.freq_hz).unwrap_or(0);
            // Окно ±0.5 октавы вокруг полосы
            let lo = (idx.saturating_sub(20)).max(0);
            let hi = (idx + 20).min(real_phase.len() - 1);
            let phase_range = real_phase[lo..=hi].iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), &p| (mn.min(p), mx.max(p)));
            let rotation_span = phase_range.1 - phase_range.0;

            assert!(rotation_span > 20.0,
                "{}: PEQ band at {} Hz expected rotation ≥20°, got {:.2}°",
                cfg.name, band.freq_hz, rotation_span);
        }
    }
}
```

При запуске на текущем коде этот тест должен **FAIL** для всех
linear_phase_main=true configs c PEQ — это и есть локализация бага.

## Шаг 1: фикс Rust — три источника phase

### Изменения в `src-tauri/src/fir/helpers.rs:composite_phase_inner`

Добавить параметр `peq_mag_db: &[f64]`:

```rust
pub(crate) fn composite_phase_inner(
    total_mag_db: &[f64],
    subsonic_mag_db: &[f64],
    peq_mag_db: &[f64],         // НОВЫЙ параметр
    n_fft: usize,
    linear_phase_main: bool,
    noise_floor_db: f64,
) -> Vec<f64> {
    // base_mag = total - subsonic - peq (главный фильтр без PEQ и subsonic)
    let base_mag: Vec<f64> = total_mag_db.iter()
        .zip(subsonic_mag_db.iter())
        .zip(peq_mag_db.iter())
        .map(|((t, s), p)| t - s - p)
        .collect();

    // base_phase: уважает выбор пользователя
    let base_phase = if linear_phase_main {
        vec![0.0; total_mag_db.len()]
    } else {
        minimum_phase_from_magnitude(&base_mag, n_fft)
    };

    // PEQ phase — ВСЕГДА min-phase (биквады по физике)
    let peq_phase = minimum_phase_from_magnitude(peq_mag_db, n_fft);

    // Subsonic phase — ВСЕГДА min-phase (Butterworth-8 в DSP)
    let subsonic_phase = minimum_phase_from_magnitude(subsonic_mag_db, n_fft);

    // Total phase = main + peq + subsonic
    base_phase.iter()
        .zip(peq_phase.iter())
        .zip(subsonic_phase.iter())
        .map(|((b, p), s)| b + p + s)
        .collect()
}
```

### Прокидывание peq_mag_db до composite_phase_inner

Найти в `generate_model_fir` все вызовы `composite_phase_inner` и
передать `peq_mag` (он уже доступен как параметр функции).
Аналогично в `iterative_refine` (helpers.rs).

### Sanity check: minimum_phase от vec![0.0; n] = vec![0.0; n]

Если PEQ нет (peq_mag_db все 0), `minimum_phase_from_magnitude`
должна возвращать zeros. Проверить — добавить защитную ветку:

```rust
let peq_phase = if peq_mag_db.iter().all(|&v| v.abs() < 1e-9) {
    vec![0.0; peq_mag_db.len()]
} else {
    minimum_phase_from_magnitude(peq_mag_db, n_fft)
};
```

То же для subsonic.

## Шаг 2: expected_phase в E2E теста

Уже учитывает PEQ + subsonic min-phase (см. b140.0 phase
assertion). Но проверить что на peak частоте PEQ ожидаемая rotation
действительно > 20° для тестовых полос. Если sample_peq_bands()
возвращает слабые полосы — усилить (Q=4, gain=±6 dB чтобы rotation
был заметным).

## Acceptance после фикса

1. `e2e_peq_phase_present_in_realized` PASS для всех 4 configs c
   PEQ (включая `linear_no_subsonic_with_peq`).
2. `e2e_acceptance_matrix` — все 8 configs PASS с phase_err < 5°
   (раньше 3 FAIL были с subsonic).
3. Все existing 170+ cargo тестов PASS.
4. Vitest 143+ PASS.
5. Golden hash baseline (LR4) не изменился.

## Subagent верификация

После фикса оркестрировать subagent:

> Запусти `cargo test --test e2e_export -- --nocapture` в
> /Users/olegryzhikov/phaseforge. Отчитайся:
> 1. e2e_acceptance_matrix: 8 configs — каждый PASS/FAIL с
>    mag_err и phase_err.
> 2. e2e_peq_phase_present_in_realized: PASS/FAIL для каждого
>    config с PEQ.
> 3. Все existing cargo: PASS count / FAIL list.
> 4. vitest: PASS count / FAIL list.
>
> Подтвердить что expectation: 8/8 acceptance + 4/4 PEQ rotation + 170+
> existing — все PASS.

## Bump

- `src-tauri/tauri.conf.json` → `0.1.140` (numeric).
- `src-tauri/src/lib.rs` startup → b140.1.
- `src/lib/version.ts` → b140.1.
- skill `build-version`.

## Что НЕ делать

- Не менять SUM (это b140.2).
- Не менять filterEquals (это b140.3).
- Не трогать peq-optimize и auto-align.

## Правила

- Один коммит: `fix: Composite splits phase into main/PEQ/subsonic sources (b140.1)` + Co-Authored-By.
- 7-vector review.
- Без нарратива.
- При любом FAIL после фикса — diagnostic, не слепые правки.

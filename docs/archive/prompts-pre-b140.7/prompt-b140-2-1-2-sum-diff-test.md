# Промт для Code: b140.2.1.2 — diff test Legacy vs New на реальном проекте

**Тип:** диагностический cargo test. Без bump версии.

## Контекст

В b140.2.1.1 New SUM визуально не совпадает с Legacy на реальном
5-полосном проекте 5wayNew. Synthetic тесты прошли — значит
реальные данные имеют edge case который синтетика не воспроизводит
(alignment_delay, polarity, mixed phase, нормализация, что-то ещё).

Нужен diagnostic тест: загрузить реальный проект, прогнать через
**оба** pipeline, сравнить point-by-point, локализовать где расходятся.

## Pre-flight: копия проекта в репозиторий

Code-сессия не имеет доступа к Google Drive путям. Перед запуском
тестов нужно:

```
mkdir -p /Users/olegryzhikov/phaseforge/test-fixtures/5wayNew
cp -r '/Users/olegryzhikov/Library/CloudStorage/GoogleDrive-shd911@gmail.com/Мой диск/2_created with phaseforge/5wayNew/' \
      /Users/olegryzhikov/phaseforge/test-fixtures/5wayNew/
```

`.gitignore` должен содержать `test-fixtures/` чтобы реальные данные
не коммитились в публичный репозиторий.

Если копия не получается через `cp` (Cloud Storage), Кириллу нужно
скопировать вручную. До этого тест не запустится.

## Что нужно сделать

### 1. Cargo integration test `tests/e2e_sum_real_project.rs`

```rust
#[test]
#[ignore]  // отдельно через `cargo test --ignored` чтобы не падал в CI
fn diff_legacy_vs_new_5wayNew() {
    let project_dir = "/Users/olegryzhikov/phaseforge/test-fixtures/5wayNew";
    if !std::path::Path::new(project_dir).exists() {
        eprintln!("Skip: 5wayNew fixture не найден в {}", project_dir);
        return;
    }

    // 1. Load .pfproj via existing project.rs parser
    let pfproj_path = format!("{}/5wayNew.pfproj", project_dir);
    let project = load_project_from_path(&pfproj_path).expect("load project");

    // 2. Load measurements из inbox
    for band in &mut project.bands {
        if let Some(file_name) = &band.measurement_file {
            let path = format!("{}/inbox/{}", project_dir, file_name);
            band.measurement = Some(import_measurement_file(&path).expect("import"));
        }
    }

    // 3. Прогнать через "legacy aggregation" — port from renderSumMode
    //    (см. FrequencyPlot.tsx:3349-3939). Реализовать функцию
    //    legacy_aggregate_sum(bands) → SumOutput.
    let legacy_result = legacy_aggregate_sum(&project.bands);

    // 4. Прогнать через "new aggregation" — port from evaluateSum
    //    (band-evaluator.ts:509-...). Уже есть в e2e_sum.rs как
    //    coherent_sum mirror; расширить для всех polarity/delay/incoherent.
    let new_result = new_aggregate_sum(&project.bands);

    // 5. Compare point-by-point с локализацией расхождения
    let report = compare_sums(&legacy_result, &new_result);
    eprintln!("=== Legacy vs New aggregation diff on 5wayNew ===\n{}", report);

    // Не assert — это diagnostic.
}
```

### 2. Структура отчёта

```rust
fn compare_sums(legacy: &SumOutput, new: &SumOutput) -> String {
    let mut s = String::new();

    // Freq grid
    s.push_str(&format!("Freq grid: legacy {} pts, new {} pts\n",
        legacy.freq.len(), new.freq.len()));
    if legacy.freq.len() == new.freq.len() {
        let max_freq_diff = legacy.freq.iter().zip(new.freq.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        s.push_str(&format!("  max freq diff: {} Hz\n", max_freq_diff));
    } else {
        s.push_str("  ⚠ DIFFERENT GRID SIZES — реземплинг для сравнения\n");
    }

    // Σ Target magnitude
    if let (Some(l), Some(n)) = (&legacy.sum_target_mag, &new.sum_target_mag) {
        let (max_diff, max_idx) = max_abs_diff(l, n);
        s.push_str(&format!("Σ target mag: max diff {:.3} dB at {:.0} Hz\n",
            max_diff, legacy.freq[max_idx]));
    }

    // Σ Target phase, Σ Corrected mag/phase, Σ Measurement mag/phase
    // ... аналогично

    // Per-band разница: какая полоса вносит наибольшее расхождение
    for i in 0..legacy.per_band.len() {
        let lb = &legacy.per_band[i];
        let nb = &new.per_band[i];
        // ... mag_diff, phase_diff
    }

    s
}
```

### 3. Ports of aggregation logic

**Legacy port** (из renderSumMode):
- Common grid: `interpolate_log` extension до 20–20k если нужно.
- Per-band normalization: average passband 200–2000 Hz.
- Coherent sum: complex (cos/sin), polarity flip, alignment_delay
  phase rotation = `360 × f × delay`.
- Incoherent fallback: `10·log10(Σ 10^(m/10))`, polarity ignored.

**New port** (из evaluateSum):
- Common grid через `buildLogGrid(nMax, fMin, fMax)`.
- Resample per-band на common grid.
- Без per-band normalization passband.
- Coherent / incoherent — те же формулы.

**Главные подозреваемые точки расхождения:**
- Common grid алгоритм (extension vs union).
- Per-band normalization (legacy applies, new — нет).
- Resample interpolation (legacy → `interpolate_log` Rust, new →
  TypeScript helper).
- Alignment delay direction (sign convention).

### 4. Что прислать обратно

Отчёт diff с цифрами:

```
=== Legacy vs New aggregation diff on 5wayNew ===
Freq grid: legacy 1024 pts, new 512 pts
  ⚠ DIFFERENT GRID SIZES — main suspect

Σ target mag: max diff 4.3 dB at 35 Hz
Σ corrected mag: max diff 6.1 dB at 250 Hz
Σ corrected phase: max diff 180° at 12 Hz (likely wrap, but check)
Σ measurement: legacy null, new null (no diff)

Per-band:
  Band 1 (Purifi): mag align ok, phase rotation differs by 180° in 200–500 Hz (polarity?)
  Band 2: ...
```

Это и есть локализация бага.

### 5. Что НЕ делать

- Не править код production до получения diff отчёта.
- Не предлагать гипотезы.
- Не коммитить — это диагностика.

## Правила

- Если test fail при отсутствии fixture — `eprintln!` skip и Ok.
- Если fixture есть — выполнить, прислать отчёт.
- Не делать bump версии.
- Не коммитить.

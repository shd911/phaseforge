# ТЗ: Анализ качества замера после импорта (b135–b136)

**Цель:** автоматически проверять качество загруженного измерения сразу
после импорта и предлагать пользователю применимые рекомендации
(установить границы оптимизации, добавить exclusion zone, применить
сглаживание). Это убирает сюрпризы на этапе Optimize и помогает
отделить артефакты замера от полезного сигнала.

**Out of scope:** автоматическое исправление без подтверждения, детекция
отражений / комб-фильтрации (Floor Bounce — отдельный модуль), детекция
сетевой наводки 50/60 Гц, анализ микрофонных щелчков и clipping.

---

## Решения по дизайну (зафиксировано)

1. **Триггеры запуска анализа:** только при первом импорте измерения
   и при merge NF+FF. Replace measurement и Floor Bounce — НЕ триггерят.
2. **Поведение:** блокирующая модалка сразу после успешного импорта/merge.
3. **Хранение результата:** в проекте, поле `band.settings.analysis`.
   Флаг `analysis_dismissed: bool` — чтобы не показывать повторно.
4. **Реализация в два этапа:** b135 — три «надёжных» детектора, b136 —
   фазовый когерентный анализ.

---

## Структура данных

В `src-tauri/src/project.rs` расширить `SettingsData`:

```rust
#[serde(default)]
pub analysis: Option<AnalysisResult>,
#[serde(default)]
pub analysis_dismissed: bool,
```

Новый модуль `src-tauri/src/analysis/mod.rs`:

```rust
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AnalysisResult {
    pub timestamp: String,        // ISO 8601
    pub app_version: String,      // "0.1.0-b135"
    pub findings: Vec<Finding>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Finding {
    pub id: String,               // "noise_floor_low", "rolloff_hf", ...
    pub severity: Severity,       // Info | Warning | Error
    pub title: String,
    pub description: String,
    pub freq_range: Option<(f64, f64)>,
    pub recommendations: Vec<Recommendation>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum Severity { Info, Warning, Error }

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Recommendation {
    pub action: ActionType,
    pub label: String,            // «Установить нижнюю границу = 35 Гц»
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type", content = "value")]
pub enum ActionType {
    SetOptLowerBound(f64),
    SetOptUpperBound(f64),
    AddExclusionZone { low_hz: f64, high_hz: f64 },
    ApplySmoothing(String),       // "1/3", "1/6", "1/12"
}
```

Tauri-команда:

```rust
#[tauri::command]
pub fn analyze_measurement(measurement: Measurement) -> Result<AnalysisResult, String>;
```

---

## Этап b135: три детектора

### Детектор 1: Шумовой пол (snizhu и sverkhu)

**Алгоритм:**

1. Перевести SPL в dB: `spl_db = 20*log10(magnitude)`.
2. По log-частотной сетке скользящее окно шириной 1 октава.
3. В каждом окне посчитать `std(spl_db)` и `mean(spl_db)`.
4. **Шумовой пол снизу:** найти максимальную частоту `f_low` такую,
   что для всех частот ниже `f_low` `std < 2 dB` на протяжении ≥ 1 октавы.
5. **Шумовой пол сверху:** аналогично, минимальная частота `f_high`
   такая что выше — стабилизация на noise floor.

**Finding:**

```
id: "noise_floor_low" / "noise_floor_high"
severity: Warning
title: «Шумовой пол ниже 35 Гц»
description: «SPL стабилизируется на −42 дБ, данные ниже не несут полезной информации»
freq_range: (1.0, 35.0)
recommendation: SetOptLowerBound(35.0), label: «Установить нижнюю границу оптимизации = 35 Гц»
```

### Детектор 2: Низкочастотный rolloff из-за окна

**Алгоритм:**

1. Если у `Measurement` есть `impulse_response` (поле IR) — оценить
   эффективную длину окна `t_w` как время до падения огибающей IR
   на 40 dB ниже peak.
2. Минимальная разрешимая частота: `f_min = 2 / t_w` (две точки на период).
3. Если IR отсутствует — эвристика:
   - найти участок ниже резонансной части где SPL спадает монотонно
     со skoroстью ≥ 18 dB/oct без resonance features (отсутствие
     локальных экстремумов на ширине < 1/3 октавы);
   - принять `f_min` как точку начала этого rolloff.

**Finding:**

```
id: "lf_rolloff_window"
severity: Warning
title: «Низкочастотный спад связан с окном замера»
description: «Эффективное окно ~250 мс, ниже 8 Гц данные недостоверны»
freq_range: (1.0, 8.0)
recommendation: SetOptLowerBound(8.0)
```

Если `f_min` ≤ существующего `noise_floor_low` — finding не дублируется
(приоритет noise floor warning).

### Детектор 3: Высокочастотный обрыв

**Алгоритм:**

1. Анализировать SPL только на верхних 1.5 октавах (например, выше 8 кГц
   при sample rate 48 кГц, до Найквиста минус 5%).
2. Найти частоту `f_cliff` где начинается monotonic спад со скоростью
   ≥ 24 dB/oct без resonance features до конца спектра.
3. Это типичная anti-aliasing filter / mic-rolloff характеристика.

**Finding:**

```
id: "hf_cliff"
severity: Info
title: «Высокочастотный обрыв на 18.5 кГц»
description: «Резкий спад без резонансов — anti-aliasing или микрофонная характеристика»
freq_range: (18500.0, 24000.0)
recommendation: SetOptUpperBound(18500.0)
```

### Объединённое поведение

Если детектор не нашёл проблем — соответствующий Finding не создаётся.
Если все три детектора пусты — `findings = []`, модалка показывает
«Анализ замера: проблем не обнаружено» и сразу даёт кнопку «Закрыть».

---

## Этап b136: фазовый когерентный анализ

**Один новый детектор**: «Несоответствие магнитуды и фазы».

**Алгоритм:**

1. Вычислить `phase_min(f)` через `compute_minimum_phase(magnitude)`
   (Hilbert transform). Эта функция уже реализована в Rust.
2. Снять propagation delay из измеренной фазы (использовать
   `delay_seconds` из `band.settings`, если он установлен; иначе —
   `remove_measurement_delay`).
3. `phase_excess(f) = phase_meas(f) − phase_min(f)`.
4. Найти участки где `|d phase_excess / d log₂(f)| > 90°/окт` при
   `|d magnitude_db / d log₂(f)| < 3 dB/окт`. Это значит фаза скачет,
   а магнитуда — нет, что физически невозможно для минимально-фазовой
   системы.
5. Группировать соседние такие участки в диапазоны шириной ≤ 1/12 октавы.

**Finding:**

```
id: "phase_magnitude_mismatch"
severity: Warning
title: «Скачок фазы на 2.1 кГц»
description: «Несоответствие magnitude/phase, ширина ~1/30 октавы — вероятный артефакт замера»
freq_range: (2050.0, 2150.0)
recommendation: AddExclusionZone(2050.0, 2150.0)
```

**Тонкости:**
- Алгоритм чувствителен к выбору propagation delay. Лучше сначала прогнать
  `remove_measurement_delay` и сохранить result в `band.settings`, потом
  анализировать.
- Может давать ложные срабатывания на crossover-allpass участках. На b136
  ограничиться только узкими (< 1/12 октавы) скачками — широкие all-pass
  characteristics не трогать.

---

## Фронтенд: `MeasurementAnalysisDialog.tsx`

Структура:

```
┌─ Анализ замера: speaker-LF.txt ──────────────────┐
│                                                  │
│  Найдено: 2 предупреждения, 1 заметка            │
│                                                  │
│  ☐ ⚠ Шумовой пол ниже 35 Гц                     │
│      SPL стабилизируется на −42 дБ               │
│      [ Установить нижнюю границу = 35 Гц ]       │
│                                                  │
│  ☐ ⚠ Высокочастотный обрыв на 18.5 кГц          │
│      Резкий спад без резонансов                  │
│      [ Установить верхнюю границу = 18.5 кГц ]   │
│                                                  │
│  ☐ ℹ Динамический диапазон ниже 30 дБ            │
│      Пиковый уровень −12 дБ, шум −38 дБ          │
│      [ Применить сглаживание 1/6 окт ]           │
│                                                  │
│  [ Применить отмеченное ]   [ Игнорировать всё ] │
└──────────────────────────────────────────────────┘
```

**Поведение:**
- Чекбоксы у каждого finding — пользователь сам выбирает что применить.
- Кнопка под finding (например, «Установить нижнюю границу = 35 Гц») —
  применяет одну рекомендацию сразу, отмечает finding выполненным.
- «Применить отмеченное» — применяет всё с галочками, закрывает модалку,
  ставит `analysis_dismissed = true`.
- «Игнорировать всё» — закрывает без действий, ставит `analysis_dismissed = true`.
- Закрытие через × / Escape — не сохраняет dismissed (модалка появится
  снова при следующем импорте этого же файла? — нет, не появится, потому
  что флаг триггера привязан к самому событию импорта; см. ниже).

**Триггеры показа модалки:**

- В `measurement-actions.ts:handleImportMeasurement()` после успешного
  `import_measurement` → `analyze_measurement` → если есть findings,
  показать модалку.
- В `merge-actions` (или где у тебя merge live) после успешного `merge_measurements`
  → аналогично.
- `analysis_dismissed` проверяется ТОЛЬКО при загрузке проекта: если
  `false` и есть `analysis` с findings — модалка показывается один раз
  при первом открытии полосы. Если `true` — игнорируется навсегда (для
  этого измерения).

**Применение рекомендаций — какие функции вызывать:**

- `SetOptLowerBound(f)` → установить `peqDirectLow(f)` и переключить
  `peqRangeMode("manual")`.
- `SetOptUpperBound(f)` → установить `peqDirectHigh(f)` + manual mode.
- `AddExclusionZone(low, high)` → push в `band.exclusionZones`.
- `ApplySmoothing(level)` → установить `band.settings.smoothing` + перезапустить
  smoothing pipeline (использовать существующую функцию).

После применения любой рекомендации — `pushHistory()` (b132 session undo
должен ловить это действие, чтобы можно было откатить).

---

## Edge cases

| Сценарий | Поведение |
|---|---|
| Анализ не нашёл проблем | Модалка с сообщением «проблем не обнаружено», кнопка Закрыть |
| Импорт 4 файлов подряд через Add band loop | Модалка появляется после каждого импорта — это тяжело UX, но соответствует решению «при первом импорте» |
| Restore проекта где analysis_dismissed=true | Модалка не показывается |
| Restore проекта где analysis_dismissed=false и есть findings | Модалка показывается один раз |
| Merge → потом replace measurement | Replace не триггерит анализ. analysis устаревает, но остаётся в state |
| Применение рекомендации меняет SPL (smoothing) | Анализ НЕ перезапускается автоматически. Пользователь может вручную |

---

## Acceptance

**b135:**
1. Импорт измерения с явным шумовым полом снизу → модалка показывает finding,
   применение устанавливает peqDirectLow.
2. Импорт измерения с резким HF cliff → finding HF + apply устанавливает peqDirectHigh.
3. Импорт измерения без явных проблем → модалка «всё в порядке», dismiss.
4. Merge NF+FF → анализ запускается на результате.
5. Replace measurement → анализ НЕ запускается.
6. Открытие проекта где analysis_dismissed=true → модалка не появляется.
7. Применение рекомендации создаёт запись в session undo (Cmd+Z откатывает).

**b136:**
8. Импорт измерения с искусственным узким phase glitch → finding с правильным
   диапазоном.
9. Чисто минимально-фазовое тестовое измерение → нет phase mismatch findings.
10. Crossover-passive measurement (естественный all-pass на кроссовере) → нет
    ложных findings (потому что широкий участок, > 1/12 октавы).

---

## Этапы внедрения

| Билд | Содержание |
|---|---|
| **b135** | Rust analysis модуль, 3 детектора (noise floor low/high, LF rolloff, HF cliff), модалка фронта, интеграция с импортом и merge |
| **b136** | 4-й детектор (phase coherence), без изменения UI |

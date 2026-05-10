# Промт для Code: b135 — анализ замера после импорта (3 детектора)

ТЗ целиком: `docs/TZ-measurement-analysis.md` (этап b135).
Текущий билд: 0.1.0-b134 → bump до 0.1.0-b135.

## Что нужно сделать

### 1. Rust: модуль `src-tauri/src/analysis/mod.rs`

Структуры `AnalysisResult`, `Finding`, `Severity`, `Recommendation`,
`ActionType` — см. ТЗ, секция «Структура данных».

Tauri-команда:

```rust
#[tauri::command]
pub fn analyze_measurement(measurement: Measurement) -> Result<AnalysisResult, String>;
```

Внутри pipeline вызывает три pure-функции:

```rust
fn detect_noise_floor(freq: &[f64], magnitude: &[f64]) -> Option<NoiseFloorResult>;
fn detect_lf_rolloff(measurement: &Measurement) -> Option<Finding>;
fn detect_hf_cliff(freq: &[f64], magnitude: &[f64], sample_rate: u32) -> Option<Finding>;
```

Алгоритмы — см. ТЗ, «Этап b135: три детектора».

Важно:
- Все вычисления ведутся в SPL (dB), не в линейной магнитуде.
- log-частотная сетка для скользящих окон.
- Оценки порогов (std < 2 dB, slope > 18 dB/oct, ширина окна 1 октава) —
  захардкодить как `const` в начале модуля, **не выносить** в параметры
  пользователя на первой итерации.
- Покрыть unit-тестами в `src-tauri/src/analysis/mod.rs` (mod tests):
  один тест на каждый детектор с синтетическими входами (плоский SPL,
  rolloff, cliff).

Зарегистрировать команду в `src-tauri/src/lib.rs:invoke_handler`.

### 2. Расширение `ProjectFile`

В `src-tauri/src/project.rs` добавить в `SettingsData`:

```rust
#[serde(default)]
pub analysis: Option<AnalysisResult>,
#[serde(default)]
pub analysis_dismissed: bool,
```

Импортировать `AnalysisResult` из `crate::analysis`.

### 3. Frontend: компонент `src/components/MeasurementAnalysisDialog.tsx`

Принимает props:

```typescript
interface Props {
  open: boolean;
  bandId: string;
  bandName: string;
  fileName: string;        // отображается в заголовке
  result: AnalysisResult;
  onApply: (selected: Finding[]) => void;
  onDismiss: () => void;   // «Игнорировать всё» / Escape
}
```

Структура UI — см. ТЗ, «Фронтенд: MeasurementAnalysisDialog.tsx».

Особенности:
- Чекбокс рядом с каждым finding (по умолчанию выключен).
- Кнопка под finding применяет одну рекомендацию и помечает finding
  как выполненный (визуально серым с галочкой).
- «Применить отмеченное» применяет все отмеченные галочками + ставит
  `analysis_dismissed = true`.
- «Игнорировать всё» закрывает без действий + `analysis_dismissed = true`.
- Escape = «Игнорировать всё».
- Если `findings = []` → показать одноэкранное сообщение «Замер выглядит
  чисто, проблем не обнаружено» + кнопка Закрыть.

### 4. Применение рекомендаций

Хелпер `src/lib/analysis-actions.ts`:

```typescript
export function applyRecommendation(
  bandId: string,
  rec: Recommendation,
): void;
```

Маппинг `ActionType` → существующие функции:

| Action | Что делает |
|---|---|
| `SetOptLowerBound(f)` | `setPeqDirectLow(f)` + `setPeqRangeMode("manual")` |
| `SetOptUpperBound(f)` | `setPeqDirectHigh(f)` + `setPeqRangeMode("manual")` |
| `AddExclusionZone {low, high}` | push в `band.exclusionZones` через store API |
| `ApplySmoothing(level)` | `setBandSmoothing(bandId, level)` + перезапуск smoothing pipeline |

Каждое применение **должно** триггерить `pushHistory("Apply recommendation: <id>")`
из b132 session undo.

### 5. Интеграция с импортом и merge

В `src/lib/measurement-actions.ts:handleImportMeasurement()`:

1. После успешного `import_measurement` и присвоения measurement полосе:
2. `const result = await invoke<AnalysisResult>("analyze_measurement", { measurement });`
3. Сохранить в `band.settings.analysis = result`, `analysis_dismissed = false`.
4. Если `result.findings.length > 0` — открыть `MeasurementAnalysisDialog`.

В `merge-actions` (или где у тебя живёт merge_measurements handler) —
аналогично, после получения `MergeResult.measurement`.

### 6. Показ диалога при загрузке проекта

В `project-io.ts:doLoad()` после `restoreState`:

1. Найти первую полосу где `settings.analysis_dismissed === false` и
   `settings.analysis?.findings.length > 0`.
2. Показать для неё `MeasurementAnalysisDialog`.
3. После закрытия — `analysis_dismissed = true` сохраняется в state.

Если несколько полос с не-dismissed analysis — показать только для
первой (избежать спама диалогов).

### 7. Bump версии

- `src-tauri/tauri.conf.json` — version + productName/title до b135.
- `src-tauri/src/lib.rs` — startup-лог.
- После билда — skill `build-version`.

## Acceptance

1. Импорт измерения с шумовым полом ниже 30 Гц → модалка показывает
   warning, кнопка устанавливает peqDirectLow.
2. Импорт измерения с HF cliff на 18 кГц → finding HF, apply устанавливает
   peqDirectHigh + range_mode="manual".
3. Импорт чистого измерения → модалка «проблем не обнаружено», dismiss.
4. Merge NF+FF → анализ запускается на результате слияния.
5. Replace measurement (через UI смены файла на той же полосе) → анализ
   НЕ запускается (по дизайну).
6. Применение рекомендации создаёт запись в session undo (Cmd+Z откатывает
   изменение peqDirectLow).
7. Сохранение проекта → `analysis` и `analysis_dismissed` пишутся в `.pfproj`.
8. Открытие проекта с dismissed=true → модалка не появляется.
9. Открытие проекта с dismissed=false и непустыми findings → модалка
   появляется один раз для первой найденной полосы.

## Регрессионная проверка

- b131–b134 не сломаны: close confirmation, undo/redo, named snapshots,
  shortcuts/tooltips/escape.
- Импорт без проблемных файлов — приложение не виснет на анализе.
- Merge NF+FF и Floor Bounce работают штатно.
- Производительность: анализ для измерения 65k точек должен укладываться
  в < 100 мс (это pure Rust, должно быть быстро).
- Восстановление снимка через Versions диалог не показывает analysis-модалку
  повторно (если `analysis_dismissed=true` в снимке).

## Правила (из CLAUDE.md)

- Один коммит: `feat: post-import measurement analysis (b135)` +
  Co-Authored-By.
- 7-vector review.
- Без нарратива прогресса.
- Unit-тесты для Rust детекторов обязательны (по одному на каждый,
  с синтетическими входами).
- `cargo tauri dev` для проверки, `cargo tauri build` для финальной сборки.

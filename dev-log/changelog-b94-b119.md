# PhaseForge — Changelog b94→b119

## 🎛 Что нового за последние 25 билдов

---

### b97 — Gaussian Min-Phase & IR Sum
- Gaussian-фильтры теперь поддерживают min-phase режим через Hilbert-transform
- Когерентная сумма на IR/Step с нормализацией
- Переключение легенды IR/Step

### b98 — Snapshots
- Полный переезд системы снэпшотов: видимость, раздельный capture по категориям, dB-режим

### b99 — Export & Phase
- Вкладка Export: отображение фазы
- Gaussian min-phase: корректное отображение и WAV-экспорт с паритетом фазы
- Тесты: comprehensive filter phase coverage (все типы, порядки, комбинации)

### b99.1 — SolidJS Store Fixes
- Исправлен критический баг: shared-node в SolidJS store — cross-contamination между FilterConfig разных полос
- Стабилизирован drag PEQ: gen guard, убран peqDragJustEnded
- Исправлен mouse wheel на PEQ freq/gain/Q inputs

### b99.2 — MixedPhase FIR & UX
- MixedPhase FIR: per-filter Gaussian Hilbert, peak-centered windowing
- Tauri async dialog для подтверждения удаления полосы
- Убраны дублирующие кнопки Meas/Tgt/Corr из IR/Step toolbar

### b100 — Release
- Первый публичный релиз: CI/CD pipeline с macOS DMG + Windows MSI

### b101 — Per-Band Alignment Delay & SUM View
Крупный релиз. Новая система alignment delay:
- Per-band задержки в SUM-режиме с применением к IR/Step/GD
- Mouse wheel управление delay-инпутами
- Delay row на IR/Step вкладке в SUM view
- Рефакторинг: типографские design tokens, extract evaluateBand, split Rust модулей (peq, fir), устранение code duplication

### b102 — Auto-Align v1
- Первая версия Auto-Align: gradient descent оптимизация задержек
- Precondition guard, исправлен checkbox

### b104–b106 — Refactoring
- Унифицированная кнопочная система (dlg-btn, size variants)
- Централизованные цвета кривых (named constants)
- Консолидация рефактора

### b107 — SUM Matrix
- Индикаторы оптимизации в SUM matrix
- Дефолтная видимость фазы
- Hover highlight

### b108 — Auto-Align HF→LF
- Алгоритм переработан: HF→LF последовательная оптимизация с propagation отрицательных задержек
- Физически корректная модель: задержка всегда ≥ 0, HF полоса — reference

### b109 — Auto-Align Freq Grid
- Интерполяция всех замеров на общую частотную сетку перед align
- Устранение артефактов на XO при разных сетках замеров

### b113 — SUM Phase Compensation
- Frequency-dependent power-weighted delay compensation для SUM phase display
- Фаза суммы теперь читаема после auto-align без набега HF

### b114 — Spacing Design Tokens
- 122 хардкодированных padding/gap/margin заменены на `--space-*` токены
- Консистентные отступы по всему UI

### b116 — Auto-Align Correctness
- Persistence: `alignment_delay` теперь сохраняется в проект (serde default)
- Geometric mean для центра XO (log-scale корректно)
- Phase unwrap после PEQ/XO суммирования
- Guard для вырожденной интерполяции
- Корректные length guards в когерентной сумме

### b117 — Security
- Path traversal защита в `save_project` и `export_fir_wav`
- Sanitization `project_name` через `file_name()`
- CSP включён: `default-src 'self'`

### b118 — Performance
- `renderSumMode`: все `interpolate_log` → `Promise.all` (параллельно)
- `evaluate_target`: устранён дублирующий вызов, замена scalar offset
- `compute_peq_complex` + `compute_cross_section` → параллельно per band
- Результат: N× снижение IPC latency при рендере SUM

### b119 — Audit Complete
Закрытие всех оставшихся findings из pre-release аудита:
- Gen guards после каждого await в renderSumMode (защита от concurrent render)
- Length assertions после invoke в auto-align
- Clamp gradient descent к физическому диапазону
- Reset задержек всех полос при early return
- Path traversal guard на фронтенде (merge paths)
- Null safety для `m.phase` перед `remove_measurement_delay`
- Удалён dead code (`bandCenterFreq`, неиспользуемые exports/params)

---

**Итого b94→b119:** Auto-Align от идеи до production-ready, полный аудит безопасности и корректности, performance оптимизация IPC, design system (typography + spacing tokens), CI/CD с автоматическим релизом Win/macOS.

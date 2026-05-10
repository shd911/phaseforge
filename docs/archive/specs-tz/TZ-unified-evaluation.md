# ТЗ: Unified Evaluation Pipeline (refactor)

**Цель:** превратить разветвлённый pipeline target/SPL/FIR/IR/SUM в
единое каноническое вычисление, где все вкладки и экспорт читают
**один результат**. Добавление новых полей (subsonic_protect и т.п.)
должно требовать правки в **одном** месте, не в N.

**Триггер:** каскад b138.0–b138.4 показал что одна фича (subsonic
phase) требовала фиксы в 3+ pipeline (target evaluation, SPL phase,
FIR phase, потенциально IR/SUM/group delay).

---

## Текущая карта pipeline (из аудита кода)

### Что вычисляется параллельно

| Сигнал | Pipeline | Где живёт | Phase источник |
|---|---|---|---|
| Target curve (SPL view) | `evaluate_target` + `addGaussianMinPhase` | `band-evaluation.ts:evaluateBand` | Rust returns 0 для Gaussian → frontend Hilbert |
| Target curve (FIR) | `evaluate_target_standalone` (своя freq grid 5–40k Hz) | `fir-export.ts:generateBandImpulse` | передаётся `fill(0)`, Rust сам Hilbert в MinimumPhase mode |
| PEQ correction (display) | `compute_peq_complex` | вызывается из FrequencyPlot и auto-align | Rust возвращает phase |
| PEQ correction (FIR) | передаётся только `peqMag` | `generate_model_fir` | **PEQ phase теряется!** |
| Impulse / Step | `compute_impulse` | FrequencyPlot для отдельных вкладок | от собственно собранного mag+phase |
| SUM view | свой агрегат + `compute_impulse` | FrequencyPlot SUM-секция | пересобирается ad-hoc |
| Group delay | `compute_group_delay` (frontend gradient) | FrequencyPlot | от phase любого источника |

### Главные структурные дефекты

1. **Target phase вычисляется 3 раза** в 3 разных местах с разными
   входами (один — на freq измерения, другой — 5–40k Hz, третий — внутри
   FIR mode selection).

2. **Скрытый баг:** `generate_model_fir` получает `peq_mag`, но не
   `peq_phase`. PEQ-полосы вносят phase shift (минимально-фазовые), но
   FIR об этом не знает. Может проявляться когда PEQ полос много с
   высокими Q.

3. **`addGaussianMinPhase` (frontend Hilbert per-filter)** — патч
   поверх архитектурного провала. Должен быть встроен в основной
   evaluator, не отдельным шагом.

4. **Subsonic protection** добавлена в 3 местах с одной формулой
   (`(f_sub/f)^16`): Rust `apply_filter`, frontend `subsonicMagDb`,
   FIR phase mode demotion. После b138.4 формула в одном источнике
   `subsonicMagDb` — это промежуточный фикс.

5. **FilterConfig копируется в 4-6 мест** поле за полем (`unwrapFilter`,
   `unwrapFilterConfig`, `withOverride`, `cloneFilter`, тестовые
   miрроры). После b138.2 все включают `subsonic_protect`. Но любое
   новое поле потребует обхода всех 4-6 мест.

---

## Целевая архитектура

### Один canonical evaluator

Frontend-side функция-агрегатор. Возвращает всё что нужно для
отображения на любой вкладке + для экспорта. Кэшируется через SolidJS
resource — пересчёт только при изменении входов.

```typescript
// src/lib/band-evaluator.ts (новый)

export interface BandEvalRequest {
  band: BandState;
  freq?: number[];          // если undefined — берётся из measurement или log-grid 20–20k
  includeFir?: boolean;     // FIR coefficients (lazy, дорого)
  includeIr?: boolean;      // IR/step от measurement+correction
}

export interface BandEvalResult {
  freq: number[];           // частотная сетка (либо из measurement, либо log-grid)

  // Magnitude pipeline
  measurementMag: number[] | null;  // smoothed measurement
  targetMag: number[];              // pure target (HP × LP × shelves × tilt × subsonic)
  peqMag: number[];                 // PEQ correction
  combinedTargetMag: number[];      // target + peq (для отображения)

  // Phase pipeline (всегда полная, со всем — Gaussian min-phase, subsonic min-phase, PEQ phase)
  targetPhase: number[];            // pure target phase, со всеми реконструкциями
  peqPhase: number[];               // PEQ phase
  combinedTargetPhase: number[];    // target + peq phase

  // Опционально
  fir?: { impulse: number[]; time: number[] };   // если includeFir
  ir?: { impulse: number[]; step: number[]; time: number[] };  // если includeIr

  // Метаданные для UI
  refLevel: number;                 // вычисленный auto-reference offset
}

export async function evaluateBandFull(req: BandEvalRequest): Promise<BandEvalResult>;

// SolidJS resource для реактивного кэширования
export function createBandEvalResource(
  band: () => BandState,
  freq?: () => number[] | undefined,
  options?: { includeFir?: boolean; includeIr?: boolean }
): Resource<BandEvalResult>;
```

Внутри `evaluateBandFull`:
1. Применяет smoothing к measurement (если есть).
2. Один раз вызывает `evaluate_target` (один freq grid, один результат).
3. Один раз вызывает `compute_peq_complex` (mag + phase вместе).
4. Применяет Gaussian min-phase Hilbert если нужно (текущая логика
   `addGaussianMinPhase` встраивается сюда).
5. Применяет subsonic min-phase Hilbert если нужно (текущая логика).
6. Складывает target + peq для combined.
7. Опционально: FIR через `generate_model_fir` с уже корректными
   target и phase (не fill(0)).
8. Опционально: IR через `compute_impulse`.

Все consumers (FrequencyPlot SPL/IR/Step/GD/SUM, fir-export, любые
другие) читают из этого результата. Никаких inline invoke цепей.

### Что устраняет

- Дубликат phase pipeline (3 точки → 1).
- Скрытый баг с PEQ phase в FIR.
- Inline invoke цепи в FrequencyPlot и fir-export.
- `addGaussianMinPhase` как отдельная функция (встраивается).
- Расхождение freq grid между display и export.

### Что остаётся

- Низкоуровневые Tauri commands (evaluate_target, compute_peq_complex,
  compute_minimum_phase) **не трогаем** — они используются как
  building blocks evaluator-а.
- PEQ optimizer (`auto_peq_lma`) — независимый pipeline, не часть
  display.
- Auto-align — отдельный pipeline (использует cross-section напрямую),
  не тронем в этом refactor.
- Анализ замера (b135) — независимая функция, не тронем.

---

## План этапов

Принцип: **каждый этап работоспособен сам по себе**. Старый код не
удаляется до тех пор, пока новый не проверен. Каждый этап имеет
golden-тесты которые гарантируют output identical к старому pipeline.

### Этап 0: Тестовая инфраструктура (ОБЯЗАТЕЛЬНЫЙ ПРЕДВАРИТЕЛЬНЫЙ)

**До любого refactor** создать golden-тесты на текущий output, чтобы
сравнивать new vs old.

**Что делается:**
- `src/lib/__tests__/golden-pipeline.test.ts` — снэпшот-тесты на
  output `evaluateBand` для эталонной полосы (синтетический measurement
  + target).
- 6 эталонных конфигураций:
  1. Gaussian HP=632, linear_phase=true, subsonic OFF
  2. Gaussian HP=632, linear_phase=true, subsonic ON
  3. Gaussian HP=632, linear_phase=false, subsonic OFF
  4. Gaussian HP=632, linear_phase=false, subsonic ON
  5. LR4 HP=80, без subsonic (не-Gaussian baseline)
  6. Без HP (full-range полоса)
- Снимок выходов в JSON-файл (vitest `toMatchSnapshot`).
- Cargo тесты для текущих Rust commands — добавить snapshot-тесты на
  `evaluate_target` для тех же 6 конфигураций (чтобы Rust output
  фиксировался).

**Acceptance:**
- Все 6 snapshot тестов записаны и проходят на текущем коде (b138.4).
- `npm test` и `cargo test` оба зелёные.
- Snapshot файлы закоммичены — это reference для будущих этапов.

**Bump:** b139.0

**Размер:** маленький, ~1 день.

**Риск:** низкий. Только новые файлы, ничего не меняем.

---

### Этап 1: Создать `BandEvaluator` параллельно со старым

**Что делается:**
- Новый файл `src/lib/band-evaluator.ts` с `evaluateBandFull` и
  `createBandEvalResource`.
- Внутри использует **существующие** invoke (evaluate_target,
  compute_peq_complex, compute_minimum_phase, compute_impulse,
  generate_model_fir).
- Логика встраивает текущие addGaussianMinPhase + subsonic Hilbert в
  единую функцию.
- Старый код (`evaluateBand`, `addGaussianMinPhase`,
  `generateBandImpulse`) остаётся **нетронутым**.

**Что НЕ делается:**
- Никаких изменений в callers (FrequencyPlot, fir-export, peq-optimize).
- Никаких изменений в Rust.

**Тесты:**
- Vitest unit-тесты на `evaluateBandFull` для тех же 6 эталонных
  конфигураций. Сравнить выход с golden snapshots из Этапа 0.
- **Должны совпадать побитово** (или с ограниченной численной
  погрешностью 1e-10) — это доказывает что новый evaluator
  эквивалентен старому.

**Acceptance:**
- 6 snapshot тестов на `evaluateBandFull` проходят.
- Diff между `evaluateBand`+`addGaussianMinPhase` и `evaluateBandFull`
  для одних и тех же входов < 1e-10 dB по magnitude и < 0.01° по
  phase.
- Старые тесты `evaluateBand` всё ещё проходят (старый код не тронут).
- `cargo test` и `npm test` зелёные.

**Bump:** b139.1

**Размер:** средний, основная работа этапа.

**Риск:** средний — может не получиться эквивалентность сразу. Если
есть числовая разница — diagnostic logging до достижения
эквивалентности. Не двигаться к Этапу 2 пока не достигнута.

---

### Этап 2: Переключить FrequencyPlot SPL view на BandEvaluator

**Что делается:**
- В `FrequencyPlot.tsx` найти места где вызываются `evaluate_target`
  и `compute_peq_complex` для SPL/phase plot.
- Заменить на чтение из `createBandEvalResource(activeBand)`.
- Старая `evaluateBand` НЕ удаляется — может ещё использоваться в
  других местах.

**Тесты:**
- Manual checklist (быстрый, через .dmg):
  - SPL plot для Gaussian HP linear=true + subsonic ON выглядит
    идентично b138.4
  - Phase plot для всех 4 комбинаций — идентично b138.4
  - Group delay не сломан
  - PEQ marker rendering работает
- Snapshot golden test: выходные пиксели/SVG точек графика сравниваются
  с reference (если возможно через jsdom + canvas mock).

**Acceptance:**
- SPL и Phase plot выглядят идентично (визуально + golden test).
- 6 acceptance конфигураций для b138.4 проходят.
- Все existing тесты зелёные.

**Bump:** b139.2

**Размер:** средний.

**Риск:** низкий-средний. Главная опасность — пропустить inline
invoke где-нибудь в FrequencyPlot (он большой, ~4000 строк). Audit
перед изменением: grep на `invoke.*evaluate_target` и `invoke.*compute_peq_complex`
внутри FrequencyPlot.

---

### Этап 3: Переключить FIR export на BandEvaluator

**Что делается:**
- `fir-export.ts:generateBandImpulse` использует `evaluateBandFull` с
  `includeFir: true`.
- В `evaluateBandFull` встроена FIR-генерация: получается combined
  target+peq mag и phase, передаётся в `generate_model_fir` уже с
  правильным `model_phase` (не fill(0)).
- Это **исправляет скрытый баг** (PEQ phase в FIR).

**Тесты:**
- Manual: экспорт FIR для всех 4 комбинаций Gaussian. Проверить:
  - Phase response сгенерированного FIR соответствует SPL plot
  - Impulse меняется при включении subsonic (b138.4 acceptance)
  - Step response отражает subsonic
  - Существующие FIR без Gaussian не изменились
- Cargo тест: существующий тест `generate_model_fir` всё ещё проходит.
- Reference golden FIR: для одной эталонной полосы (LR4 HP=80) до
  refactor сохранить bytewise FIR coefficients. После refactor
  убедиться что они **не изменились** для тех же входов (без
  PEQ-фазового вклада, который раньше игнорировался).

**Acceptance:**
- FIR с subsonic ВКЛ показывает phase rotation в зоне инфразвука.
- FIR без Gaussian/subsonic — bytewise идентичен (regression).
- FIR с PEQ — phase теперь отражает PEQ contribution. Это **новое
  поведение**, не regression. Но: sanity check — без PEQ FIR не
  изменился.

**Bump:** b139.3

**Размер:** средний.

**Риск:** средний — может выявить что PEQ-phase contribution меняет
FIR в случаях где раньше его не было. Нужна явная проверка на
эталонных данных.

---

### Этап 4: SUM view + IR/Step view на BandEvaluator

**Что делается:**
- В FrequencyPlot.tsx секция SUM использует bandEval каждой полосы и
  складывает.
- IR/Step view для отдельной полосы использует `evaluateBandFull` с
  `includeIr: true`.

**Тесты:**
- Manual: SUM для проекта с 3 полосами Gaussian/LR4/None — выглядит
  идентично b138.4.
- IR view для каждой полосы — идентичен.
- Step view — идентичен.

**Acceptance:**
- SUM, IR, Step выглядят идентично b138.4.
- subsonic в любой полосе корректно отражается в SUM IR.

**Bump:** b139.4

**Размер:** средний (SUM view сложнее, agregating логика).

**Риск:** средний.

---

### Этап 5: Удаление дубликатов

**Что делается:**
- Удаление `addGaussianMinPhase` из `band-evaluation.ts` (логика
  встроена в evaluator).
- Удаление inline invoke цепей в FrequencyPlot.
- Удаление `evaluate_target_standalone` из Rust **только если** не
  используется (audit перед удалением).
- `evaluateBand` либо удаляется (если все callers переключены), либо
  становится thin wrapper над `evaluateBandFull` (для backward compat
  с peq-optimize.ts если он там используется).
- subsonic detection logic остаётся в одном месте (evaluator).

**Тесты:**
- Все existing тесты зелёные.
- Manual прогон: Optimize, FIR export, SUM, IR/Step, Save/Open, Cmd+Z,
  Versions — всё работает.

**Acceptance:**
- Bundle size уменьшился (удалили дубликаты).
- Никакой dead code (ESLint).
- All tests pass.

**Bump:** b139.5

**Размер:** маленький.

**Риск:** низкий — это finalisation.

---

## Тестовая стратегия

### Уровни тестов

1. **Unit (vitest)** — `evaluateBandFull` на синтетических входах,
   проверка против golden snapshots. Каждое изменение в evaluator
   должно поддерживать снапшоты.

2. **Snapshot (vitest)** — выход `evaluateBandFull` для 6 эталонных
   конфигураций. Изменение поведения = намеренное обновление снапшота
   с явным reasoning.

3. **Integration (cargo test)** — Rust commands как раньше. Не
   меняются — никаких новых тестов на этом уровне.

4. **Regression (manual checklist)** — после каждого этапа
   фиксированный список из 10 пунктов прогоняется на `.dmg`. Список
   находится в `docs/regression-checklist.md` (создаётся в Этапе 0).

5. **Golden FIR bytewise** — для одной эталонной конфигурации (LR4
   HP=80, без PEQ, без subsonic) сохраняется reference FIR. После
   каждого этапа сравнение должно дать identity match.

### Правило валидации каждого этапа

Этап **не считается завершённым**, пока:
1. `cargo test` зелёный.
2. `npm test` зелёный.
3. Golden snapshots не разошлись (или разошлись по согласованной
   причине).
4. Regression checklist пройден на `.dmg`.
5. Audit в начале этапа не обнаружил новых дубликатов.

---

## Edge cases

| Сценарий | Поведение |
|---|---|
| Полоса без measurement | evaluateBandFull возвращает только target+peq, measurement=null |
| Полоса с peqBands=[] | peqMag=zeros, peqPhase=zeros, combined = target |
| Полоса с все frozen PEQ | peqBands → compute_peq_complex без проблем |
| Полоса с targetEnabled=false | evaluateBandFull возвращает targetMag=null или skip target часть |
| Изменение target во время `evaluateBandFull` (race) | SolidJS resource отменяет старый запрос, перезапускает |
| BandState immutable update | `createBandEvalResource` triggers пересчёт |
| Сохранение/загрузка проекта | Тот же state → тот же evaluator output |
| Cmd+Z (b132 history) | History восстанавливает state → resource триггерит пересчёт автоматически |

---

## Что НЕ делаем в этом refactor

- Не меняем PEQ optimizer (`auto_peq_lma`) — независимый pipeline.
- Не меняем auto-align — низкоуровневый, оставляем как есть.
- Не меняем analyze_measurement (b135) — отдельная функция.
- Не объединяем низкоуровневые Tauri commands в одну.
- Не трогаем PEQ stale logic (b136) — она работает поверх state, не
  pipeline.
- Не оптимизируем performance (это отдельная задача после refactor).

---

## Учёт уроков b138 каскада

1. **Перед каждым этапом — grep audit** на дубликаты и hidden
   miрроры. Известные точки — список выше.

2. **Diagnostic-first при провале этапа**. Если acceptance не
   проходит — НЕ повторять фикс вслепую. Запросить логи.

3. **Тесты валидации до коммита**. Каждый коммит проходит cargo test
   + vitest + golden snapshots.

4. **Версия в заголовке** — обязательная проверка что .dmg = текущий
   билд.

5. **Один этап = один коммит**. Не объединять.

6. **Старый код не удаляется до полной миграции**. Этапы 1–4 живут с
   обоими pipeline. Только Этап 5 делает cleanup.

---

## Открытые вопросы для обсуждения

1. **Frontend-side vs Rust-side унификация.** В этом ТЗ выбрана
   frontend (новая `band-evaluator.ts`). Альтернатива: новая Rust
   command `evaluate_band_full(...)`. Frontend проще (меньше изменений
   в Rust, больше реактивности через SolidJS resource), но FFI-overhead
   на каждом invoke остаётся. Согласен с frontend?

2. **PEQ phase в FIR — это исправление или регрессия?** Сейчас FIR
   игнорирует PEQ phase. После refactor — учитывает. Это **меняет
   output FIR** для проектов с PEQ. Сообщить пользователям как
   intentional improvement, или сделать fallback флаг?

3. **`evaluate_target_standalone` — оставить или удалить?** Сейчас
   используется только в FIR pipeline (свой freq grid 5–40k). После
   refactor evaluator поддерживает любой grid → standalone не нужен.
   Удалить?

4. **Размер reference FIR golden test.** Сравнивать побитово impulse
   массив (8192–65536 коэффициентов float32) — большой файл. Сравнивать
   spectral characteristics (SHA-256 от FFT magnitude) — компактнее, но
   менее строго. Какой подход?

5. **Сроки.** ~5 этапов × 1-2 дня каждый = 1-2 недели. Норм или нужно
   быстрее (сократить scope)?

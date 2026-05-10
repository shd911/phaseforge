# ТЗ: Project Lifecycle (b131+)

**Цель:** привести базовый управление проектом в детерминированное состояние:
закрытие/открытие без потерь, undo в рамках сессии, именованные версии с описанием.

**Out of scope (отдельные задачи):** версионирование `export/` (архив старых
WAV/PEQ при перегенерации), облачное хранилище, мульти-сессионные блокировки.

---

## Часть 1. Базовые триггеры (защита от потери)

### 1.1 Закрытие окна / Cmd+Q

**Сейчас:** `WindowEvent::CloseRequested` не перехвачен → красный крестик и
Cmd+Q убивают приложение мимо `confirmIfDirty()`. Несохранённая работа теряется.

**Нужно:** в `src-tauri/src/lib.rs` повесить `on_window_event` на главное окно.
При `CloseRequested`:

1. Если `isDirty == false` → разрешить закрытие.
2. Если `isDirty == true` → `event.prevent_close()`, эмитнуть событие
   `request-close-confirm` во фронт.
3. Фронт показывает модалку с тремя кнопками: **Save** / **Don't Save** / **Cancel**.
   - Save: вызвать `saveProject()`, при успехе — `invoke("close_window_now")`.
   - Don't Save: сразу `invoke("close_window_now")`.
   - Cancel: ничего.
4. Tauri-команда `close_window_now` вызывает `window.close()` без повторного
   prevent (сделать через флаг `app_state.allow_close = AtomicBool`).

**Acceptance:** при `isDirty=true` крестик и Cmd+Q вызывают модалку. Save сохраняет
и закрывает. Don't Save закрывает без сохранения. Cancel оставляет окно открытым.

### 1.2 Открытие проекта при dirty

**Сейчас:** `loadProject()` зовёт `confirmIfDirty()` — но текущий диалог
бинарный (Discard / Cancel). Save-варианта нет.

**Нужно:** заменить `ask()` на тот же трёхкнопочный диалог Save/Don't Save/Cancel,
вынести его в отдельный компонент `UnsavedChangesDialog.tsx`, переиспользовать
для close, New, Open.

---

## Часть 2. Session Undo (2-3 шага в памяти)

### 2.1 Что считается "шагом"

**Принятая стратегия:** снимок хранит только «лёгкое» состояние —
`peqBands`, `target`, `alignmentDelay`, имена полос, флаги
`targetEnabled/inverted/linkedToNext`, параметры PEQ/FIR/export. Массивы
измерений (`measurement.freq/magnitude/phase`, `originalPhase`) **не**
включаются в снимок. Это держит память на единицах КБ на запись и убирает
риски рассинхронизации PEQ с пересчитанным измерением.

Снимок кладётся в стек **перед** каждым из:

- `Optimize` (Auto Optimizer) — результат может быть неудачным.
- Применение Target (HP/LP filter, shape, slope), переключение
  `targetEnabled`, `inverted`, `linkedToNext`.
- Add / Remove / Rename band.
- Сдвиг `alignmentDelay` (значение — число, лёгкое поле).
- Bulk edit PEQ (выбор нескольких полос → действие).
- Ручная правка PEQ полосы (debounced, см. ниже).

**НЕ снимок** (действия меняют `measurement.*` массивы или их пересчёт —
вне зоны охвата лёгкого undo):

- Импорт / удаление / замена measurement.
- Merge NF+FF.
- Floor Bounce apply.
- Auto-align (если меняет `measurement.phase`).
- UI-only: смена активной вкладки, toggle `show_phase/show_mag/show_target`,
  смена активной полосы.

Защита от случайной потери для действий, не покрытых undo, — отдельные
подтверждения в момент действия (например, диалог при удалении полосы
с измерением).

**Debounced** (один снимок на длинное действие):
- Перетаскивание PEQ маркера — фиксируется после `peqDragging=false` + 300 мс.
- Драг ползунка Q/Gain/Freq в FilterBlock — аналогично.

### 2.2 Структура данных

В `src/stores/history.ts`:

```typescript
type LightBand = {
  id: string;
  name: string;
  peqBands: PeqBand[];
  target: TargetCurve;
  targetEnabled: boolean;
  inverted: boolean;
  linkedToNext: boolean;
  alignmentDelay: number;
  // measurement, settings, originalPhase — НЕ включены
};

type HistoryEntry = {
  bands: LightBand[];          // только лёгкие поля
  activeBandId: string;
  nextBandNum: number;
  peqParams: PeqOptimizeSnapshot;
  firParams: FirSnapshot;
  exportParams: ExportSnapshot;
  label: string;               // "Optimize", "Add band: LF", ...
  ts: number;
};

const MAX_HISTORY = 5;         // Кирилл просит 2-3, кладём 5 с запасом
const undoStack: HistoryEntry[] = [];
const redoStack: HistoryEntry[] = [];
```

Push в `undoStack` через хелпер `pushHistory(label: string)`. При `MAX_HISTORY`
overflow — `shift()` старейший. После любого push → `redoStack = []`.

История **только в памяти**, на диск не сохраняется. При New/Open/закрытии
проекта — оба стека очищаются.

### 2.3 UI

Кнопки **Undo** / **Redo** в основном тулбаре (рядом с File меню). Подписи:
"Undo: Optimize" по hover. Шорткаты: Cmd+Z / Shift+Cmd+Z.

Disabled когда стек пустой.

### 2.4 SolidJS-нюанс

`JSON.parse(JSON.stringify(...))` для лёгких полей подходит — массивов
измерений в снимке нет, размер каждого entry — единицы КБ.

При накатывании снимка — обёртка в `batch()`, чтобы эффекты не срабатывали на
промежуточном состоянии (паттерн `restoreState` в `project-io.ts`).
`measurement` и `settings` каждой полосы при `applyHistory` берутся из
**текущего** appState и переносятся как есть — undo не трогает измерения.

При несовпадении набора полос (snapshot содержит band id, которого больше
нет в текущем state, потому что измерение под него удалили) — bands
пересоздаются по snapshot, но `measurement = null`, `settings = null`.
Пользователь увидит «пустую» полосу и сможет повторно импортировать.

---

## Часть 3. Named Snapshots (версии с описанием)

### 3.1 Структура на диске

```
<project>/
  <name>.pfproj           # текущее состояние
  inbox/
  target/
  export/
  snapshots/
    index.json            # манифест всех версий
    <uuid>.pfproj         # копия .pfproj на момент снапшота
```

`snapshots/index.json`:

```json
{
  "version": 1,
  "entries": [
    {
      "id": "01HW4...",       // ULID
      "ts": "2026-04-30T14:22:15Z",
      "description": "После увеличения регуляризации до 0.3",
      "app_version": "0.1.0-b131",
      "file": "01HW4....pfproj"
    }
  ]
}
```

Снапшот = копия `.pfproj` целиком. Файлы из `inbox/`, `target/`, `export/` НЕ
копируются (раздули бы папку проекта в разы). Это значит: если пользователь
удалил измерение из inbox/ и потом восстановил снапшот — измерение не вернётся,
restore выдаст warning "1 measurement file missing".

### 3.2 Tauri commands (`src-tauri/src/snapshots.rs` — новый модуль)

```rust
#[tauri::command]
fn create_snapshot(project_dir: String, description: String, project: ProjectFile)
    -> Result<SnapshotEntry, String>;

#[tauri::command]
fn list_snapshots(project_dir: String) -> Result<Vec<SnapshotEntry>, String>;

#[tauri::command]
fn load_snapshot(project_dir: String, id: String) -> Result<ProjectFile, String>;

#[tauri::command]
fn delete_snapshot(project_dir: String, id: String) -> Result<(), String>;
```

Все commands валидируют: `project_dir` существует, `id` ULID-формата (regex),
никаких `..` в путях. `create_snapshot` создаёт `snapshots/` если её нет.

### 3.3 UI: VersionsDialog

В File меню добавить пункт **Versions...** (между Save As и Recent Projects).

Окно:

- Кнопка **Save Version** — открывает inline-prompt: textarea для описания
  + кнопки Save/Cancel. Описание обязательно (пустое — disabled Save).
- Список существующих версий: `[дата] [описание] [build] [Restore] [Delete]`.
- Restore: спросить "Save current state as a version before restoring?" → Yes/No/Cancel.
  Yes → создать снапшот с auto-описанием "Auto-snapshot перед загрузкой '<target>'",
  затем загрузить выбранный снапшот через `restoreState()`.

### 3.4 Что Save Version делает с current

- Текущее состояние **не меняется** — пользователь продолжает работать с тем же
  `.pfproj`.
- `isDirty` после Save Version → не сбрасывается (это копия, не save).
- Снапшот фиксирует: bands, PEQ params, FIR params, target, alignment.

---

## Edge cases

| Сценарий | Поведение |
|---|---|
| Restore при dirty=true | Диалог: "Save current as version / Discard / Cancel" |
| Snapshot при dirty=true | Разрешить — снапшот = текущая live-копия appState |
| `inbox/` файл удалён, restore | Warning toast: "Measurement X not found, band cleared" |
| Snapshot в проекте v1 (старый формат без папки) | Disabled, тултип: "Save As to migrate to v2 first" |
| Описание с emoji/кириллицей | UTF-8 в JSON, OK |
| `snapshots/index.json` повреждён | Не падать; показать пустой список + кнопка "Rebuild from .pfproj files" |
| Undo прямо после Optimize | Восстанавливает PEQ как было, плюс target/measurement тоже (full appState) |

---

## Этапы внедрения

| Билд | Содержание |
|---|---|
| **b131** | Часть 1 целиком (close handler + UnsavedChangesDialog). Минимальный фикс. |
| **b132** | Часть 2 (Undo/Redo, in-memory). |
| **b133** | Часть 3 (Snapshots на диск, VersionsDialog). |
| **b134** | Полировка: shortcuts, тултипы, edge-case warnings. |

Каждый билд — отдельный коммит, проходит 7-vector review до закрытия.

---

## Acceptance Criteria

**b131:**
1. Cmd+Q при dirty → модалка Save/Don't Save/Cancel.
2. Save в модалке успешно сохраняет и закрывает.
3. New/Open/close используют один и тот же `UnsavedChangesDialog`.

**b132:**
4. После Optimize — Undo возвращает PEQ полосы и target в pre-Optimize состояние.
5. Стек ограничен 5 шагами.
6. Cmd+Z / Shift+Cmd+Z работают глобально.
7. Перетаскивание PEQ-маркера за один drag = один history entry.

**b133:**
8. Save Version создаёт `snapshots/<uuid>.pfproj` + запись в index.json.
9. Restore загружает снапшот, opt-предлагает auto-snapshot текущего.
10. Versions диалог отображает все версии с датой/описанием.

---

## Принятые решения

1. **Авто-снимок по таймеру не делается** ни в первой итерации, ни в
   последующих. Версии создаются только вручную через «Сохранить версию»,
   с обязательным описанием.

2. **Лимит количества версий не задаётся.** Удаление — вручную, кнопкой
   рядом с каждой записью в окне «Версии».

3. **Поле `next_band_num` при возврате к версии** берётся из снимка как есть.
   Возврат полностью заменяет состояние проекта, поэтому конфликтов
   с идентификаторами полос не возникает.

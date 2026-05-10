# Промт для Code: b133 — именованные версии проекта на диск

ТЗ целиком: `docs/TZ-project-lifecycle.md` (Часть 3).
Текущий билд: 0.1.0-b132 → bump до 0.1.0-b133.

## Принципиальный выбор

Версия = копия `.pfproj` с описанием. Файлы из `inbox/`, `target/`,
`export/` **не** копируются. Это значит: если измерение удалено из `inbox/`,
после восстановления версии полоса будет с `measurement = null`, появится
toast-предупреждение «Файл измерения X не найден».

Версии создаются **только вручную** через диалог «Версии». Авто-снимков
по таймеру нет.

## Что нужно сделать

### 1. Rust-модуль `src-tauri/src/snapshots.rs`

Структуры:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotEntry {
    pub id: String,            // ULID или timestamp+random hex (см. ниже)
    pub ts: String,            // ISO 8601 UTC
    pub description: String,
    pub app_version: String,   // напр. "0.1.0-b133"
    pub file: String,          // "<id>.pfproj"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotIndex {
    pub version: u32,          // 1
    pub entries: Vec<SnapshotEntry>,
}
```

Tauri-команды:

```rust
#[tauri::command]
pub fn create_snapshot(
    project_dir: String,
    description: String,
    app_version: String,
    project: ProjectFile,
) -> Result<SnapshotEntry, String>;

#[tauri::command]
pub fn list_snapshots(project_dir: String) -> Result<Vec<SnapshotEntry>, String>;

#[tauri::command]
pub fn load_snapshot(project_dir: String, id: String) -> Result<ProjectFile, String>;

#[tauri::command]
pub fn delete_snapshot(project_dir: String, id: String) -> Result<(), String>;

#[tauri::command]
pub fn rebuild_snapshot_index(project_dir: String) -> Result<u32, String>;
```

Идентификатор `id`: если есть свободная зависимость в Cargo.toml — взять `ulid`.
Иначе — собрать из `chrono::Utc::now().format("%Y%m%dT%H%M%S")` + 4 hex
символа из `rand`. Главное — лексикографически сортируется по времени.

Все команды:
- валидируют что `project_dir` существует и не содержит `..`;
- `id` — regex `^[A-Za-z0-9_-]+$`, никаких слешей;
- путь к файлу собирается через `Path::join`, **не** конкатенацией строк;
- `create_snapshot` создаёт `snapshots/` если её нет;
- `list_snapshots` при отсутствии `index.json` возвращает пустой массив,
  не ошибку;
- если `index.json` повреждён (parse error) — `list_snapshots` возвращает
  `Err("INDEX_CORRUPTED")` (фронт обработает кнопкой Rebuild);
- `rebuild_snapshot_index` сканирует `snapshots/*.pfproj`, восстанавливает
  index из метаданных каждого файла (description берёт из самого `.pfproj`
  если он там есть, иначе пустую строку).

Зарегистрировать команды в `lib.rs` `invoke_handler`.

### 2. Хранение description в `.pfproj`

Расширить `ProjectFile` в `src-tauri/src/project.rs`:

```rust
#[serde(default)]
pub snapshot_description: Option<String>,
#[serde(default)]
pub snapshot_id: Option<String>,
```

Эти поля заполняются только в файлах внутри `snapshots/`. В основном
`<name>.pfproj` они `None`. Это позволяет `rebuild_snapshot_index` собрать
индекс из orphan-файлов.

### 3. Frontend: `src/lib/snapshots.ts`

Обёртки над Tauri-командами + state-сигналы:

```typescript
export const [snapshotsList, setSnapshotsList] =
    createSignal<SnapshotEntry[]>([]);

export async function refreshSnapshots(): Promise<void>;
export async function createSnapshotForCurrentProject(
    description: string
): Promise<SnapshotEntry>;
export async function restoreSnapshot(
    id: string,
    saveCurrentFirst: boolean,
    autoDescription: string,
): Promise<void>;
export async function deleteSnapshot(id: string): Promise<void>;
export async function rebuildSnapshotIndex(): Promise<number>;
```

`createSnapshotForCurrentProject`:
- собирает текущий `ProjectFile` через `buildProjectData()` (вынести её
  в экспорт из `project-io.ts`);
- зовёт `create_snapshot` с `description`, версией приложения из
  `tauri.conf.json` (или хардкод-константа в `lib/version.ts`);
- после успеха — `refreshSnapshots()`;
- `isDirty` **не** сбрасывает (это копия, не save).

`restoreSnapshot`:
- если `saveCurrentFirst === true` — сначала вызвать
  `createSnapshotForCurrentProject(autoDescription)`;
- `load_snapshot(project_dir, id)` → `ProjectFile`;
- запустить тот же путь что `doLoad()` в `project-io.ts`: `restoreState`
  с правильным `projDir` и `version`. Аккуратно: текущий
  `currentProjectPath()` остаётся прежним (это live `.pfproj`), не
  меняется на путь снимка;
- `clearHistory()` (стек session undo очищается);
- `setIsDirty(true)` — после restore состояние отличается от
  записанного `.pfproj`, пользователь может захотеть Save.

### 4. Компонент `src/components/VersionsDialog.tsx`

Структура:

```
┌─ Версии проекта ─────────────────────────────────┐
│                                                  │
│  [+ Сохранить версию]                            │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │ 2026-04-30 14:22 │ После увеличения reg... │  │
│  │ b133              [Восстановить] [×]      │  │
│  ├────────────────────────────────────────────┤  │
│  │ 2026-04-30 12:05 │ Базовая настройка после │  │
│  │ b133              [Восстановить] [×]      │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│                                       [Закрыть]  │
└──────────────────────────────────────────────────┘
```

Поведение:
- При открытии диалога — `refreshSnapshots()`.
- «Сохранить версию» открывает inline textarea + кнопки Сохранить/Отмена.
  Сохранить disabled пока textarea пустая.
- Если `snapshotsList()` пустой — показать «Версий ещё нет».
- Если `refreshSnapshots()` вернул ошибку `INDEX_CORRUPTED` — показать
  «Индекс версий повреждён» + кнопку «Перестроить индекс»
  (`rebuildSnapshotIndex()`).
- Кнопка «×» рядом с записью — `confirm`-диалог перед `deleteSnapshot`.
- «Восстановить»:
  1. Если `isDirty` или есть несохранённые изменения — `confirm`-диалог
     «Сохранить текущее состояние как версию перед восстановлением?»
     с кнопками Да / Нет / Отмена.
  2. Да → `restoreSnapshot(id, true, "Авто-снимок перед загрузкой '<desc>'")`.
  3. Нет → `restoreSnapshot(id, false, "")`.
  4. Отмена → ничего.

### 5. Интеграция в FileMenu

В `src/components/FileMenu.tsx` добавить пункт «Версии…» между «Save As»
и «Recent Projects». Disabled когда `projectDir() === null` (проект v1
или не сохранён) с тултипом «Сохраните проект, чтобы создавать версии».

### 6. Bump версии

- `src-tauri/tauri.conf.json` — version + productName/title до b133.
- `src-tauri/src/lib.rs` — startup-лог.
- После билда — skill `build-version`.

## Edge cases

| Сценарий | Поведение |
|---|---|
| Проект v1 (без папки) | Пункт «Версии…» disabled, тултип «Save As to migrate to v2 first» |
| Snapshot при dirty=true | Разрешён, snapshot = текущая live-копия |
| `inbox/` файл удалён, restore | Восстановление проходит, полоса без измерения, toast «Файл измерения X не найден» (вернёт сам `restoreState`) |
| `index.json` повреждён | Кнопка «Перестроить индекс» в диалоге |
| Описание с emoji/кириллицей | UTF-8 в JSON, OK |
| Восстановление при наличии session-undo стека | После restore — `clearHistory()`. Откатить сам restore через Cmd+Z нельзя — он вне session undo |

## Acceptance

1. «Сохранить версию» с описанием создаёт `snapshots/<id>.pfproj`
   и запись в `snapshots/index.json`.
2. Список версий обновляется без перезапуска приложения.
3. «Восстановить» при наличии изменений предлагает диалог Да/Нет/Отмена.
4. После восстановления состояние совпадает с моментом создания версии
   (PEQ, target, alignment, параметры).
5. Удаление версии убирает файл и запись из index.
6. Удаление файла из `inbox/` и последующий restore показывает toast,
   приложение не падает.
7. Повреждённый `index.json` обрабатывается через кнопку «Перестроить».
8. Пункт меню «Версии…» disabled на v1-проектах.
9. После restore текущий `<name>.pfproj` не перезаписан, флаг `isDirty=true`.

## Регрессионная проверка

- b131 (close confirmation) и b132 (session undo) работают.
- New / Open / Save / Save As / Recent.
- Импорт измерений, экспорт FIR/PEQ.
- Особое внимание: restoreState вызывается из двух мест (Open и
  restoreSnapshot) — поведение должно быть идентично.

## Правила (из CLAUDE.md)

- Один коммит: `feat: named project snapshots (b133)` + Co-Authored-By.
- 7-vector review.
- Без нарратива прогресса.
- `cargo tauri dev` для проверки, `cargo tauri build` для финальной сборки.

# Промт для Code: b131 — защита от потери при закрытии/открытии

ТЗ целиком: `docs/TZ-project-lifecycle.md` (Часть 1).
Текущий билд: 0.1.0-b130 → bump до 0.1.0-b131.

## Что нужно сделать

1. **Tauri close handler.** В `src-tauri/src/lib.rs` повесить `on_window_event`
   на главное окно. При `WindowEvent::CloseRequested`:
   - если фронт сообщил «можно закрывать» (через флаг `allow_close: AtomicBool`
     в `tauri::State`) — пропустить;
   - иначе `api.prevent_close()` и `app_handle.emit("request-close-confirm", ())`.

2. **Tauri-команда `close_window_now`.** Устанавливает флаг `allow_close = true`
   и зовёт `window.close()` повторно. Зарегистрировать в `invoke_handler`.

3. **Компонент `src/components/UnsavedChangesDialog.tsx`** с тремя кнопками:
   Сохранить / Не сохранять / Отмена. Промис-API как у `showProjectNamePrompt`
   в `project-io.ts` — экспортировать `showUnsavedChangesDialog(): Promise<"save" | "discard" | "cancel">`.

4. **Подписка в `src/App.tsx`** на событие `request-close-confirm`. Логика:
   - `save` → `await saveProject()`, при успехе `invoke("close_window_now")`;
   - `discard` → `invoke("close_window_now")`;
   - `cancel` → ничего.

5. **Перевести `confirmIfDirty()` в `project-io.ts`** на новый диалог.
   Текущий `ask()` Save-варианта не имеет — заменить на трёхкнопочный.
   Сценарии: New / Open / Recent.

6. **Bump версии:**
   - `src-tauri/tauri.conf.json`: `version` и `productName/title` → b131
   - `src-tauri/src/lib.rs`: startup-лог
   - после билда: переименовать артефакты (см. skill `build-version`)

## Acceptance

- Cmd+Q при `isDirty=true` → модалка Сохранить / Не сохранять / Отмена.
- Сохранить → файл записан, окно закрылось.
- Не сохранять → окно закрылось без записи.
- Отмена → окно осталось открытым, изменения целы.
- Красный крестик ведёт себя так же, как Cmd+Q.
- Тот же диалог появляется при New / Open / Recent если `isDirty=true`.

## Правила (из CLAUDE.md)

- Один блок изменений → один коммит: `feat: window close confirmation (b131)`
  с Co-Authored-By.
- 7-vector review до закрытия задачи.
- Никакой нарратив прогресса. Только: изменённые пути + одна строка результата.
- `cargo tauri dev` для проверки. `cargo tauri build` только перед коммитом
  итога. **Не** `cargo build --release`.
- Read только нужные участки: grep по `confirmIfDirty`, `WindowEvent`,
  `invoke_handler`, не читать файлы целиком.

## Регрессионная проверка

После b131 убедиться что всё ещё работают:
- New Project (диалог имени, создание папки, авто-сохранение)
- Open / Recent Projects
- Save / Save As (Save As копирует inbox/target/export)
- импорт измерений и floor bounce
- экспорт FIR/PEQ

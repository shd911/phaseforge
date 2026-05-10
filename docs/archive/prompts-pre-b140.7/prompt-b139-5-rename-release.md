# Промт для Code: переименование релиза b139.4c → b139.5

**Тип:** rename для совместимости Windows MSI. Bump до 0.1.0-b139.5.

## Контекст

CI Windows MSI падает с ошибкой:
`pre-release identifier in app version must be numeric-only`

Буква `c` в `0.1.139-4c` не принимается WiX. Решение — переименовать
билд в `b139.5` (numeric-only). Будущие точечные фиксы внутри билда
нумеруются `.1`, `.2` и т.д. (всё numeric).

## Что нужно сделать

### 1. Удалить старый тег b139.4c

Локально и на remote:

```
cd /Users/olegryzhikov/phaseforge
git tag -d b139.4c
git push origin :refs/tags/b139.4c
```

Это нужно чтобы Release `b139.4c` на GitHub можно было удалить
(или он автоматически очистится).

### 2. Bump до b139.5

- `src-tauri/tauri.conf.json` — version: `0.1.139` (без pre-release suffix).
  Title окна: `PhaseForge v0.1.0-b139.5`.
- `src-tauri/src/lib.rs` — startup-лог: b139.5.
- `src/lib/version.ts` — `0.1.0-b139.5`.

### 3. Артефакты через skill build-version

Имена артефактов: `PhaseForge_0.1.139_aarch64.dmg`,
`PhaseForge_0.1.139_x64-setup.exe` (или `.msi`).

### 4. Коммит и push

```
git add ...
git commit -m "chore: rename b139.4c → b139.5 (numeric-only for MSI) (b139.5)"
git push origin main
git tag -a b139.5 -m "PhaseForge b139.5 — composite mode + unified pipeline"
git push origin b139.5
```

CI должен пройти на этот раз.

### 5. Обновить release notes

Тот же текст что был для b139.4c, только заменить заголовок и
упомянуть переименование:

```markdown
## PhaseForge b139.5 — защитный фильтр + единая сущность для всех вкладок

(переименовано с b139.4c для совместимости с Windows MSI builder —
WiX требует numeric-only pre-release identifier)

[остальной текст release notes как раньше]
```

После того как CI создаст Release `b139.5` — открыть на GitHub и
вставить notes.

## Acceptance

1. Тег `b139.4c` удалён локально и на origin.
2. Версия в `tauri.conf.json` numeric: `0.1.139`.
3. Тег `b139.5` запушен.
4. CI workflow `Build & Release` прошёл зелёным на macOS, Windows,
   Linux test.
5. Release `b139.5` создан с обоими бинарниками (.dmg + .exe/.msi).

## Что НЕ делать

- Не оставлять `0.1.139-4c` или другие буквенные suffixes в version
  field.
- Не пытаться "пропатчить" WiX — формат версии должен быть
  совместимым с обоими bundlers.

## Правила

- Один коммит для rename: `chore: rename b139.4c → b139.5 (numeric-only for MSI) (b139.5)` + Co-Authored-By.
- Без нарратива.

# Промт для Code: релиз b140.4 на GitHub

**Тип:** релиз. CI собирает обе платформы по тегу `b*`.

## Что нужно сделать

### 1. Push main + tag

```
cd /Users/olegryzhikov/phaseforge
git status   # должно быть clean
git log -3 --oneline   # верх — b140.3.8 коммит
git push origin main
git tag -a b140.4 -m "PhaseForge b140.4 — Composite mode + unified SUM"
git push origin b140.4
```

### 2. Мониторинг CI

```
gh run list --limit 3
gh run watch
```

Или через web: https://github.com/<owner>/phaseforge/actions

### 3. После прохождения CI

GitHub автоматически создаст Release `b140.4` с прикреплёнными:
- `PhaseForge_0.1.140_aarch64.dmg`
- `PhaseForge_0.1.140_x64-setup.exe` (или `.msi`)

Зайти в Releases на GitHub и **отредактировать описание**, вставив
содержимое `docs/release-notes-b140.4.md`.

### 4. Если CI failed

- Windows MSI: проверить version field в `tauri.conf.json` =
  `0.1.140` (numeric, без буквенных suffix). Если есть suffix — fix.
- macOS bundle_dmg.sh flaky: retry `gh run rerun` или через web UI.

### 5. Acceptance

- Тег `b140.4` виден на GitHub.
- Release создан с обоими бинарниками.
- Описание заполнено release notes.

## Что НЕ делать

- Не bumping до b140.5.
- Не объединять с другими коммитами.

## Правила

- Если push требует креды — пользователь подтвердит вручную.
- Не делать git rebase / amend на текущих коммитах.

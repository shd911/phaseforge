# Промт для Code: релиз b137 на GitHub

**Тип:** релиз. Без bump версии (b137 уже зафиксирован).

CI: `.github/workflows/build.yml` уже настроен. При пуше тега `b*`
или `v*` собирает macOS (.dmg) и Windows (.msi + .exe), создаёт
GitHub Release и заливает артефакты.

## Что нужно сделать

### 1. Убедиться что main чистый

```
cd /Users/olegryzhikov/phaseforge
git status
git log -3 --oneline
```

Ожидается: `71cea02` (b137 Q envelope) на HEAD, working tree clean.
Если есть незакоммиченные изменения — остановиться и сообщить.

### 2. Push main

```
git push origin main
```

### 3. Создать annotated tag b137

```
git tag -a b137 -m "PhaseForge b137 — Project Lifecycle, Measurement Analysis & Q Envelope"
git push origin b137
```

### 4. Запустить мониторинг CI

После push — workflow `Build & Release` автоматически стартует.
Посмотреть статус:

```
gh run list --limit 3
gh run watch
```

(Если `gh` CLI не установлен — открыть в браузере
`https://github.com/<owner>/phaseforge/actions`.)

### 5. После успешной сборки

CI создаст GitHub Release `b137` с прикреплёнными:
- `PhaseForge_0.1.137_aarch64.dmg`
- `PhaseForge_0.1.137_x64-setup.exe` (или `.msi`)

Зайти в Releases на GitHub и **отредактировать описание**, вставив
release notes из `docs/release-notes-b137.md` (создать его в шаге 6
ниже) — или из текста, который пользователь вставит вручную в UI.

### 6. Сохранить release notes в репо для истории

Создать файл `docs/release-notes-b137.md` со списком изменений из
обсуждения с пользователем (text приведён в Cowork-сессии). Файл
закоммитить в отдельном коммите:

```
git add docs/release-notes-b137.md
git commit -m "docs: release notes for b137"
git push origin main
```

(Делать это **после** того как тег уже запушен и CI запустился —
release notes commit не должен попасть в собранный `.dmg`.)

## Acceptance

1. Тег `b137` виден на GitHub.
2. CI workflow `Build & Release` прошёл зелёным на macOS, Windows
   и Linux test job.
3. Release `b137` создан и содержит как минимум один `.dmg` для macOS
   и один установщик для Windows.
4. Описание Release заполнено release-notes из чата.
5. На локальной машине `.dmg` из release загружается и запускается без
   warning про неподписанный binary (если Apple notarization настроен —
   опционально).

## Если что-то пошло не так

- **CI failed на Windows:** проверить логи на GitHub. Часто проблема
  с node-gyp или MSVC toolchain. Не пытаться чинить вслепую — отправить
  логи пользователю.
- **CI failed на macOS:** обычно signing/notarization. Если signing не
  настроен — ничего страшного, .dmg всё равно соберётся.
- **Release создан без файлов:** значит upload-artifact step не
  поднялся. Перезапустить workflow через `gh run rerun`.

## Что НЕ делать

- Не bump'ать версию до b138 — релиз для b137.
- Не удалять и не пересоздавать тег `b137` если CI упал — починить
  workflow и retrigger через workflow_dispatch или новый коммит на
  той же SHA.
- Не редактировать release notes в самом теге (они в release description,
  не в tag annotation).

## Правила (CLAUDE.md)

- Без нарратива прогресса.
- Команды по одной строке (CLAUDE.md правило терминала).
- Если `git push` требует креды — пользователь подтвердит вручную.

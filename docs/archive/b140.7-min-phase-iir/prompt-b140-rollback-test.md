# Промт для Code: A/B test b140.0 vs b140.4 на flat файле

**Тип:** диагностический rollback test. Без bump, без коммита.

## Контекст

Кирилл сообщает регрессию: на flat файле + bandpass target rolloff
сдвинут на ~1/2 октавы в FIR относительно Model. Не зависит от phase
mode (linear / min). До b140.1 (Composite mode добавлен) такого не
было.

Гипотеза: regression в Composite path при iterative_refine.

Быстрый способ подтвердить — собрать b140.0 (до Composite), сравнить
визуально с b140.4 на том же проекте.

## Что нужно сделать

### 1. Сохранить текущее состояние

```
cd /Users/olegryzhikov/phaseforge
git status
git stash push -m "WIP before b140.0 rollback test" 2>/dev/null || true
```

### 2. Чекнуть b140.0

```
git checkout b140.0
git status   # должно быть clean
```

### 3. Сборка b140.0 dev

```
osascript -e 'tell application "PhaseForge" to quit' 2>/dev/null || true
pkill -9 -f -i "phaseforge" 2>/dev/null || true
pkill -9 -f "tauri dev" 2>/dev/null || true
sleep 1
lsof -ti:1420 | xargs kill -9 2>/dev/null || true
sleep 1
nohup cargo tauri dev > /tmp/phaseforge-b140-0.log 2>&1 &
```

Сообщить пользователю: «b140.0 запущен, открой flat проект, вкладка
Export, сделай скриншот, пришли. Логи /tmp/phaseforge-b140-0.log».

### 4. Ждать обратной связи

После того как Кирилл пришлёт скриншот b140.0 → return обратно на main:

```
osascript -e 'tell application "PhaseForge" to quit' 2>/dev/null || true
pkill -9 -f -i "phaseforge" 2>/dev/null || true
pkill -9 -f "tauri dev" 2>/dev/null || true
sleep 1
lsof -ti:1420 | xargs kill -9 2>/dev/null || true
sleep 1
git checkout main
git stash pop 2>/dev/null || true
nohup cargo tauri dev > /tmp/phaseforge-dev.log 2>&1 &
```

Сообщить «вернулись на main (b140.4 + b140.3.8), dev запущен».

## Что прислать обратно после первого checkout

«b140.0 запущено в dev, открой flat проект и пришли скриншот вкладки
Export такой же конфигурации (с теми же HP/LP/sample_rate/taps).
Также пришли строки `iterative_refine:` и `realized_max:` из
терминала».

## Что НЕ делать

- Не bumping версию.
- Не коммитить — это временный rollback test.
- Не запускать `cargo tauri build` (release) — только dev.
- При checkout main back — restore stash чтобы не потерять uncommitted.

## Правила

- Без нарратива.
- Сообщить чётко в каком состоянии git сейчас (b140.0 или main).

# Промт для Code: b140.7.3 — audit sr-dependent grid logic в IIR path

**Тип:** code investigation + bump. Без коммита, без правок DSP до
получения evidence.

## Step 0

```
cd /Users/olegryzhikov/phaseforge
git status
git diff --stat
```

## Контекст

User сообщает что баг (REW broken на 44.1/48k, OK на 88+) идентичен
по паттерну b140.6 регрессии (rolloff shift на тех же sr). Тогда
корень был: разные log grid-ы (firFreq=5..fMaxFir vs evalRes.freq=5..40k),
где `fMaxFir = min(40000, sr·0.95/2)` < 40000 при sr<84k.

Diag b140.7.1 подтвердил: impulse в памяти **одинаков** в IIR PATH,
TS, Rust entry export_fir_wav. WAV сохраняется через хороший Float64.
Но REW показывает broken на 48k.

Гипотеза: где-то в IIR path или его обработке используется
sr-dependent grid (типа fMaxFir или log range), который масштабирует
magnitude/phase на UI plot (поэтому plot выглядит корректно), но
**не применяется** при сохранении импульса в WAV (поэтому WAV broken).

## Что нужно сделать

### 1. Bump до b140.7.3

- `tauri.conf.json` → b140.7.3.
- `lib.rs` startup → b140.7.3.
- `version.ts` → b140.7.3.

### 2. Audit (read-only)

Найти все места в IIR path и downstream где используются:
- `fMaxFir`, `f_max_fir`
- `min(40000`, `40_000.0`
- `sr * 0.95 / 2`, `sample_rate * 0.95`
- Любая sr-dependent grid логика

```
grep -rn "fMaxFir\|f_max_fir\|40000\|40_000\|0.95\s*/\s*2\|0\.95.*sample_rate" src-tauri/src/fir/iir_path.rs src-tauri/src/fir/mod.rs src/lib/band-evaluator.ts src/lib/fir-export.ts
```

Также:
```
grep -rn "resample\|interp\|log_grid\|f_max\|fmax" src-tauri/src/fir/iir_path.rs
```

В `iir_path.rs::generate_min_phase_fir_iir`:
- Какая log grid строится?
- Где она используется?
- Как влияет на realized_mag/phase в output?
- Как влияет на возвращаемый `impulse`?

### 3. Сравнение с b140.6 fix

Look at git diff or commit message of b140.6 fix:
```
git log --all --oneline | head -20
git show <b140.6 commit hash>  # или диф через docs
```

Понять что именно было изменено в b140.6, и применить ту же логику
к IIR path если применимо.

### 4. Записать аудит в docs/iir-grid-audit-b140-7-3.md

Структура:
```
# IIR path grid audit (b140.7.3)

## Все sr-dependent grid точки в IIR pipeline
- file:line — описание

## Сравнение: что одинаково на UI plot и WAV export, что нет
- UI plot путь: ...
- WAV export путь: ...
- Расхождения: ...

## Гипотеза bug-а
- Где именно происходит sr-dependent масштабирование
- Почему UI скрывает bug

## Предлагаемый fix
- Что изменить (без правки в этом промте)
```

### 5. STOP — без правок DSP

Только аудит. Не патчить пока user не одобрит план.

### 6. End-of-prompt

```
osascript -e 'tell application "PhaseForge" to quit' 2>/dev/null || true
pkill -9 -f -i "phaseforge" 2>/dev/null || true
pkill -9 -f "tauri dev" 2>/dev/null || true
pkill -9 -f "tauri-driver" 2>/dev/null || true
sleep 1
lsof -ti:1420 | xargs kill -9 2>/dev/null || true
lsof -ti:5173 | xargs kill -9 2>/dev/null || true
sleep 1
cd /Users/olegryzhikov/phaseforge && nohup cargo tauri dev > /tmp/phaseforge-dev.log 2>&1 &
```

В startup должен быть **b140.7.3** + текущий [DIAG ACTIVE] export-wav
маркер.

## Acceptance

- Title bar = b140.7.3.
- `docs/iir-grid-audit-b140-7-3.md` готов с конкретными file:line.
- Гипотеза бага локализована.
- План фикса описан, ждёт одобрения.

## Что НЕ делать

- Не патчить DSP.
- Не убирать diag tracing.
- Не предполагать без чтения кода.

## Правила

- Только цитаты file:line.
- Один аудит report.

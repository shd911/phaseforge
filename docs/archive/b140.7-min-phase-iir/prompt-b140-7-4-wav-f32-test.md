# Промт для Code: b140.7.4 — тест Float32 WAV без коммита

**Тип:** experimental swap. Bump до b140.7.4. Без коммита до verify.
Если не поможет — откат к Float64.

## Step 0

```
cd /Users/olegryzhikov/phaseforge
git status
git diff --stat
```

## Изменение (one-line swap)

Per audit b140.7.3:
- `src-tauri/src/lib.rs:574` (или там где `export_fir_wav` команда):
  ```rust
  // было:
  fir::export_wav_f64(&impulse, sr, &path)?;
  // стало:
  fir::export_wav_f32(&impulse, sr, &path)?;
  ```

`export_wav_f32` уже существует в `src-tauri/src/fir/mod.rs:408`.

## Bump до b140.7.4

- `tauri.conf.json` → b140.7.4.
- `lib.rs` startup → b140.7.4.
- `version.ts` → b140.7.4.

Diag tracing оставить — пригодится при verify.

## Build + verify

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

Сообщить user-у:
- Title bar = **b140.7.4**.
- Действия:
  1. Перейти на Band 3 HP=2000 sr=48k → Export WAV.
  2. Проверить afinfo:
     ```
     afinfo "<path>"
     ```
     Должно показать **Float32** или **F32**.
  3. Загрузить свежий WAV в REW.
  4. Скрин что REW показывает сейчас.
  5. Также проверить sr=176.4k (regression check) — должно остаться корректным.

## Что НЕ делать

- Не коммитить.
- Не убирать diag tracing.
- Не менять что-то ещё помимо одной строки swap + bump.

## Acceptance

- afinfo показывает Float32 для нового WAV.
- REW для 48k WAV показывает корректный HP shape (или LP для Band 1).
- 176k WAV всё ещё корректный.

## Если не помогает

- Откат: один git command:
  ```
  git checkout -- src-tauri/src/lib.rs
  ```
  (восстанавливает Float64 swap, bump оставляем как маркер тестирования).
- Сообщить user-у что Float64 hypothesis опровергнута.
- Готовить следующий diagnostic — Python-скрипт сравнения WAV bytes vs PhaseForge realized_mag.

## Правила

- Без нарратива.
- Один short report после verify.

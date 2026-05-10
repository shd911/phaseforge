# Промт для Code: b140.7.1 — диагностика export_fir_wav vs UI plot impulse

**Тип:** diagnostic + bump. Bump до b140.7.1 (видимая визуальная
проверка), коммит после verify.

## Step 0

```
cd /Users/olegryzhikov/phaseforge
git status
git log -3 --oneline
```

Текущее состояние: b140.7 IIR path code на месте (не закоммичен или
закоммичен — проверить). UI plot работает корректно, REW WAV broken
на sr=48k.

## Контекст проблемы

UI plot Band 3 HP=2000 sr=48k Min-Phase: Causal 72%, Pre-ring 0.02 ms,
Mag err 0.03 dB — корректно (IIR path активен).

REW WAV на 48k: показывает неверную кривую (notch вместо HP).

`export_fir_wav` либо использует другой impulse чем UI plot, либо
сохраняет правильный impulse но с corruption.

User также пожаловался на отсутствие build increment — невозможно
определить какой билд запущен. **Поэтому в каждом промте теперь — bump.**

## Что нужно сделать

### 1. Bump до b140.7.1

- `src-tauri/tauri.conf.json` → title `"PhaseForge — b140.7.1"`.
- `src-tauri/src/lib.rs` → startup log `b140.7.1`.
- `src/lib/version.ts` (если есть) → b140.7.1.

User должен увидеть **b140.7.1** в title bar после rebuild — это
proof что новый код применён.

### 2. Update CLAUDE.md rule

Добавить (или обновить существующую секцию) о том что **каждый
commit/fix должен bump version** даже на инкрементальный фикс
внутри одного "релиза":

```
## Versioning rule (с b140.7.1)
Каждый commit или применённый патч (включая инкрементальные fix-ы
внутри одной major версии) — обязан bump версию: b140.7 → b140.7.1
→ b140.7.2 etc. Это даёт visual confirmation в title bar что новый
код применён, без неё невозможно отличить пересборку от no-op.
```

### 3. Diag tracing в export_fir_wav

Найти `export_fir_wav` Tauri command:
```
grep -n "export_fir_wav\|fn export_fir" src-tauri/src
```

Добавить лог в начале функции и перед записью WAV:

```rust
tracing::info!(
    "[EXPORT WAV DIAG] entry: sr={}, taps={}, impulse[0..5]={:?}, peak_abs={}, sum={:.6e}",
    sr, impulse.len(), &impulse[0..5.min(impulse.len())],
    impulse.iter().fold(0.0f64, |m, &v| m.max(v.abs())),
    impulse.iter().sum::<f64>()
);
```

Также найти точку где impulse попадает в `export_fir_wav` от frontend.
Если frontend передаёт impulse — то логировать тот же impulse в TS
**перед** invoke (`band-evaluator.ts` или `lib/fir-export.ts`):

```ts
const impulsePreview = impulse.slice(0, 5);
const peakAbs = Math.max(...impulse.map(Math.abs));
const sum = impulse.reduce((a, b) => a + b, 0);
console.log(
  `[EXPORT WAV DIAG TS] sr=${sr}, taps=${impulse.length}, ` +
  `impulse[0..5]=${JSON.stringify(impulsePreview)}, ` +
  `peak_abs=${peakAbs}, sum=${sum.toExponential(6)}`
);
await invoke("export_fir_wav", { /* ... */ });
```

### 4. Diag tracing в IIR PATH output

В `src-tauri/src/fir/iir_path.rs::generate_min_phase_fir_iir` после
вычисления impulse добавить **тот же формат** лога:

```rust
tracing::info!(
    "[IIR PATH DIAG] output: sr={}, taps={}, impulse[0..5]={:?}, peak_abs={}, sum={:.6e}",
    sr, impulse.len(), &impulse[0..5.min(impulse.len())],
    impulse.iter().fold(0.0f64, |m, &v| m.max(v.abs())),
    impulse.iter().sum::<f64>()
);
```

### 5. Также startup marker

Добавить в `lib.rs` startup:
```rust
tracing::info!("[DIAG ACTIVE] export-wav: tracing impulse data flow IIR → WAV");
```

### 6. Запуск и сбор данных

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
- В title bar теперь **PhaseForge v0.1.0-b140.7.1** — это подтверждение
  что новый код применён.
- В первой строке log: `[DIAG ACTIVE] export-wav: ...` — тоже подтверждение.
- Проинструктировать:
  1. Открыть проект, sr=**48k**, Band 3 HP=2000.
  2. Перейти на **Export** tab — это вызовет `[IIR PATH DIAG] output: ...`.
  3. Нажать **Export WAV**, сохранить файл — это вызовет
     `[EXPORT WAV DIAG TS] ...` и `[EXPORT WAV DIAG] entry: ...`.
  4. Скопировать **все** строки `[*DIAG*]` из лога.

### 7. Что НЕ делать

- Не патчить код DSP — только diagnostic tracing.
- Не коммитить (хотя bump — нет, bump коммитится отдельно после verify).
- Не запускать REW тесты до диагностики.

### 8. End-of-prompt

Уже включено в шаг 6.

## Acceptance

- В title bar — b140.7.1.
- В startup log — `[DIAG ACTIVE] export-wav: ...`.
- На Export tab переключении — `[IIR PATH DIAG] output: ...`.
- На Export WAV нажатии — `[EXPORT WAV DIAG TS]` + `[EXPORT WAV DIAG]`.

Сравнение трёх блоков покажет:
- Если `[IIR PATH DIAG]` impulse[0..5] и peak == `[EXPORT WAV DIAG TS]`
  и `[EXPORT WAV DIAG]` — значит правильный impulse передаётся в WAV,
  bug где-то в encoding/storage.
- Если они различаются — значит export_fir_wav генерирует другой
  impulse, IIR path не используется при экспорте.

## Правила

- Без нарратива.
- Bump обязателен.
- Diagnostic tracing — в трёх точках для контраста.

# Промт для Code: export_fir_wav routes через IIR path

**Тип:** bug fix продолжение b140.7. Bump не нужен (та же версия,
inкрементальный фикс), коммит после verify.

## Step 0

```
cd /Users/olegryzhikov/phaseforge
git status
```

Должно быть clean: variant (B) откачен, b140.7 IIR path не закоммичен
(текущие изменения от предыдущего промта). Проверить:
```
git log --oneline -5
git diff --stat
```

Видно modified файлы: tauri.conf.json, lib.rs, version.ts, fir/types.rs,
fir/mod.rs, fir/helpers.rs (если касались), новый fir/iir_path.rs,
band-evaluator.ts, и тесты.

## Контекст

UI plot показывает корректный Min-Phase FIR через IIR path
(`[IIR PATH]` строки в логе). WAV экспорт через `export_fir_wav` — НЕ
использует IIR path (отдельная команда), идёт по старому FFT-cepstral
пути → REW показывает broken response на sr=48k.

Логи подтверждают:
```
04:49:00 [IIR PATH] sr=48000 LP=200 peak_idx=227    ← UI plot
04:50:54 export_fir_wav: 65536 samples, sr=48000    ← WAV без IIR
```

## Что нужно сделать

### 1. Найти export_fir_wav

```
grep -n "export_fir_wav\|fn export_fir" src-tauri/src
```

Это Tauri command. Прочитать end-to-end какой FIR pipeline вызывает.

### 2. Routing fix

Внутри `export_fir_wav`: применить ту же routing логику что и в
`generate_model_fir`:

```rust
let iir_applicable = !linear_phase_main
    && phase_mode == PhaseMode::Composite
    && hp_is_iir_realizable(&hp)
    && lp_is_iir_realizable(&lp)
    && subsonic_cutoff_hz.is_none();

let impulse: Vec<f64> = if iir_applicable {
    let iir_in = IirPathInput { /* ... */ };
    let iir_out = crate::fir::iir_path::generate_min_phase_fir_iir(&iir_in)?;
    iir_out.impulse
} else {
    // existing FFT path для не-IIR случаев
    /* legacy generate_model_fir call */
};
```

Затем `impulse` сохраняется в WAV как раньше.

**Альтернатива**: если `export_fir_wav` уже принимает готовый `impulse`
от frontend (через IPC payload), то проблема в frontend-side — TS
вызывает старый `generate_model_fir` для WAV экспорта вместо нового
IIR path. Тогда фикс в TS.

Посмотреть и определить где именно решается какая команда вызывается.

### 3. Проверить frontend

```
grep -n "export.*wav\|exportWav\|saveAsWav" src/
```

Найти кнопку Export WAV handler. Посмотреть какой invoke делает.
Если invoke `generate_model_fir` (старый) → переделать на использование
результата от `evaluateBandFull` (который уже использует IIR).

### 4. Cargo + vitest

```
cd src-tauri && cargo test --lib 2>&1 | tail -10
cd /Users/olegryzhikov/phaseforge && npm run test 2>&1 | tail -10
```

Должно быть 185+ cargo / 104+ vitest PASS.

### 5. UI verify

После rebuild — Export WAV на Band 3 HP=2000 sr=48k. В логе должно
появиться `[IIR PATH]` ровно перед `export_fir_wav`. WAV в REW —
phase должна совпадать с моделью.

```bash
# Тестирование
date '+%H:%M:%S' > /tmp/last_check.txt
# нажать Export WAV в приложении
awk -v t="$(cat /tmp/last_check.txt)" '$0 > t' /tmp/phaseforge-dev.log
```

В выводе должны быть строки `[IIR PATH]` + `export_fir_wav` подряд.

### 6. Commit (только после UI verify PASS)

```
git add -A
git commit -m "$(cat <<'EOF'
feat: IIR-based min-phase FIR pipeline + WAV export routing (b140.7)

Replace FFT/cepstral min-phase reconstruction with analytical IIR
cascade for non-Gaussian Min-Phase user choice. Pipeline:
analog filter design → bilinear → digital biquads → cascade impulse
→ truncate to N taps. Peak-at-0 by construction (analog poles in
LHP = min-phase physically).

Scope: LR/BU HP+LP and PEQ biquads, Min-Phase user choice
(linear_phase_main=false), no subsonic protect, no Composite mode.
FFT path retained for Linear-Phase, Composite with subsonic,
Gaussian, Bessel, and Custom measured targets.

WAV export (export_fir_wav) routed through same IIR path when
applicable — fixes REW phase mismatch on sr=48k WAV exports.

Cause of REW phase mismatch in min-phase FIR exports: cepstral on
sparse linear FFT grid created high-quefrency artifact. IIR cascade
is the natural representation for min-phase from analytical filter
specs.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

### 7. End-of-prompt

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

## Что НЕ делать

- Не менять IIR path сам.
- Не bumping (b140.7 уже).
- Не коммитить если REW phase ещё не совпадает с моделью на 48k.
- Не отделять export_fir_wav как самостоятельную задачу — это часть
  variant IIR который должен покрывать **и UI и WAV** одновременно.

## Acceptance

- `[IIR PATH]` в логе при Export WAV на Band 1/2/3 sr=48k и sr=176.4k
  (Min-Phase mode).
- WAV в REW: phase совпадает с моделью на всём диапазоне.
- 185+ cargo + 104+ vitest PASS.
- Commit с co-author.

## Правила

- Без нарратива.
- Один report после verify.

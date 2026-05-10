# Промт для Code: variant (B) pre-flight audit + plan

**Тип:** read-only audit + plan documentation. Без bump, без коммита,
без правок DSP. После плана — STOP, ждём user-а для ревью перед
implementation.

## Step 0

```
cd /Users/olegryzhikov/phaseforge
git status
git diff --stat
```

Должно быть clean (post-shift experiment откачен). Если modified —
STOP, сообщить что осталось.

## Цель

Variant (B) — analytical phase pipeline:
- TS computes target+peq+subsonic phase analytically on log grid (5..40k)
  через `reconstructTargetPhase` (как для SPL — там фаза корректна).
- Resample log→linear FFT grid с phase unwrapping.
- Pass в Rust как fixed `model_phase`.
- Rust: пропустить `composite_phase_inner` cepstral.
- iterative_refine: phase фиксирована (analytical truth), refinement
  только magnitude.

Цель audit-а — точно понять текущий поток и составить **минимальный план
изменений** без блайнд-патчинга.

## Что нужно сделать

### 1. Audit TS side

Прочесть `src/lib/band-evaluator.ts` секции:
- Построение `firTargetPhase` (через `reconstructTargetPhase` на firFreq).
- Построение `firPeqPhase` (через `compute_peq_complex`).
- Subsonic phase — где и как добавляется (проверить что
  `reconstructTargetPhase` его учитывает).
- `firCombinedPhase = firTargetPhase + firPeqPhase` — какая суммарная
  размерность и тип grid (log/linear).
- Что именно передаётся в `generate_model_fir` как `modelPhase`.

В отчёт записать: какая фаза уже передаётся, на каком grid, и что
из неё реально доходит до IFFT в Rust.

### 2. Audit Rust side

Прочесть end-to-end:
- `src-tauri/src/fir/mod.rs::generate_model_fir` — путь от приёма
  `model_phase` параметра до IFFT.
- `src-tauri/src/fir/helpers.rs::composite_phase_inner` — что именно
  делает с `model_phase`, recomputes ли через Hilbert.
- `src-tauri/src/fir/helpers.rs::iterative_refine` — как обновляет
  phase per iter, использует ли `model_phase` или recomputes.

В отчёт записать **точно**: где Rust игнорирует переданный `model_phase`
и пересчитывает через cepstral. Какие call sites нужно поменять.

### 3. Subsonic phase coverage

Subsonic protect filter (Composite mode) — important edge case.
Проверить:
- В `reconstructTargetPhase` (TS) — учитывается ли `subsonic_cutoff_hz`
  при reconstruction phase?
- Если нет — subsonic phase computed только в Rust (через cepstral) —
  значит для variant (B) нужно добавить subsonic phase в TS pipeline.

Записать в отчёт: subsonic phase coverage status и что нужно добавить.

### 4. Plan документ

Создать `docs/variant-b-plan.md` со структурой:

```
# Variant (B) Plan: analytical phase pipeline

## Текущая архитектура
- TS строит analytical phase: <details>
- TS передаёт modelPhase в Rust: <where, on what grid>
- Rust в Composite + min-phase main path: <action — cepstral recompute / passthrough>

## Целевая архитектура
- TS computes analytical phase including subsonic on log grid: <details>
- TS resamples log → linear FFT grid: <method, phase unwrap>
- Rust: использует passed model_phase как canonical, skip cepstral
- iterative_refine: phase static, mag only

## Изменения по файлам
1. src/lib/band-evaluator.ts:
   - <конкретные изменения>
2. src-tauri/src/fir/helpers.rs:
   - <конкретные изменения>
3. src-tauri/src/fir/mod.rs:
   - <конкретные изменения>
4. <тесты>:
   - <обновить какие, добавить какие>

## Golden snapshots для regression check
- Список fixtures и метрик для capture перед refactor:
  - For Band 1 LP=200 sr=48k: realized_mag hash, impulse[0..100] hash
  - For Band 2 BP 200-2000 sr=48k: то же
  - For Band 3 HP=2000 sr=48k: то же
  - sr=176.4k для всех трёх

## Risk analysis
- Refinement convergence без phase update — может ли расходиться
  на больших mag errors? (учитывая что Band 3 уже расходится в b140.6)
- Phase resample log→linear — boundary effects на низких freq и Nyquist
- Subsonic phase — если не покрыт TS-side, нужна расширение pipeline

## Acceptance после implementation
- Pre-ring=0, Causal=100% на всех трёх полосах.
- Model° == FIR° на Export plot.
- REW-проверка экспортированного WAV — фаза совпадает с моделью.
- 179+ cargo + 104 vitest PASS, golden snapshots match.
```

### 5. STOP

После сохранения плана — **сообщить user-у** что план готов, ждать
ревью. **Никаких правок DSP / refactor до подтверждения**.

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

Должно работать чистое b140.6 без diagnostic.

## Что НЕ делать

- Не менять DSP код.
- Не bumping.
- Не запускать тесты (только аудит).
- Не предлагать другие варианты — только variant (B) план.
- Не теоретизировать — только что прочитал в коде.

## Правила

- Без нарратива в чате.
- Только цитаты строк кода с file:line.
- Один report (план).

# Промт для Code: rollback variant (B), вернуться на b140.6

**Тип:** rollback. Без commit (variant B не закоммичен), отбросить
изменения.

## Step 0

```
cd /Users/olegryzhikov/phaseforge
git status
```

Должно показать modified файлы variant (B): tauri.conf.json, lib.rs,
version.ts, fir/types.rs, fir/mod.rs, fir/helpers.rs,
src/lib/band-evaluator.ts, и тесты.

## Действие

```
git checkout -- src-tauri/ src/
git clean -fd src-tauri/tests/golden_b140_6_hashes.rs 2>/dev/null || true
git status   # должен быть clean
```

Если что-то осталось modified — сообщить, разобрать вручную.

Версия должна вернуться к b140.6:
```
grep version src-tauri/tauri.conf.json
grep b140 src-tauri/src/lib.rs
```

## Документировать failure

Создать `docs/variant-b-failure-analysis.md`:

```
# Variant (B) failure — analytical phase pipeline (rolled back)

## Что было сделано
b140.7 implementation: TS computes phase analytically on log grid,
resamples log→linear FFT grid (unwrap+interp+wrap, DC=0), passes
to Rust as modelPhaseLinear with use_provided_phase=true. Rust skips
composite_phase_inner cepstral, uses passed phase directly.

## Тестовый результат на production (sr=48k, Band 1 LP=200 LR4)
- Pre-ring: 243.65 ms (b140.6 baseline: 4.25 ms — RegRESS).
- Causal: 46% (no improvement).
- Mag err: 18.69 dB (b140.6: 0.01 dB — BROKEN).
- GD ripple: 562.53 ms.
- Phase warp visually огромный.

## Math root cause
Для discrete IFFT(mag · exp(j·phase)) → real-valued causal impulse
с peak-at-0 необходимо: **phase = Hilbert(log_mag) на ТОЙ ЖЕ
linear FFT grid**. Любая другая фаза даёт complex impulse → real part
теряет imaginary energy → magnitude corrupts on round-trip.

Analog continuous phase (от target::evaluate, LR4 rational TF) ≠
discrete Hilbert(log_mag). Resample log→linear не превращает analog
в discrete-Hilbert-equivalent. Это fundamental DSP fact, не
implementation bug.

## Урок
SPL phase (визуально корректна) — это analog phase, plotted as is.
Не значит что её можно использовать как генератор impulse через IFFT.
Для FFT-based FIR generation cepstral на linear grid — **обязателен**;
вопрос только в смягчении его artifacts (lifter / floor / smoothing).

## Альтернативы для будущей работы
1. **Lifter** (cepstrum windowing): умножать c_min[n] на
   1/(1+(n/n_lifter)^2) перед FFT. Подавляет high-quefrency leak.
2. **Pre-smooth lin_mag** (1/12 oct) только перед cepstrum step —
   blurs cliff, сохраняет mag для assemble_complex.
3. **Analog IIR → bilinear → truncated FIR** (REPhase архитектура).
   Большой refactor, заменяет FFT-based FIR generation на IIR-based.
4. **Time-domain min-phase reconstruction** (ARMA spectral
   factorization) — тоже большой refactor.

Нет приоритета без дополнительной диагностики и обсуждения.
```

## Verify

```
cd src-tauri && cargo test --lib 2>&1 | tail -5
```

Должно быть **179 PASS** (b140.6 baseline).

```
cd /Users/olegryzhikov/phaseforge && npm run test 2>&1 | tail -5
```

Должно быть **104 PASS**.

## End-of-prompt

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

Startup должен показать b140.6.

## Что НЕ делать

- Не коммитить failure-analysis перед обсуждением (просто файл).
- Не пытаться частично сохранить variant (B) код.
- Не предлагать другой fix в этом промте.

## Правила

- Без нарратива.
- Подтвердить tests + версия после rollback.

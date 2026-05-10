# Промт для Code: Min-Phase FIR имеет pre-ringing — read end-to-end

**Тип:** code investigation. Без bump, без коммита, без патча.

## Evidence (уже подтверждено)

Min-Phase FIR в b140.6 имеет pre-ringing:
- Band 1 (LP=200 LR4): Pre-ring 4.25 ms, Causal 51%, sr=176.4k.
- Band 3 (HP=2000 LR4): Pre-ring 0.06 ms, Causal 78%, sr=176.4k.
- Truly min-phase: Pre-ring=0 ms, Causal=100% (peak в samples[0],
  decay только вперёд).

User проверил в REW: фаза экспортированного FIR не совпадает с
моделью, "закручивается вверх по частоте". Это constant group delay
= linear-phase component, ровно соответствует pre-ring 4.25 ms
смещению пика от начала.

Эталон: REPhase. Там min-phase FIR exporting работает корректно
(фаза совпадает с моделью).

## Гипотеза

С b140.1 добавлен Composite phase mode. Возможно:
1. User Min-Phase choice идёт по Composite path с linear-phase
   реконструкцией.
2. Импульс центрирован к середине taps (как в linear-phase) и не
   сдвигается к началу при Min-Phase.
3. composite_phase_inner неправильно собирает фазу при выключенном
   linear_phase_main.

## Что нужно сделать

### Прочитать END-TO-END (без правок)

1. **`src-tauri/src/fir/mod.rs:generate_model_fir`** — целиком,
   проследить путь для Composite + linear_phase_main=false (это
   user-selected Min-Phase).
2. **`src-tauri/src/fir/helpers.rs:iterative_refine` +
   `composite_phase_inner`** — что именно записывается в
   target_phase для Min-Phase user-mode.
3. **`src-tauri/src/dsp/impulse.rs:compute_impulse_response`** — peak
   finding и reordering. Конкретно: после IFFT куда становится peak —
   в samples[0] или в середине? Для Min-Phase должен быть в
   samples[0].
4. **`src/lib/band-evaluator.ts`** — что передаётся в
   `linear_phase_main` и `phase_mode`. Когда Min-Phase user choice +
   subsonic enabled — какой phase_mode идёт в Rust?

### Что искать

- Условие, где Min-Phase user choice конвертируется в Composite mode.
- Место, где target_phase для FIR generation становится 0 (linear)
  вместо Hilbert(target_mag) (min-phase).
- Reordering / peak shift в compute_impulse_response — применяется ли
  он одинаково для всех phase_mode или только для LinearPhase.

### Что вернуть

Краткий report (markdown файл `docs/min-phase-trace.md` или ответом
в Cowork):

1. Полный путь от `phase_mode` в evaluateBandFull → invoke ->
   Rust path → IFFT → peak position. Для Min-Phase user choice
   с subsonic OFF и subsonic ON.
2. Где импульс не получает peak-at-start (если так).
3. Конкретные line numbers, без догадок про "почему".
4. Если bug очевиден из чтения — описать, но **НЕ патчить**.

### Что НЕ делать

- Не запускать тесты.
- Не менять код.
- Не bumping.
- Не коммитить.
- Не запускать dev (read-only сессия).

## Правила

- Без нарратива.
- Только цитаты строк кода с указанием file:line.
- Один report.

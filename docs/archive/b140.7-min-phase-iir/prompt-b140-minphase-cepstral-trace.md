# Промт для Code: read minimum_phase_from_magnitude end-to-end

**Тип:** code investigation. Без bump, без коммита, без патча.

## Step 0 — проверка

```
cd /Users/olegryzhikov/phaseforge
git diff --stat
```

Если есть modified файлы помимо `[DIAG-MP]` патча — STOP, обсудить.

Также: текущий `[DIAG-MP]` патч не добавил строку `[DIAG ACTIVE] MP-peak`
в startup log (`src-tauri/src/lib.rs`) рядом с версией. По правилу
CLAUDE.md "Diagnostic patches — обязательный маркер (с b140.6)" она
обязательна. Добавить одну строку:

```rust
tracing::info!("[DIAG ACTIVE] MP-peak: post-IFFT/window/iter peak idx + active_bins stats");
```

в `lib.rs` startup. Без bump, без коммита.

## Контекст

Diagnostic data подтвердила:

| Band | active_bins% | post-IFFT peak idx | causal |
|---|---|---|---|
| LP=200 | 7.2% | 204 | 51% |
| BP 200-2000 | 72.0% | 32 | 54% |
| HP=2000 | 94.2% | 2 | 92% |

Корреляция: меньше active bins → дальше peak от 0. Peak установлен
initial IFFT (post-IFFT) и не меняется window+iter — refinement не
лечит.

Гипотеза: `minimum_phase_from_magnitude` использует наивный
Hilbert(log_mag) вместо корректного cepstral folding для min-phase
reconstruction. На sparse spectra (большие плоские noise_floor zones)
Hilbert от плоского log_mag даёт constant group delay в этих зонах
→ impulse shifted from idx=0.

Эталонный алгоритм cepstral для min-phase:
1. lin_mag → log_mag (с floor против log(0)).
2. cepstrum = real(IFFT(log_mag)) — real cepstrum.
3. Fold к causal: c[0] остаётся, c[N/2] остаётся, c[1..N/2-1] *= 2,
   c[N/2+1..N-1] = 0.
4. log_H_min = FFT(folded_cepstrum) — даёт log_mag + j*min_phase.
5. H_min = exp(log_H_min); impulse = real(IFFT(H_min)) — peak at idx=0
   by construction.

## Что нужно сделать (read-only)

### 1. Найти minimum_phase_from_magnitude

```
grep -rn "fn minimum_phase\|minimum_phase_from\|min_phase_from" src-tauri/src
```

### 2. Прочитать функцию end-to-end

Скопировать в отчёт `docs/min-phase-cepstral-trace.md`:
- Полный код функции с line numbers.
- Краткая аннотация каждого шага: что делается, на каком домене.
- Сравнение с cepstral эталоном выше: какие шаги совпадают, какие
  отличаются, какие отсутствуют.

### 3. Найти все вызовы

```
grep -rn "minimum_phase_from_magnitude\|min_phase_from_magnitude" src-tauri/src src/
```

Перечислить call sites с контекстом (что передаётся: log_mag или
lin_mag, какой floor, какой N).

### 4. Bonus — проверить расхождение iterative_refine на Band 3

В логе `iterative_refine: iter=1, max_err=22.777 → iter=3, max_err=31.113`
для HP=2000 sr=48k. Расходится. Прочесть iter loop в helpers.rs,
описать в одном абзаце почему может расходиться при initial impulse
с peak idx=2 (близко к нулю но не точно 0).

### 5. Что вернуть

`docs/min-phase-cepstral-trace.md`:

1. Текущая реализация `minimum_phase_from_magnitude` — code + аннотация.
2. Diff vs cepstral эталон — bullet list.
3. Call sites — list.
4. Iterative_refine divergence на Band 3 — короткая гипотеза.

Без правок кода. Без bump. Без тестов.

## Что НЕ делать

- Не патчить DSP.
- Не запускать тесты.
- Не bumping.
- Не коммитить.
- Не теоретизировать про fix — только описать что есть и что должно быть.

## End-of-prompt (обязательно по CLAUDE.md)

После записи отчёта и добавления `[DIAG ACTIVE]` маркера в lib.rs:

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

Сообщить путь к логам — `/tmp/phaseforge-dev.log` — и подтвердить что
в startup появилась строка `[DIAG ACTIVE] MP-peak`.

## Правила

- Без нарратива в чате.
- Только цитаты строк кода с file:line.
- Один report.

# PhaseForge — Project Rules

> **Last reviewed:** 2026-06-11 (after b141.13 release: audit fixes, BandEvalResult cache, IR/Step conventions, UI batch).
> Файл актуализируется в конце каждой длинной сессии. Перед началом новой
> — пробежать сверху вниз и удалить устаревшее.

## Project Architecture
- **Stack**: Rust/Tauri 2 backend + SolidJS/TypeScript frontend. Do NOT assume Python/numpy for DSP code.
- **Versioning**: bNN (b89, b89.1, etc.) — increment on each release
- **Key dirs**: `src-tauri/src/` (Rust DSP, FIR, target eval), `src/components/` (SolidJS UI), `src/stores/` (state), `src/lib/` (helpers)
- **Version files**: `src-tauri/tauri.conf.json` (version + title), `src-tauri/src/lib.rs` (startup log), `src/lib/version.ts`

## DSP conventions
- **Linkwitz-Riley**: PhaseForge LR-N response = `(BU-N)² = 12N dB/oct`
  (`target/mod.rs::filter_lp_response::LinkwitzRiley` doubles BU-N
  magnitude in dB and phase). Internally consistent and REPhase-validated;
  textbook "LR-N = N total order" convention is NOT used here. UI Slope
  dropdown reflects the actual slope (LR4 → 48 dB/oct, b140.7.13).
- **FIR pipeline routing** (`src/lib/band-evaluator.ts`):
  - **IIR analytical cascade** (`fir/iir_path.rs`) — Min-Phase main +
    non-Gaussian (LR / Butterworth / Custom HP+LP + PEQ peaking) +
    no subsonic. Peak-at-0 by construction, REPhase-parity tested.
  - **FFT cepstral path** (`fir/mod.rs::generate_model_fir`) — every
    other configuration: Linear-Phase main, Composite + subsonic,
    Gaussian, Bessel, Custom measured targets.
- **Alignment delay convention** (b141.10): положительное значение =
  полоса ИГРАЕТ ПОЗЖЕ (паритет с HQPlayer). Единая точка —
  `alignmentPhaseDeg` (types.ts, −360·f·τ); legacy-проекты мигрируются
  при загрузке (`migrateDelayConvention`). Менять знак где-либо
  точечно — запрещено.
- **Golden baselines платформо-зависимы** (b141.13): хэши FIR бит-точные,
  libm (tan/cos) различается macOS↔Linux. golden_fir скипается на чужой
  ОС, pipeline_contract сравнивает только routes. Re-baseline: удалить
  json и прогнать тест дважды.
- **WAV peak convention** (b141.14): ЕДИНАЯ — все пути кладут пик у N/2.
  Cepstral min-phase получил тот же адаптивный сдвиг
  `min(N/2, n-1-last_significant)`, что и IIR-путь (cepstral.rs, после
  realized-анализа — кривые плота не сдвигаются). Toast-warning о
  смешении конвенций удалён. Идея «генерация на 2N» оказалась не нужна:
  финальный файл всё равно N тапов, комната для хвоста при пике N/2 —
  ровно N/2, адаптивный сдвиг честнее (контент важнее центрирования).
  Acceptance: `tests/wav_peak_convention.rs` (пик у N/2 на всех маршрутах).
- **Bilinear digital cascade** (IIR path) has frequency-dependent
  deviation up to ~20° vs analog reference accumulated over 8 biquads.
  REPhase reference comparison gives tighter empirical bound (≤ 2.5° on
  sr=44.1k). UI plot guardrail uses 25° tolerance — hard fail signals a
  real regression beyond bilinear math.

## Debugging Rules
- **NEVER guess at root cause.** Always add diagnostic logging first, get user's console output, then fix based on evidence.
- Do not attempt more than 2 fix iterations without diagnostic data.
- Before fixing chart/rendering bugs, read the relevant render function END TO END — don't patch blindly.
- SolidJS gotcha: signals inside async functions are NOT tracked by createEffect. Read them synchronously before any await.
- **Cascade detection.** If the same fix is needed in 3+ places (duplicate copy functions, parallel pipelines, mirror tests) — STOP, surface the architectural duplication to the user before pointwise patches. One refactor beats five cascaded fixes.

## Adding fields to shared structures
- Before adding a field to a struct (e.g. FilterConfig, BandState): grep for ALL functions that copy this struct field-by-field. Known duplicate sites must be enumerated in the prompt and updated in the same commit.
- Copy sites in this repo (актуально с b141.6): `cloneFilterConfig` (lib/types.ts — single source of truth; все прод-сайты идут через него), `captureOptimizedTarget` (stores/peq-optimize.ts), test mirrors (FilterBlock.test.tsx, bands.test.ts), site re-implementations в lib/__tests__/filter-clone.test.ts.
- Forgetting one site = silent loss of the new field across part of the pipeline. Caught only via UI testing on real workflow, expensive.

## Testing / Verification
- After every fix, verify that existing functionality on OTHER tabs/views still works.
- Fixes to one component must not regress others (Export fix must not break IR, snapshot fix must not break legends).
- Before committing, mentally trace: "what else reads/writes this state?" — if unsure, grep for the variable name.
- **Repro tests must use production parameters, not synthetic.** Copy ALL parameters from the user's logs (freq grid, taps, sample rate, narrowband_limit, etc.) — not simplified defaults. A test that uses `5–40 kHz` instead of the user's `20–20 kHz` may pass while the bug persists.
- **Golden snapshots before refactor.** Before any architectural refactor: capture golden snapshots / hashes of current outputs for the canonical fixtures. Only way to prove "no behaviour change" on each stage.

### DSP testing strategy (lesson from b140.7.x — 14+ iteration cycle)
When a DSP bug has visual manifestation (REW, Plot, etc.) but the
in-code behaviour reads correct:

1. **External reference tests beat UI verify cycles.** Build a fixture
   directory (`test-fixtures/<reference-tool>/`) with WAV/snapshot
   output from an industry tool the user trusts (REPhase, Acourate,
   REW). Cargo integration test compares PhaseForge output bin-by-bin
   at key passband freqs. Tolerances calibrated to the math floor
   (e.g. bilinear vs analog: ~3° passband). PASS → DSP correct,
   problem is elsewhere (renderer, encoder, reader).
2. **Test fixtures gitignored** — large binary files. Tests skip
   cleanly when missing so CI / clean checkouts still pass.
3. **Subjective UI verify is a tie-breaker, not the primary signal.**
   Once REPhase tests PASS, treat user "still looks wrong" as a UI /
   reader question, not a DSP regression.
4. **Internal guardrail tests use loose tolerances** (20–25°). They
   catch math broken beyond bilinear deviation, not bilinear vs
   analog mismatch which is fundamental.
5. **Iteration pacing**: if 3+ rounds of "fix → user says still
   broken", STOP and demand external-reference fixtures. Continuing
   without them is the entry to a 10-iteration cascade (see
   `docs/archive/b140.7-min-phase-iir/`).

## Workflow
- Commit after EACH completed block of changes — not after a giant batch.
- Format: `type: description (bNN)` — feat/fix/cleanup/chore
- Run code review (7-vector audit) before marking a feature complete.
- Co-Authored-By trailer in every commit.

## Versioning rule (с b140.7.1)
Каждый промт / прим. fix / diagnostic patch (даже инкрементальный
внутри одной major версии) — **обязан bump версию**: b140.7 →
b140.7.1 → b140.7.2 etc. И tauri.conf.json title, и lib.rs startup,
и version.ts — все три. Это даёт visual confirmation в title bar
что новый код применён; без bump-а невозможно отличить пересборку
от no-op (пользователь не знает старый бинарник запущен или новый).

## Build & Development
- **Dev**: `cargo tauri dev`
- **Release**: `cargo tauri build` -> launch from `target/release/bundle/macos/PhaseForge.app`
- **NEVER**: `cargo build --release` — it does NOT embed frontend assets correctly
- If white/black screen after build: kill all processes on port 1420, rebuild clean

## End-of-prompt automation (с b140.3.x)
В конце **каждого** промта (включая diagnostic patches без коммита) —
Code-сессия выполняет автоматически (без отдельной просьбы пользователя):

1. **Закрыть ВСЕ варианты запущенного PhaseForge** (dev, release, dmg):
   ```
   osascript -e 'tell application "PhaseForge" to quit' 2>/dev/null || true
   pkill -9 -f -i "phaseforge" 2>/dev/null || true
   pkill -9 -f "tauri dev" 2>/dev/null || true
   pkill -9 -f "tauri-driver" 2>/dev/null || true
   sleep 1
   lsof -ti:1420 | xargs kill -9 2>/dev/null || true
   lsof -ti:5173 | xargs kill -9 2>/dev/null || true
   sleep 1
   ```

2. **Запустить dev в фоне** (для быстрой итерации):
   ```
   cd /Users/olegryzhikov/phaseforge && nohup cargo tauri dev > /tmp/phaseforge-dev.log 2>&1 &
   ```

3. Сообщить пользователю что dev запущен, путь к логам — `/tmp/phaseforge-dev.log`.

Это экономит время на каждом цикле — пользователь сразу видит результат
без ручного перезапуска. Для финальной сборки `.dmg` (release) —
по запросу пользователя, не автоматом.

## Diagnostic patches — обязательный маркер (с b140.6)

Любой временный diagnostic patch (без bump, без коммита) **обязан**
добавить строку в startup log `src-tauri/src/lib.rs` рядом с версией:

```rust
tracing::info!("[DIAG ACTIVE] <тег>: <краткое описание>");
```

Пример: `[DIAG ACTIVE] MP-peak: post-IFFT/window/iter peak idx`.

**Зачем**: версия `bNN` не меняется на diagnostic патчах →
неотличимо «диагностика применена» vs «инкрементальная пересборка
без изменений». Маркер виден в первой же строке после рестарта dev,
путаницы нет.

**Откат**: при удалении диагностики — убрать и маркер.

**В каждый diagnostic промт включать step 0**:
```
cd /Users/olegryzhikov/phaseforge
git diff --stat
```
Должны быть modified файлы в `src-tauri/`. Если пусто — патч не
применён, не запускать build.

## Token Economy (Claude Max subscription)
- **No progress narration.** Don't describe what you're about to do — just do it.
- **No confirmations.** Don't ask "shall I proceed?" — proceed.
- **No summaries after completion.** User can read the diff.
- **Minimal output.** Return only: changed file paths + one-line result. No explanations unless asked.
- **Read only what's needed.** Don't read whole files — grep for the specific function/line first.
- **One task per message.** Don't batch unrelated changes into one response.
- **Errors:** show only last 20-30 lines of output, not full logs.

## Timestamps (Claude Code CLI)
Каждый ответ Claude в CLI начинается с **актуального** timestamp для
разметки работы пользователя. Никаких выдуманных дат.

- **В начале каждого ответа** — выполнить `date '+%H:%M'` через bash и
  использовать вывод. Формат: `[HH:MM]` префикс.
- **Никогда** не писать даты или время "от себя" — только из `date`.
- **При записи в файлы** (CLAUDE.md, changelog, TODO, commit messages
  кроме `(bNN)` маркера) — если нужна дата, сначала `date '+%Y-%m-%d'`,
  иначе — не писать (использовать relative references типа "после b138
  каскада").
- Git commit timestamps авторитетны — для истории смотреть
  `git log --format='%ai %h %s'`.

## Communication style (Cowork side)
- All chat replies in Russian. No internal code identifiers (function names, type names) in user-facing explanations. Refer to UI labels and DSP concepts the user can see.
- One step per response. Do NOT predict the next step ("after this works we'll do X") — wait for the user's confirmation of current step result.
- After a failed fix attempt: STOP, request diagnostic data, do NOT propose a second hypothesis blindly.

## Refactor — when NOT to propose unified entity migration
Из опыта b140.2.x (11 итераций каскада на миграцию SUM через копирование legacy, паритет не достигнут):

- ❌ Legacy DSP функция ≥ 500 строк inline → НЕ предлагать миграцию через копирование legacy hacks. Накопленные неявные deps (avgRef, normalization timing, phase reconstruction depth, display state, separate IR grid) выявляются по одному при visual testing.
- ❌ Visual-only тестируемый код без automated visual regression → каждый fix требует ручную проверку, каскад почти неизбежен.
- ✅ < 200 строк, чистая pure function → unified migration ок.
- ✅ Полное automated test coverage (включая phase, не только magnitude) → миграция безопасна.

**Что РАБОТАЕТ для большого legacy кода (опыт b140.3.x — 8 итераций до паритета):**
- **Start from scratch** с минимальным новым pipeline.
- Физически мотивированные правила (per-band normalize, width-aware excess limiter, target+Hilbert extension), не legacy mimicry.
- Маленькие шаги: одна категория кривых за один промт (target → corrected → measurement → IR).
- Single point of truth: extension/normalization логика в одном месте (evaluateBandFull, не в evaluateSum).
- Honest pipeline: что используется в SUM, то видно на band view (faded для synthesis).

Накопленные emergency hacks legacy не унифицируются — выкидывать и переоткрывать необходимые правила физически.

## Prompt effectiveness tracking
С b140.2.2 каждый промт начинается с самооценки и заканчивается
фактической после результата. Накопленные слабые места:
- Synthetic tests PASS, real FAIL — fixture from real production
  project (`test-fixtures/` gitignored). For DSP — external reference
  WAVs (REPhase / Acourate), see `tests/rephase_compare.rs`.
- Phase domain не покрывается magnitude-тестами — acceptance явно
  проверяет phase где relevant.
- При extension magnitude — phase должна быть согласована (та же
  target phase или Hilbert от extended magnitude).
- Большие промты с N фиксами → partial реализация → каскад.
  Trade-off: малые промты сами по себе создают каскад. Pre-flight
  audit перед промтом ловит false hypotheses.
- **UI-driven debugging spawns cascades.** Если "пользователь видит
  X неправильно" → объективные тесты должны быть до DSP правок.
  Пример: b140.7.x от subjective REW screenshot → 14 итераций без
  acceptance до того как добавили REPhase fixture tests.

## Early architectural detection — главный системный гайд
Перед точечным фиксом в любой DSP/UI функции:

**Триггеры архитектурного сигнала** (любой = STOP, поднять архитектурный вопрос):
1. Один и тот же расчёт делается в нескольких файлах.
2. Добавление поля в структуру требует правки в N местах.
3. Фикс бага требует одинаковой правки в N местах.
4. Тестирование одного сценария PASS, другой FAIL.
5. Пользователь говорит «X раз пересчитываем», «дублирование».

**Действие при триггере:**
1. Описать дублирование явно (где какие функции).
2. Предложить два пути: точечный фикс vs refactor.
3. НЕ патчить пока пользователь не выберет.

**Когда поздно:** если пользователь сам говорит «давайте refactor» — я опоздал на 2-3 итерации. Должен был заметить раньше.

## SolidJS Patterns
- `batch()`: wrap multiple signal updates to prevent intermediate effects
- PEQ drag: debounce via `peqDragging` signal + `setTimeout(150ms)`
- Store proxies: deep clone via `JSON.parse(JSON.stringify(obj))` before passing to async
- Gaussian min-phase: Rust returns 0 phase for Gaussian filters; frontend calls `compute_minimum_phase` (Hilbert) when `linear_phase=false`

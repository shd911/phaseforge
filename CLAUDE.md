# PhaseForge — Project Rules

> **Last reviewed:** 2026-05-07 (after b140.3.x clean SUM rebuild + b140.3.8 divergence fix).
> Файл актуализируется в конце каждой длинной сессии. Перед началом новой
> — пробежать сверху вниз и удалить устаревшее.

## Project Architecture
- **Stack**: Rust/Tauri 2 backend + SolidJS/TypeScript frontend. Do NOT assume Python/numpy for DSP code.
- **Versioning**: bNN (b89, b89.1, etc.) — increment on each release
- **Key dirs**: `src-tauri/src/` (Rust DSP, FIR, target eval), `src/components/` (SolidJS UI), `src/stores/` (state), `src/lib/` (helpers)
- **Version files**: `src-tauri/tauri.conf.json` (version + title), `src-tauri/src/lib.rs` (startup log)

## Debugging Rules
- **NEVER guess at root cause.** Always add diagnostic logging first, get user's console output, then fix based on evidence.
- Do not attempt more than 2 fix iterations without diagnostic data.
- Before fixing chart/rendering bugs, read the relevant render function END TO END — don't patch blindly.
- SolidJS gotcha: signals inside async functions are NOT tracked by createEffect. Read them synchronously before any await.
- **Cascade detection.** If the same fix is needed in 3+ places (duplicate copy functions, parallel pipelines, mirror tests) — STOP, surface the architectural duplication to the user before pointwise patches. One refactor beats five cascaded fixes.

## Adding fields to shared structures
- Before adding a field to a struct (e.g. FilterConfig, BandState): grep for ALL functions that copy this struct field-by-field. Known duplicate sites must be enumerated in the prompt and updated in the same commit.
- Examples of copy sites in this repo: `unwrapFilterConfig` (bands.ts), `unwrapFilter` + `withOverride` (ControlPanel.tsx), `cloneFilter` (project-io.ts), test mirrors (FilterBlock.test.tsx).
- Forgetting one site = silent loss of the new field across part of the pipeline. Caught only via UI testing on real workflow, expensive.

## Testing / Verification
- After every fix, verify that existing functionality on OTHER tabs/views still works.
- Fixes to one component must not regress others (Export fix must not break IR, snapshot fix must not break legends).
- Before committing, mentally trace: "what else reads/writes this state?" — if unsure, grep for the variable name.
- **Repro tests must use production parameters, not synthetic.** When writing a cargo test to reproduce a user-reported bug, copy ALL parameters from the user's logs (freq grid, taps, sample rate, narrowband_limit, etc.) — not simplified defaults. A test that uses `5–40 kHz` instead of the user's `20–20 kHz` may pass while the bug persists.
- **Golden snapshots before refactor.** Before any architectural refactor (multi-stage migration, pipeline unification): capture golden snapshots / hashes of current outputs for the canonical fixtures. They are the only way to prove "no behaviour change" on each stage.

## Workflow
- Commit after EACH completed block of changes — not after a giant batch.
- Format: `type: description (bNN)` — feat/fix/cleanup/chore
- Run code review (7-vector audit) before marking a feature complete.
- Co-Authored-By trailer in every commit.

## Build & Development
- **Dev**: `cargo tauri dev`
- **Release**: `cargo tauri build` -> launch from `target/release/bundle/macos/PhaseForge.app`
- **NEVER**: `cargo build --release` — it does NOT embed frontend assets correctly
- If white/black screen after build: kill all processes on port 1420, rebuild clean

## End-of-prompt automation (с b140.3.x)
В конце каждого промта где есть изменения кода — Code-сессия выполняет
автоматически (без отдельной просьбы пользователя):

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
С b140.2.2 каждый промт начинается с самооценки и заканчивается фактической после результата. Накопленные слабые места:
- Synthetic тесты PASS, реальные FAIL — каждый промт делать fixture на копии реального проекта (`test-fixtures/` в `.gitignore`).
- Phase domain не покрывается magnitude-тестами — acceptance явно проверяет phase где relevant.
- При extension magnitude — phase должна быть согласована (та же target phase или Hilbert от extended magnitude).
- Большие промты с N фиксами → partial реализация → каскад. Trade-off: малые промты сами по себе создают каскад. Найти баланс через pre-flight audit.

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

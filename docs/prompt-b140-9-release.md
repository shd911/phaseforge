# Промт для Code: release b140.9 на GitHub

**Тип:** release. Bump до b140.9 (consolidated release covering
b140.8.1 + b140.8.2 dev iterations).

## Step 0

```
cd /Users/olegryzhikov/phaseforge
git status
git log --oneline -10
```

Должен быть clean working tree, последние commits b140.8.1 и b140.8.2.

## Phase 1 — Bump до b140.9

- `tauri.conf.json` → version `0.1.140`, title `"PhaseForge — b140.9"`.
- `src-tauri/src/lib.rs` startup → `b140.9`.
- `src/lib/version.ts` → `b140.9`.

## Phase 2 — Release notes `docs/release-notes-b140.9.md`

```
# PhaseForge b140.9 — UI slope dropdown + render fixes

Минорный релиз с улучшениями UX и UI fixes после b140.8.

## UI: Order → Slope dropdown (b140.7.13 / в b140.9 stable)
- HP/LP filter UI показывает фактический slope в dB/oct вместо
  abstract "Order N".
- LR slopes: 12, 24, 36, 48, 60, 72, 84, 96 dB/oct (orders 1-8 × 12).
- BU/Bessel/Custom: 6, 12, 18, 24, 30, 36, 42, 48 dB/oct.
- Gaussian использует shape parameter — slope dropdown скрыт.
- Save format JSON не меняется (filter.order persisted), existing
  projects auto-mapped при load.

## Slope dropdown render fixes (b140.8.2)
- Устранён desync при switching bands — dropdown отображал stale
  slope value не соответствующий filter.order. Корень: SolidJS
  attr/children race в `<select value={...}>`. Replaced на
  `<option selected={...}>`.
- Bessel odd slopes (18/30/42 dB/oct) — UI consistent после fix.

## Filter linking edge case (b140.8.2)
- При удалении среднего band-а в проекте с 5+ полосами связи
  HP/LP между соседними полосами сохраняются (раньше cleared
  по error из linkedToNext propagation).
- Fix в assignDefaultTargets: linkedToNext=true применяется до
  measurement-skip continue.

## Tests / infrastructure (b140.8.1, b140.8.2)
- 8 vitest unit tests на slope mapping invariants
  (orderToSlope/slopeToOrder roundtrip across all filter types).
- 3 vitest tests на removeBand сценарии (link preservation).
- Test-logs file workflow для async test verification —
  `.test-logs/` directory, gitignored.

## Project hygiene (b140.8.1)
- Spent prompts и intermediate analysis из b140.7.x перемещены
  в `docs/archive/b140.7-min-phase-iir/`.
- CLAUDE.md обновлён: добавлены DSP conventions section,
  testing strategy section, известные ограничения bilinear
  cascade, lesson "UI-driven debugging spawns cascades".

## Под капотом
- 180 cargo lib + 4 REPhase + 114 vitest tests PASS.
- Build version visible в title bar для verification что
  applied build матчит состояние кода.

## Известные ограничения
- IIR path не покрывает: Gaussian, Composite + subsonic, Custom
  measured targets — все идут на FFT path как раньше.
- Bilinear digital cascade vs analog reference: ~2.5° max phase
  deviation на sr=44.1k (REPhase reference verified). Внутренний
  guardrail тест допускает до 25° (frequency-dependent
  accumulation over biquad cascade).
```

## Phase 3 — Verify tests

```
cd /Users/olegryzhikov/phaseforge
mkdir -p .test-logs
cd src-tauri
cargo test --lib > /Users/olegryzhikov/phaseforge/.test-logs/release-cargo-lib.log 2>&1
cargo test --test rephase_compare > /Users/olegryzhikov/phaseforge/.test-logs/release-rephase.log 2>&1
cd /Users/olegryzhikov/phaseforge
npx vitest run > /Users/olegryzhikov/phaseforge/.test-logs/release-vitest.log 2>&1

# Summary
{
  echo "=== Cargo lib ==="
  grep "test result" .test-logs/release-cargo-lib.log | tail -3
  echo ""
  echo "=== REPhase ==="
  grep "test result" .test-logs/release-rephase.log | tail -3
  echo ""
  echo "=== Vitest ==="
  grep -E "Test Files|Tests" .test-logs/release-vitest.log | head -5
} > /Users/olegryzhikov/phaseforge/.test-logs/release-summary.log
```

Cowork прочитает summary.

## Phase 4 — Commit + push + tag

После Cowork verify summary:

```
git add -A
git commit -m "release: PhaseForge b140.9

Co-Authored-By: Claude <noreply@anthropic.com>"

git push origin main
git tag -a b140.9 -m "PhaseForge b140.9 — UI slope dropdown + render fixes"
git push origin b140.9
```

## Phase 5 — Monitor CI

```
gh run list --limit 3 > /Users/olegryzhikov/phaseforge/.test-logs/ci-status.log 2>&1
```

Cowork прочтёт. После CI PASS — отредактировать GitHub release
description с содержимым `docs/release-notes-b140.9.md`:

```
gh release edit b140.9 -F docs/release-notes-b140.9.md
```

## Phase 6 — End-of-prompt

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

## STOP triggers

- Phase 3 test fail → STOP, не release.
- Phase 4 push fail → user интервенция.
- Phase 5 CI fail → debug.

## Acceptance

- Title bar = b140.9.
- 180+ cargo / 4 REPhase / 114+ vitest PASS.
- Tag b140.9 на GitHub.
- CI PASS обе платформы.
- GitHub Release description заполнен.

## Правила

- Без нарратива.
- Phases по порядку.
- Test logs в `.test-logs/`.

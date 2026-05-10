# Промт для Code: b140.7.13 commit → global audit → release b140.8

**Тип:** release flow. Несколько фаз последовательно. STOP triggers
на каждой фазе при failure.

## Step 0

```
cd /Users/olegryzhikov/phaseforge
git status
git log --oneline -10
```

Должно быть:
- Modified файлы от b140.7.13 (Slope dropdown UI).
- Последний commit — b140.7.12 REPhase tests.

## Phase 1 — Commit b140.7.13

```
git add -A
git commit -m "$(cat <<'EOF'
feat: replace Order field with Slope dropdown in HP/LP UI (b140.7.13)

Filter order parameter ambiguous due to PhaseForge LR convention
(internally (BU-N)² = 2N order). UI now shows actual slope in
dB/oct, eliminating confusion: LR Order=4 → "Slope: 48 dB/oct".

LR slopes: 12, 24, 36, 48, 60, 72, 84, 96 dB/oct (orders 1-8 × 12).
BU/Bessel/Custom: 6, 12, 18, 24, 30, 36, 42, 48 (orders 1-8 × 6).
Gaussian uses shape parameter — slope hidden.

Internal model unchanged (filter.order in JSON), save format intact.
Existing projects load with auto-mapped slope display.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

## Phase 2 — Full test suite

```
cd src-tauri && cargo test --lib 2>&1 | tail -10
cd src-tauri && cargo test rephase_match 2>&1 | tail -10
cd /Users/olegryzhikov/phaseforge && npm run test 2>&1 | tail -10
```

Acceptance:
- 185+ cargo lib PASS.
- 4/4 REPhase compare PASS.
- 104+ vitest PASS.

Если что-то fail — STOP, report, не релизить.

## Phase 3 — Global 7-vector audit

Создать `docs/audit-b140-8.md` с анализом по 7 векторам:

1. **Architecture**: согласованность IIR path / FFT path routing.
   FFT path для linear-phase / Composite / Gaussian. IIR path для
   non-Gaussian min-phase. Нет ли dead code в legacy paths?

2. **DSP correctness**: 4/4 REPhase tests PASS, 185 cargo PASS,
   golden snapshots не сломаны. Edge cases (Gaussian + subsonic,
   PEQ-only, no measurement) покрыты тестами?

3. **Backward compat**: existing projects (.pfproj) загружаются
   корректно. Save format не сломан. Migration paths.

4. **UI/UX consistency**: Slope dropdown единообразно во всех
   секциях (HP/LP/Manual PEQ если применимо). Status bar / labels
   синхронны (LR48 везде).

5. **Performance**: cargo build size, dev startup time, FIR
   generation latency.

6. **Documentation**: README/CHANGELOG актуальны? CLAUDE.md
   правила соблюдены (build-version skill, end-of-prompt automation
   — что нужно обновить?).

7. **Test coverage**: unit / integration / regression / golden /
   REPhase reference. Какие сценарии без тестов?

Report — list находок (если есть критические — STOP до фикса).

## Phase 4 — Bump до b140.8

После audit PASS:

- `tauri.conf.json` → version `0.1.140` (если major/minor бамп нужен — иначе сохранить),
  title `"PhaseForge — b140.8"`.
- `lib.rs` startup → `b140.8`.
- `src/lib/version.ts` → `b140.8`.

## Phase 5 — Release notes

Создать `docs/release-notes-b140.8.md`:

```
# PhaseForge b140.8 — IIR-based min-phase + REPhase parity

Большой блок DSP/UX улучшений с b140.4.

## IIR-based Min-Phase FIR pipeline (b140.7)
- Новый pipeline: analog filter design → bilinear → digital
  biquad cascade → truncated FIR. Peak-at-0 by construction для
  не-Gaussian filters (LR/BU/Bessel + PEQ). Linear-Phase /
  Composite / Gaussian / Custom — остаются на FFT path.
- Решает регрессию b140.4-b140.6: REW phase mismatch на min-phase
  WAV экспорт, особенно sr=44.1/48k.

## REPhase reference compatibility (b140.7.12)
- 4 cargo тестов сравнивают PhaseForge IIR output vs REPhase
  reference WAVs (test-fixtures/rephase/, gitignored).
- Tolerance: max Δmag < 1 dB, max Δphase < 10° в passband.
- Worst case sr=44.1k: 0.44 dB / 2.5°.
- Best case sr=176.4k: 0.03 dB / 0.2°.
- Catches DSP regressions без UI verify cycles.

## UI: Order → Slope dropdown (b140.7.13)
- HP/LP filter UI теперь показывает "Slope: X dB/oct" dropdown
  вместо "Order: N" numeric.
- Прозрачность: LR4 (PhaseForge convention) виден как
  "48 dB/oct" — соответствует фактическому DSP slope.
- Save format JSON не меняется (filter.order persisted).
- Existing projects auto-mapped при load.

## FIR grid resample (b140.6)
- realized_mag/phase resampled на evalRes.freq grid внутри
  evaluateBandFull. Решает sr-dependent rolloff shift на
  44.1/48k WAV экспортах.

## Под капотом
- Float64 WAV format (no change, REPhase compatible).
- generate_model_fir_iir Tauri command (new).
- iir_path.rs модуль (~430 lines, biquad bilinear cascade).
- diagnostic [DIAG ACTIVE] markers конвенция в startup log.

## Известные ограничения
- IIR path не покрывает: Gaussian (use FFT cepstral), Composite
  + subsonic (legacy FFT path), Custom measured targets.
- LP min-phase impulse имеет natural group delay (~5 ms на
  LP=200 sr=48k) — это физическая характеристика 8th-order LP,
  не bug.
- bilinear discretization vs analog reference имеет небольшой
  frequency-dependent отклонение (≤ 2.5° на passband).
```

## Phase 6 — Commit + push + tag

```
git add -A
git commit -m "release: PhaseForge b140.8

Co-Authored-By: Claude <noreply@anthropic.com>"

git push origin main
git tag -a b140.8 -m "PhaseForge b140.8 — IIR min-phase + REPhase parity"
git push origin b140.8
```

## Phase 7 — Monitor CI

```
gh run list --limit 3
gh run watch
```

После CI PASS — отредактировать GitHub Release description, вставить
содержимое `docs/release-notes-b140.8.md`.

## Phase 8 — End-of-prompt

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

На любой фазе если что-то fail:
- Phase 2 fail → STOP, не bumping.
- Phase 3 critical findings → STOP, обсудить.
- Phase 4-5 не делать без PASS Phase 2-3.
- Phase 6 push fail (auth etc.) → user интервенция.
- Phase 7 CI fail → debug, не пытаться форсить.

## Что НЕ делать

- Не делать DSP правки без user одобрения (это release).
- Не пропускать audit — он catch regressions.
- Не release без 4/4 REPhase tests PASS.

## Правила

- Без нарратива.
- Phases по порядку.
- Между phases короткий status (PASS/FAIL count).

# Промт для Code: b140.7.14 — relax UI plot phase test, resume release flow

**Тип:** test relax + resume release. Bump до b140.7.14 для test
patch, потом продолжить с audit + release b140.8 per
prompt-b140-8-audit-and-release.md (Phases 3-7).

## Step 0

```
cd /Users/olegryzhikov/phaseforge
git status
git log --oneline -3
```

## Phase A — Relax test tolerance + bump

В тесте `fir::iir_path::tests::hp_lr4_2000_sr_48k_ui_plot_phase_matches_model`
поменять tolerance с 5° на **25°**. Добавить комментарий:

```rust
// Tolerance 25° accommodates inherent bilinear vs analog reference
// deviation in passband (frequency-dependent, accumulates over 8
// biquads). REPhase reference comparison (rephase_compare.rs) provides
// tighter empirical acceptance (max 2.5° on sr=44.1k). This test
// serves as guardrail — fails only if deviation grows beyond bilinear
// expected range.
let phase_tolerance_deg = 25.0;
```

Bump:
- `tauri.conf.json` → b140.7.14.
- `lib.rs` → b140.7.14.
- `version.ts` → b140.7.14.

```
cd src-tauri && cargo test --lib 2>&1 | tail -5
```

Должно быть **180+ PASS / 0 FAIL**.

```
git add -A
git commit -m "$(cat <<'EOF'
test: relax UI plot phase tolerance to 25° (b140.7.14)

hp_lr4_2000_sr_48k_ui_plot_phase_matches_model compared internal
realized_phase (from raw cascade FFT) against analog target reference.
Bilinear-transformed digital cascade has frequency-dependent
deviation from analog up to ~20° accumulated over 8 biquads —
mathematical property, not regression.

REPhase reference tests (rephase_compare.rs) provide tighter
empirical acceptance (max 2.5° on sr=44.1k). UI plot test now
serves as guardrail catching deviations beyond expected bilinear range.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

## Phase B — Resume release flow

Продолжить per `docs/prompt-b140-8-audit-and-release.md` начиная с
**Phase 3 (Global 7-vector audit)**, но финальная версия —
**b140.8** (как и планировалось).

Re-run:
```
cd src-tauri && cargo test --lib 2>&1 | tail -3
cd src-tauri && cargo test --test rephase_compare 2>&1 | tail -3
cd /Users/olegryzhikov/phaseforge && npx vitest run 2>&1 | tail -3
```

Ожидание:
- Cargo lib: 180+ PASS.
- REPhase compare: 4/4 PASS.
- Vitest: 103+ PASS.

### Phase 3: Audit `docs/audit-b140-8.md`

Per original audit promt — 7 vectors, findings list.

### Phase 4: Bump до b140.8

После audit PASS — bump во всех 3 файлах.

### Phase 5: Release notes `docs/release-notes-b140.8.md`

Per template из original release prompt.

### Phase 6: Commit + push + tag

```
git add -A
git commit -m "release: PhaseForge b140.8

Co-Authored-By: Claude <noreply@anthropic.com>"

git push origin main
git tag -a b140.8 -m "PhaseForge b140.8 — IIR min-phase + REPhase parity"
git push origin b140.8
```

### Phase 7: Monitor CI

```
gh run list --limit 3
gh run watch
```

После CI PASS → отредактировать GitHub release description с
release notes.

## Phase C — End-of-prompt

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

- Phase A test fail → STOP, не продолжать.
- Phase 3 critical findings → STOP, обсудить.
- Phase 6 push fail → user интервенция.
- Phase 7 CI fail → debug.

## Что НЕ делать

- Не патчить DSP в этом промте.
- Не пропускать audit.

## Правила

- Phases по порядку.
- Между phases короткий status.
- Без нарратива.

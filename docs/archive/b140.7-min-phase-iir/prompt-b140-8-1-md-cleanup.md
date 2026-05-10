# Промт для Code: b140.8.1 — cleanup .md files + consolidate lessons

**Тип:** project hygiene. Bump до b140.8.1. Коммит после verify.

## Step 0

```
cd /Users/olegryzhikov/phaseforge
git status
git log --oneline -3
ls -la docs/ | wc -l
ls -la *.md
```

## Цель

После 14+ итераций в b140.7.x накопились spent prompts, intermediate
analysis docs, dead-end traces. Нужно:
1. Проанализировать все .md в проекте.
2. Удалить spent / obsolete (или переместить в `docs/archive/`).
3. Обновить CLAUDE.md (remove stale, add new lessons).
4. Зафиксировать pattern testing system из b140.7.x опыта.

## Phase 1 — Inventory всех .md

```
find . -name "*.md" -not -path "./node_modules/*" -not -path "./target/*" \
  -not -path "./src-tauri/target/*" -not -path "./.git/*"
```

Категоризация (записать в `docs/md-cleanup-plan.md`):

**Keep (active):**
- `CLAUDE.md` — project rules (актуализировать).
- `README.md` (если есть) — public-facing.
- `docs/release-notes-b140.X.md` — release history.
- `docs/audit-b140-8.md` — последний audit.
- `docs/wav-fft-compare.py` — diag tool, активно использован.

**Archive (исторические, переместить в `docs/archive/`):**
- `docs/prompt-b140-*.md` — все spent prompts кроме последних из текущей сессии.
- `docs/min-phase-trace.md`, `docs/variant-b-plan.md`,
  `docs/variant-b-failure-analysis.md` — intermediate analysis.
- `docs/iir-grid-audit-b140-7-3.md`, `docs/postshift-test-report.md`,
  `docs/min-phase-cepstral-trace.md` — sub-investigations.

**Delete:**
- Дубликаты, эмпти файлы.

## Phase 2 — Применить cleanup

```
mkdir -p docs/archive/b140.7-min-phase-iir
git mv docs/prompt-b140-*.md docs/archive/b140.7-min-phase-iir/  # если применимо
git mv docs/min-phase-*.md docs/archive/b140.7-min-phase-iir/
git mv docs/variant-b-*.md docs/archive/b140.7-min-phase-iir/
git mv docs/iir-grid-*.md docs/archive/b140.7-min-phase-iir/
git mv docs/postshift-*.md docs/archive/b140.7-min-phase-iir/
```

(Точные команды зависят от inventory — Code решает.)

В `docs/archive/README.md` добавить index:
```
# Archive

Spent prompts and intermediate analysis docs from past sessions.
Kept for historical reference, not active.

## b140.7 min-phase IIR rebuild (2026-05-09 to 2026-05-10)
- 14+ итераций fix цикла на min-phase FIR via cepstral → IIR.
- Final solution: IIR cascade + REPhase reference tests.
- См. docs/release-notes-b140.8.md for outcome.
```

## Phase 3 — Обновить CLAUDE.md

Прочесть текущий CLAUDE.md end-to-end, выявить:

**Удалить (устарело / решено)**:
- Версия "Last reviewed" → обновить.
- Любые TODO которые выполнены.
- Specifics от b140.3.x SUM rebuild если стабильно работает.
- Detailed b138 cascade lessons — сжать (опыт абсорбирован в general
  cascade detection rule).

**Добавить**:
- **Lesson из b140.7.x**: при DSP багах с визуальным проявлением
  → external reference tests (REPhase / industry standard) как
  primary acceptance, не UI verify. Раздел "DSP testing strategy".
- Pattern: `test-fixtures/<reference-tool>/` для regression suites.
- Bilinear digital cascade vs analog reference — known frequency-
  dependent ~20° deviation в passband. Добавить как known limit.
- PhaseForge LR convention: order=N → 12N dB/oct (после b140.7.13
  UI показывает реальный slope). Документировать.

**Section structure goal** (final CLAUDE.md):
1. Project Architecture.
2. DSP convention notes (LR/BU/Bessel/Gaussian, slope mapping).
3. Debugging Rules (cascade detection, diagnostic-first, test-first).
4. Adding fields to shared structures (с examples).
5. Testing strategy (golden / regression / external reference / production
   params).
6. Workflow (commit conventions, code review).
7. Versioning rule (bump on each commit).
8. Build & Development.
9. End-of-prompt automation.
10. Diagnostic patches markers.
11. Token Economy.
12. Communication style.
13. Refactor — when NOT to migrate (keep).
14. Early architectural detection (keep — главный gid).
15. SolidJS Patterns.

Удалить мелкие explainers где правило не актуально или поглощено
другим.

## Phase 4 — Bump b140.8.1

- `tauri.conf.json` → b140.8.1.
- `lib.rs` → b140.8.1.
- `version.ts` → b140.8.1.

## Phase 5 — Verify

```
cd src-tauri && cargo test --lib 2>&1 | tail -3
cd src-tauri && cargo test --test rephase_compare 2>&1 | tail -3
cd /Users/olegryzhikov/phaseforge && npx vitest run 2>&1 | tail -3
```

Должно остаться 194+ cargo / 4/4 REPhase / 103+ vitest PASS.

UI sanity:
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

Title bar = b140.8.1.

## Phase 6 — Commit

```
git add -A
git commit -m "$(cat <<'EOF'
chore: cleanup .md files and consolidate CLAUDE.md (b140.8.1)

Archive spent prompts and intermediate analysis from b140.7.x
min-phase IIR rebuild session (14+ iterations) into
docs/archive/b140.7-min-phase-iir/.

Update CLAUDE.md:
- Add testing strategy section (external reference tests primary,
  REPhase pattern, test-first for DSP).
- Document PhaseForge LR convention (order=N → 12N dB/oct).
- Add bilinear vs analog deviation as known limit.
- Remove resolved TODOs, condense earlier session lessons.

No DSP changes.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

## Что НЕ делать

- Не удалять release-notes-b140.X.md (история).
- Не менять DSP / tests / UI.
- Не делать без user review CLAUDE.md изменений (если major
  правки → запросить approve).

## Acceptance

- `docs/archive/` создан с README.
- spent prompts перемещены.
- CLAUDE.md обновлён, "Last reviewed" актуальная дата.
- Все tests PASS (без regressions).
- Title bar = b140.8.1.

## Правила

- Без нарратива.
- Если CLAUDE.md изменения существенные — diff показать user-у
  до commit.

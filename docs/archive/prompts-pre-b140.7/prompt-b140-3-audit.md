# Промт для Code: глобальный аудит после b140.3.x series

**Тип:** Read-only audit. Без bump, без коммита, без изменений production.

## Контекст

После 14 фиксов в b140.3.x (rebuild SUM с чистого листа + band IR/Step
unification) — нужен глобальный аудит перед релизом b140.4.

## Что проверить

### 1. Consistency единого источника

Запустить:
```
grep -n "invoke<.*\(evaluate_target\|compute_peq_complex\|compute_cross_section\|compute_impulse\)" src/components/FrequencyPlot.tsx
```

Каждый найденный inline `invoke` классифицировать:
- Должен ли использовать `evaluateBandFull` / `evaluateSum`?
- Это legacy путь (renderSumMode легаси, переключаемый)?
- UX-ускорение (PEQ live drag в строке ~1547) — оправдано.

Цель: список оставшихся inline точек с обоснованием каждой.

### 2. Состояние тестов

```
cd /Users/olegryzhikov/phaseforge
cargo test 2>&1 | tail -20
npm test 2>&1 | tail -30
```

Подтвердить:
- Cargo: 178+ PASS.
- Vitest: ~70+ PASS.
- Какие тесты flaky / ignored.

### 3. Dead code candidates

Проверить функции в `band-evaluator.ts` которые могли остаться от
предыдущих этапов:
- `extensionTargetMag` опция в `resampleOntoGrid` — после b140.3.2
  extension в evaluateBandFull, эта опция может быть unused.
- Любые `// b140.2.x` комментарии указывающие на удалённую функциональность.
- `addGaussianMinPhase` (должна быть удалена в b139.4c).
- Старые snapshot файлы которые не используются.

### 4. Legacy renderSumMode и SUM IR

Подтвердить что legacy переключатель работает:
- `sumModeSignal()` toggle Legacy/New действует.
- При выборе Legacy — старый pipeline для freq и IR.
- При выборе New — `evaluateSum` для freq + IR.

### 5. CLAUDE.md актуальность

Прочитать `/Users/olegryzhikov/phaseforge/CLAUDE.md`:
- `Last reviewed:` дата актуальна?
- Все правила применимы?
- Удалить устаревшие.

### 6. Regression checklist (pre-release)

Прочитать `/Users/olegryzhikov/phaseforge/docs/regression-checklist.md`.
Подтвердить что 5 manual UI пунктов всё ещё валидны для b140.4.

### 7. TODO список

Прочитать `/Users/olegryzhikov/phaseforge/docs/TODO.md` — что ещё
оставлено deferred. Кандидаты:
- sr=None при импорте .txt (с b135).
- Refinement drift в subsonic (с b139.3.4).
- Любые другие.

## Формат отчёта

```
=== Глобальный аудит после b140.3.7 ===

1. Inline invoke (FrequencyPlot.tsx):
   - Line N: invoke X — оправдано / migrate
   - ...

2. Tests:
   - cargo: N PASS / M FAIL / K ignored
   - vitest: N PASS / M FAIL / K skipped
   - flaky: список

3. Dead code:
   - file:line — function N — usage
   - ...

4. Legacy SUM toggle:
   - Legacy: ✓/✗
   - New: ✓/✗

5. CLAUDE.md:
   - Last reviewed: дата
   - Устаревшие разделы: список

6. Regression checklist: 5 пунктов всё ещё валидны: yes/no

7. Open TODO:
   - Item: status

=== Готовность к релизу b140.4 ===

GO / NEEDS-FIX (что фиксить)
```

## Что НЕ делать

- Не менять production код.
- Не bumping версию.
- Не коммитить.
- Не запускать `cargo tauri dev` (не нужно для аудита).

## Правила

- Без нарратива.
- Read-only.
- Если найдены критичные проблемы — заблокировать релиз с явным указанием что фиксить в b140.3.8.

# Промт для Code: b139.2 — миграция SPL view на BandEvaluator

ТЗ целиком: `docs/TZ-unified-evaluation.md` (Этап 2).
Текущий билд: 0.1.0-b139.1 → bump до 0.1.0-b139.2.

## Контекст

В b139.1 создан `src/lib/band-evaluator.ts` параллельно со старым
`evaluateBand`. 12 equivalence тестов доказали что новый и старый
выдают идентичный output для SPL case с точностью 1e-9.

Этот этап — переключить **только SPL plot отдельной полосы** на
новый evaluator. SUM, Export, IR/Step остаются на старых inline
pipeline (это Этапы 3-4). Поведение SPL plot должно остаться
идентичным b139.1 — это и есть критерий что миграция чистая.

## Pre-flight audit (обязательно ДО изменений)

### 1. Найти все callsites `evaluateBand` и связанные inline точки

Запустить и зафиксировать вывод:

```
grep -rn "evaluateBand\|addGaussianMinPhase" src/
grep -n "evaluate_target\|compute_peq_complex" src/components/FrequencyPlot.tsx
grep -n "evaluate_target\|compute_peq_complex" src/components/ControlPanel.tsx
```

Классифицировать каждую точку:
- **SPL plot отдельной полосы** — мигрируется в этом этапе
- **SUM view** — остаётся на старом (Этап 4)
- **Export tab phase preview** — остаётся (Этап 3)
- **IR/Step плот** — остаётся (Этап 4)
- **PEQ optimizer** (peq-optimize.ts) — НЕ дисплей, **не трогать**
- **auto-align** — НЕ дисплей, **не трогать**

Если точка двусмысленная (используется одновременно для SPL и SUM)
— остановиться, описать ситуацию, не делать миграцию.

### 2. Понять как FrequencyPlot потребляет evaluateBand result

Прочитать соответствующие участки FrequencyPlot.tsx — как сейчас
вызывается `evaluateBand`, как result используется (чьи signal'ы,
какие effects пересчитываются). Это нужно чтобы заменить на
`createBandEvalResource` без поломки реактивности.

## Что нужно сделать

### 1. Замена в FrequencyPlot.tsx (только SPL plot)

Для каждой SPL-related точки из audit:

**Старая схема:**
```typescript
const [bandData] = createResource(activeBand, async (band) => {
  return await evaluateBand(band);
});
```

**Новая:**
```typescript
import { createBandEvalResource } from "../lib/band-evaluator";

const bandData = createBandEvalResource(activeBand);
// или с явными опциями если нужно:
// const bandData = createBandEvalResource(activeBand, { freq: () => customFreq() });
```

`bandData()` теперь возвращает `BandEvalResult` (новая структура).
Адаптировать места чтения:
- `bandData()?.measurement` — есть в обеих структурах
- `bandData()?.targetMag` — есть
- `bandData()?.targetPhase` — есть
- `bandData()?.freq` — есть
- Новые поля (peqMag, peqPhase, combinedTargetMag, combinedTargetPhase) —
  использовать только если SPL plot их рисует. Если рисует только
  pure target — игнорировать.

### 2. Что НЕ трогать в FrequencyPlot.tsx

- SUM view секцию (line ~3590 по audit) — оставить как есть.
- Export tab phase preview (line ~1712) — оставить.
- IR/Step plot — оставить.
- PEQ optimizer не трогаем нигде.
- auto-align не трогаем.

### 3. Что НЕ трогать в других файлах

- `src/lib/band-evaluation.ts` — старая `evaluateBand` остаётся для
  callers которые ещё не мигрированы.
- `src/lib/fir-export.ts` — Этап 3.
- `src/stores/peq-optimize.ts` — независимый pipeline.
- `src/lib/auto-align.ts` — независимый.
- Никаких изменений в Rust.

### 4. Тесты

**Snapshot тесты b139.0 и b139.1 не должны измениться.** Проверить
после миграции — они в `src/lib/__tests__/__snapshots__/`.

Дополнительно — добавить **integration тест** что
`createBandEvalResource` ведёт себя так же как ручной вызов
`evaluateBandFull`. Простой smoke-тест без Tauri:

```typescript
// в band-evaluator.test.ts
it("createBandEvalResource produces same output as evaluateBandFull", async () => {
  const band = makeFixtureBand();
  const direct = await evaluateBandFull({ band });
  // resource нужно тестировать через createRoot для SolidJS
  // ... или skip если сложно настроить
});
```

Если интеграционный тест resource в jsdom сложно — пропустить.
Главное что snapshot и equivalence тесты остаются зелёными.

### 5. Bump версии

- `src-tauri/tauri.conf.json` — version + productName/title до b139.2.
- `src-tauri/src/lib.rs` — startup-лог.
- После билда — skill `build-version`.

## Acceptance

1. SPL plot отдельной полосы в `.dmg b139.2` визуально идентичен
   `.dmg b139.1`.
2. Phase plot для всех 4 Gaussian × subsonic комбинаций — идентичен:
   - linear=true + subsonic OFF: phase = 0
   - linear=true + subsonic ON: phase крутится в 5–40 Гц (как в b139.1)
   - linear=false + subsonic OFF: min-phase Gaussian
   - linear=false + subsonic ON: min-phase combined
3. PEQ markers рендерятся корректно.
4. Group delay (если рендерится в SPL view) — корректен.
5. Все existing snapshot и equivalence тесты зелёные.
6. **regression-checklist 10 пунктов на `.dmg b139.2` проходит
   идентично b139.1** — пункт 3 (linear+subsonic phase на SPL)
   работает; пункты для SUM/Export/IR в чеклисте не проверяются (их
   нет в текущем списке).
7. Никаких регрессий на других вкладках (SUM/Export/IR не должны
   измениться, потому что не трогали их код).

## Регрессионная проверка

Особое внимание:
- Реактивность SPL plot при изменении target/измерения/PEQ — должна
  работать через resource как раньше.
- Cmd+Z (b132 history) — резюме pipeline после undo.
- Save/Open (b131) — после загрузки SPL plot корректно отображается.
- Versions restore (b133) — то же.

Visual diff между b139.1 и b139.2 на SPL plot должен быть нулевой
для одинаковых данных.

## Что делать при провале acceptance

Если SPL plot после миграции отличается от b139.1 — НЕ patch'ить
вслепую. Diagnostic:
1. Какая именно разница (magnitude, phase, частоты, references)?
2. Какие fixtures из equivalence тестов (b139.1) ловят эту разницу?
3. Если equivalence тесты проходят, но визуал отличается — значит
   ошибка в потреблении result (адаптация `bandData()?.foo` в
   FrequencyPlot).
4. Если equivalence тесты НЕ ловят — значит test coverage
   недостаточен, добавить тест на конкретный сценарий который
   ломается.

## Учёт уроков b138-b139 каскада

1. **Audit before write.** Грэп должен быть сделан и приложен к
   коммиту (либо в коммит-сообщении, либо в отдельном комментарии).
   Без audit нет понимания scope.
2. **Diagnostic-first при провале.** SPL должен остаться идентичным.
   Если нет — diagnostic, не слепые правки.
3. **Версия в заголовке** = b139.2.
4. **Один этап = один коммит.** Не объединять с другими view
   миграциями.
5. **Старый код не удаляется.** `evaluateBand` всё ещё работает —
   используется в SUM/Export/IR до Этапов 3-4.

## Тестировать на `.dmg`

После сборки запустить
`src-tauri/target/release/bundle/dmg/PhaseForge_0.1.139-2_aarch64.dmg`,
проверить версию = b139.2, прогнать `docs/regression-checklist.md`
(все 10 пунктов).

Дополнительно — для всех 4 Gaussian × subsonic комбинаций на SPL view
визуально сравнить с b139.1 (.dmg запустить параллельно если возможно
или скриншоты). Цель: ноль visual diff.

## Правила (CLAUDE.md)

- Один коммит: `refactor: SPL view uses BandEvaluator (b139.2)`
  + Co-Authored-By.
- 7-vector review.
- Без нарратива прогресса.
- `cargo tauri build` для финальной сборки.

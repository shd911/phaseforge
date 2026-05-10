# Промт для Code: b139.4b — миграция экспорта, IR/Step, SUM на единую сущность

**Тип:** UI миграция. Bump до 0.1.0-b139.4b.

## Контекст

В b139.4a добавлен Composite phase mode в Rust + расширен
BandEvaluator (он передаёт `phase_mode: "Composite"` с
`linear_phase_main` + `subsonic_cutoff_hz`). Backend математически
правильный (4 cargo теста проходят).

Но UI вкладки **Export**, **IR/Step**, **SUM** до сих пор используют
свои inline расчёты в FrequencyPlot.tsx. Поэтому Composite режим
никак не задействован в реальном UI.

Этап 4b — переключить эти вкладки на BandEvaluator. После миграции
все четыре вкладки (SPL уже мигрирован в b139.2, остальные сейчас) =
один источник.

## Pre-flight audit

```
grep -n "evaluate_target\|compute_peq_complex\|generate_model_fir\|compute_impulse\|compute_corrected_impulse\|addGaussianMinPhase\|response.phase" src/components/FrequencyPlot.tsx
```

Классифицировать каждую найденную точку:
- **Export tab** (line ~1712 по предыдущему audit) — мигрируется.
- **SUM view** (line ~3590) — мигрируется.
- **IR/Step plot отдельной полосы** — мигрируется.
- **SPL plot** — уже на BandEvaluator (b139.2), не трогать.
- **Group delay** — если рендерится из targetPhase — после миграции
  автоматически работает.

Любая точка которая использует evaluate_target или compute_peq_complex
**в обход BandEvaluator** — кандидат на миграцию.

## Что нужно сделать

### 1. Export tab → BandEvaluator

Найти `renderExportTab` или эквивалент. Заменить inline вызов на
чтение из `createBandEvalResource(activeBand, { includeFir: () => true })`.

Доступ к данным:
- `result.fir.impulse` — для рисования импульса
- `result.combinedTargetMag` / `result.combinedTargetPhase` — для
  preview графика target curve
- Метрики (mag_err, GD ripple, causality) — если приходят в
  `result.fir`, использовать. Если нет — расширить BandEvalResult
  чтобы возвращать их.

Если `BandEvalResult.fir` сейчас не содержит метрики (только
impulse) — расширить его в `band-evaluator.ts`:

```typescript
fir?: {
  impulse: number[];
  sampleRate: number;
  causality?: number;
  magErrDb?: number;
  gdRippleMs?: number;
  preRingMs?: number;
  // и т.д. — то что отображает UI на Export tab
};
```

Пробросить из Rust ответа `generate_model_fir`.

Старый inline расчёт + `addGaussianMinPhase` патчи b138.4 в
renderExportTab — **удалить**.

### 2. IR/Step plot → BandEvaluator

Найти точку где рендерится impulse/step для отдельной полосы.
Заменить на `createBandEvalResource(activeBand, { includeIr: () => true })`.

Если в `BandEvalResult.ir` структура неполная (например только
impulse) — расширить:

```typescript
ir?: {
  impulse: number[];
  step: number[];
  time: number[];
};
```

Внутри `evaluateBandFull` если `includeIr` — вызвать
`compute_impulse` или `compute_corrected_impulse` с правильными
аргументами и собрать ir.

### 3. SUM view → BandEvaluator

SUM аггрегирует результат по нескольким полосам. Сейчас имеет свой
pipeline (line ~3590).

Подход:
- Для каждой полосы вызвать `createBandEvalResource(b, ...)` или
  `evaluateBandFull({band: b, ...})`.
- Сложить combinedTargetMag и combinedTargetPhase по правилам SUM
  (с учётом polarity, alignment_delay, и т.д.).
- Использовать compute_impulse если SUM показывает импульс.

Возможно для SUM нужна отдельная функция-агрегатор в
`band-evaluator.ts`:

```typescript
export async function evaluateSum(bands: BandState[], options: {
  freq?: number[];
  includeIr?: boolean;
}): Promise<SumEvalResult>;
```

Внутри — вызывает `evaluateBandFull` для каждой полосы и
аггрегирует.

### 4. Удаление дубликатов

После миграции трёх вкладок:
- `addGaussianMinPhase` в `band-evaluation.ts` больше не нужен (если
  все callers ушли). Удалить.
- Старая `evaluateBand` в `band-evaluation.ts` — если все callers
  ушли, удалить. Если остались (peq-optimize, auto-align) — оставить.
- `hasActiveSubsonicProtect` — оставить (используется в
  band-evaluator).
- Удалить inline phase reconstruction в FrequencyPlot.tsx (где
  явно вызывался `addGaussianMinPhase`).
- Удалить `isLin` / `phase_mode` selection logic — теперь это в
  BandEvaluator.

### 5. Тесты

**Cargo:** все existing 162 тестов должны остаться PASS (Rust не
трогаем кроме возможного расширения возвращаемых данных в
generate_model_fir для метрик).

**Vitest:** существующие 140 + новые snapshot тесты на
`evaluateSum` и расширенные `evaluateBandFull` с FIR метриками.

Snapshot тесты не должны разойтись для уже покрытых случаев.
Composite mode acceptance матрица из b139.4a — все 4 случая
проверяются через `evaluateBandFull` + сравнение с reference.

### 6. Bump

- `src-tauri/tauri.conf.json` → b139.4b.
- `src-tauri/src/lib.rs` startup-лог.
- skill `build-version`.

## Acceptance

1. На вкладке экспорта при чекбоксе «линейная фаза» включён +
   защитный фильтр включён → фаза в полосе пропускания плоская
   (≈ 0°), крутится только в инфразвуке. Заголовок Status показывает
   что-то соответствующее новому Composite режиму (или просто
   корректную информацию).
2. На вкладке экспорта при чекбоксе выключен + защитный включён →
   фаза крутится во всей нижней зоне (Gaussian min-phase + subsonic
   min-phase).
3. На вкладке IR/Step импульс отражает выбранный режим: симметричный
   pre-ring при линейной фазе, причинный (causal) при минимальной.
4. На вкладке SUM сумма всех полос корректно отражает каждую полосу
   с её выбранным режимом фазы.
5. Существующая SPL вкладка не сломалась (уже была на BandEvaluator,
   но регрессионная проверка обязательна).
6. cargo и vitest все PASS.

## Регрессионная проверка

- regression-checklist 5 UI пунктов на `.dmg b139.4b`.
- 4 комбинации (чекбокс × защитный) на каждой вкладке: Export,
  IR/Step, SUM. Все должны соответствовать acceptance матрице из
  b139.4a.
- Импорт реального замера → optimize → export → импорт обратно
  через Import → совпадает с target.

## Что НЕ делать

- Не трогать backend Composite mode (он готов в b139.4a).
- Не менять логику optimizer (peq-optimize.ts) и auto-align.
- Не делать оптимизацию производительности — это отдельная задача.
- Не пытаться прокинуть полный FirModelResult если он не используется
  в UI — только то что реально читается.

## Тестировать на `.dmg`

После сборки запустить
`PhaseForge_0.1.139-4b_aarch64.dmg`, проверить версию = b139.4b.

Главное:
- 4 комбинации (linear ON/OFF × subsonic ON/OFF) на вкладке экспорта.
- Phase в полосе пропускания должна соответствовать чекбоксу
  «линейная фаза».
- Phase в зоне subsonic должна крутиться при включённом защитном.

## Правила

- Один коммит: `refactor: migrate Export/IR/SUM tabs to BandEvaluator (b139.4b)`
  + Co-Authored-By.
- 7-vector review.
- Без нарратива.
- При провале acceptance — diagnostic, не слепой фикс.

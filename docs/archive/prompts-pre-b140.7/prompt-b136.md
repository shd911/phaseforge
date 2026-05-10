# Промт для Code: b136 — инвалидация PEQ при изменении target

ТЗ целиком: `docs/TZ-peq-stale.md`.
Текущий билд: 0.1.0-b135.3 → bump до 0.1.0-b136.

## Принципиальный выбор

Вариант **C** из обсуждения: PEQ не очищается и не переоптимизируется
автоматически. Вместо этого появляется визуальный индикатор «PEQ
устарел» с кнопками `[Переоптимизировать]` и `[Очистить]`. При FIR
экспорте при наличии stale полос — подтверждающий диалог.

Не делаем:
- авто-очистку PEQ при изменении target;
- авто-переоптимизацию;
- инвалидацию по глобальным параметрам PEQ (`tolerance`, `maxBands` и т.п.);
- инвалидацию при изменении `reference_level_db` (нормализация).

## Что нужно сделать

### 1. Структура данных

**Frontend `src/lib/types.ts`** — добавить тип `PeqOptimizedTarget`:

```typescript
export interface PeqOptimizedTarget {
  high_pass: FilterConfig | null;
  low_pass: FilterConfig | null;
  exclusion_zones: ExclusionZone[];
}
```

**Store `src/stores/bands.ts`:**
- В `BandState` добавить поле `peqOptimizedTarget: PeqOptimizedTarget | null`.
- Дефолт `null` в `createBand`.
- Сеттер `setBandPeqOptimizedTarget(bandId, snapshot)`.

**Rust `src-tauri/src/project.rs`:**
- В `BandData` добавить `#[serde(default)] peq_optimized_target: Option<PeqOptimizedTargetData>`.
- Структура `PeqOptimizedTargetData { high_pass, low_pass, exclusion_zones }`,
  переиспользует существующий `FilterConfig` (тот же что в TargetCurve).

**Round-trip `project-io.ts`:**
- `mapBandToProject`: `peq_optimized_target: b.peqOptimizedTarget ?? null`.
- `mapBandFromProject`: `peqOptimizedTarget: b.peq_optimized_target ?? null`.

**Light snapshot для b132 history (`bands.ts`):**
- В `_captureBandsLight` и `_applyBandsLight` добавить `peqOptimizedTarget`.
- LightBand должен включать это поле, чтобы Cmd+Z после Optimize
  корректно откатывал и snapshot.

### 2. Логика инвалидации

**`src/stores/peq-optimize.ts`:**

Функция-хелпер:

```typescript
function captureOptimizedTarget(b: BandState): PeqOptimizedTarget {
  return {
    high_pass: b.target.high_pass ? { ...b.target.high_pass } : null,
    low_pass: b.target.low_pass ? { ...b.target.low_pass } : null,
    exclusion_zones: JSON.parse(JSON.stringify(b.exclusionZones)),
  };
}
```

В `handleOptimizePeq` после успешной оптимизации:
```typescript
setBandPeqBands(b.id, mergeBands(frozenBands, result.bands));
setBandPeqOptimizedTarget(b.id, captureOptimizedTarget(b));
```

В `handleOptimizeAll` — то же внутри финального `batch()`.

В `handleClearPeq` — добавить `setBandPeqOptimizedTarget(b.id, null)`.

Эспортируемая функция:

```typescript
export function peqStale(b: BandState): boolean {
  if (!b.peqBands || b.peqBands.length === 0) return false;
  if (!b.peqOptimizedTarget) return false;
  return !targetEquals(currentTargetSnapshot(b), b.peqOptimizedTarget);
}
```

Реализация `targetEquals`, `filterEquals`, `exclusionZonesEquals` — см.
ТЗ, секция «Computed signal `peqStale(band)`». Сравнение полей
`filter_type`, `order`, `freq_hz`, `shape`, `q`. **Не** сравнивать
`linear_phase` и `reference_level_db`.

### 3. UI индикаторы

**3.1. Шапка над PEQ блоком в `src/components/ControlPanel.tsx`:**

Когда `peqStale(activeBand())` true — рендерить блок над PEQ-кнопками:

```jsx
<Show when={activeBand() && peqStale(activeBand()!)}>
  <div class="peq-stale-banner">
    <span>⚠ PEQ устарел: target изменён после последней оптимизации</span>
    <div class="peq-stale-actions">
      <button class="dlg-btn dlg-btn-primary" onClick={handleOptimizePeq}>
        Переоптимизировать
      </button>
      <button class="dlg-btn" onClick={handleClearPeq}>
        Очистить
      </button>
    </div>
  </div>
</Show>
```

Стили в `src/App.css`:
```css
.peq-stale-banner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-md);
  padding: var(--space-sm) var(--space-md);
  background: rgba(217, 119, 6, 0.12);
  border: 1px solid #d97706;
  border-radius: 4px;
  color: #d97706;
  font-size: 13px;
  margin-bottom: var(--space-md);
}
.peq-stale-actions { display: flex; gap: var(--space-sm); }
```

**3.2. Индикатор в BandTabs (`src/components/BandTabs.tsx`):**

Рядом с именем полосы — маленькая оранжевая точка когда `peqStale(b)` true.
Тултип через `title`: «PEQ устарел».

```jsx
<Show when={peqStale(b)}>
  <span class="band-stale-dot" title="PEQ устарел" />
</Show>
```

CSS:
```css
.band-stale-dot {
  display: inline-block;
  width: 6px; height: 6px;
  border-radius: 50%;
  background: #d97706;
  margin-left: 6px;
  vertical-align: middle;
}
```

### 4. Проверка при FIR-экспорте

**Новый компонент `src/components/StalePeqExportDialog.tsx`:**

Promise-based API как у `UnsavedChangesDialog`:

```typescript
export function showStaleConfirmDialog(
  bandNames: string[],
): Promise<boolean>;  // true = продолжить, false = отмена
```

Layout — см. ТЗ (заголовок «Экспорт устаревшего PEQ», список полос,
текст, две кнопки).

Смонтировать в `App.tsx`.

**Интеграция в FIR Export:**

Найти где сейчас вызывается `invoke("export_fir_audio", ...)`. Скорее
всего в `src/lib/fir-export.ts` или в обработчике кнопки в
`ControlPanel.tsx`. Перед invoke добавить:

```typescript
const staleBands = appState.bands.filter(b => b.measurement && peqStale(b));
if (staleBands.length > 0) {
  const proceed = await showStaleConfirmDialog(staleBands.map(b => b.name));
  if (!proceed) return;
}
```

Если экспорт умеет работать с одной активной полосой — фильтровать
список полос соответственно (только те что попадают в экспорт).

### 5. Bump версии

- `src-tauri/tauri.conf.json` — version + productName/title до b136.
- `src-tauri/src/lib.rs` — startup-лог.
- После билда — skill `build-version`.

## Acceptance

1. Оптимизация → snapshot записан, виден в `.pfproj` после Save как
   `peq_optimized_target`.
2. Изменение HP `freq_hz` после оптимизации → шапка-баннер появляется
   в PEQ блоке, оранжевая точка появляется во вкладке полосы.
3. Возврат HP к исходному значению → баннер и точка исчезают.
4. Изменение `reference_level_db` → баннер не появляется.
5. Кнопка «Переоптимизировать» в баннере → `handleOptimizePeq`,
   баннер гаснет после успеха.
6. Кнопка «Очистить» → `handleClearPeq`, peqBands и snapshot null,
   баннер гаснет.
7. FIR Export при stale полосах → модалка с подтверждением. «Отмена»
   останавливает экспорт, «Экспортировать всё равно» продолжает.
8. Cmd+Z после Optimize → возврат к state без peqBands и без snapshot.
9. Загрузка старого `.pfproj` (без `peq_optimized_target`) → баннер не
   появляется, экспорт не блокируется.
10. b133 Versions: создание версии после Optimize, потом изменение
    target, потом Restore → восстановленное состояние не stale (snapshot
    из версии совпадает с restored target).

## Регрессионная проверка

- b131-b135.3 целы.
- Optimize / Optimize All работают как раньше.
- Cmd+Z / Cmd+Shift+Z (b132) восстанавливают peqOptimizedTarget вместе
  с peqBands.
- Save / Save As / Open / Recent.
- Versions (b133) сохраняют и восстанавливают snapshot.
- FIR Export без stale полос — без диалога, как раньше.
- Frozen bands: при оптимизации snapshot фиксирует текущее состояние
  exclusion_zones и target. При смене любого frozen band's enabled
  → НЕ инвалидирует (это вне target snapshot).

## Что НЕ трогать

- Логика самой оптимизации (`optimizeBand`, `auto_peq_lma`).
- Frozen bands механизм.
- Глобальные параметры PEQ (`tolerance`, `maxBands` и т.д.) — они вне
  snapshot.
- `setBandPeqBands` сигнатура и существующие callers.

## Тестировать на `.dmg`

После сборки запустить
`src-tauri/target/release/bundle/dmg/PhaseForge_0.1.136_aarch64.dmg`
и пройти acceptance pp. 1-7. Не полагаться только на `cargo tauri dev`.

## Правила (CLAUDE.md)

- Один коммит: `feat: peq stale flag with target invalidation (b136)`
  + Co-Authored-By.
- 7-vector review.
- Без нарратива прогресса.
- `cargo tauri build` для финальной сборки.

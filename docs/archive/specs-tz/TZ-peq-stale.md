# ТЗ: Инвалидация PEQ при изменении target

**Цель:** при изменении значимых параметров target (HP/LP кроссовер, тип
фильтра, slope, shape) после выполненной оптимизации PEQ — пометить PEQ
как «устаревший». Визуально выделить это в UI, дать кнопки
переоптимизации и очистки. При экспорте FIR проверить и предупредить
если экспортируется устаревший PEQ.

**Out of scope:**
- Авто-переоптимизация и авто-очистка (явно отвергнуто как варианты A/B).
- Учёт изменений глобальных параметров PEQ (`tolerance`, `maxBands`,
  `gainRegularization`, `peqFloor`, `peqRangeMode`) — в первой итерации
  не считаем их инвалидирующими.
- Учёт изменений `target_enabled`, `inverted`, `linkedToNext`,
  `alignment_delay` — на оптимизацию не влияют или влияют косвенно.

---

## Что считается значимым изменением

Сравниваем сохранённый `peq_optimized_target` snapshot с текущим
состоянием полосы. Различие в любом из перечисленных полей →
`peqStale = true`.

**Поля target:**
- `high_pass.filter_type`
- `high_pass.order` (slope)
- `high_pass.freq_hz`
- `high_pass.shape`
- `high_pass.q`
- `high_pass` появился или исчез (null ↔ объект)
- те же поля для `low_pass`

**Дополнительно:** `exclusion_zones` — изменение состава или границ.

**Не сравниваем:**
- `target.reference_level_db` (используется как нормализация, оптимизатор
  пересчитывает его сам через refOffset).
- `high_pass.linear_phase` / `low_pass.linear_phase` — это для FIR
  экспорта, не влияет на target curve в смысле магнитуды.

---

## Структура данных

### Frontend `src/lib/types.ts`

```typescript
export interface PeqOptimizedTarget {
  high_pass: FilterConfig | null;
  low_pass: FilterConfig | null;
  exclusion_zones: ExclusionZone[];
}
```

### Store `src/stores/bands.ts`

В `BandState` добавить:

```typescript
peqOptimizedTarget: PeqOptimizedTarget | null;
```

Дефолт — `null` для свежесозданных полос. При `createBand` —
инициализируется null.

Сеттеры:
```typescript
export function setBandPeqOptimizedTarget(
  bandId: string,
  snapshot: PeqOptimizedTarget | null,
): void;
```

### Rust `src-tauri/src/project.rs`

В `BandData` добавить:

```rust
#[serde(default)]
pub peq_optimized_target: Option<PeqOptimizedTargetData>,

// ...

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeqOptimizedTargetData {
    pub high_pass: Option<FilterConfig>,
    pub low_pass: Option<FilterConfig>,
    pub exclusion_zones: Vec<serde_json::Value>,
}
```

`FilterConfig` уже сериализуется в проекте, использовать как есть.

### Round-trip

В `project-io.ts`:
- `mapBandToProject`: `peq_optimized_target: b.peqOptimizedTarget`.
- `mapBandFromProject`: `peqOptimizedTarget: b.peq_optimized_target ?? null`.

### Light snapshot (b132 history)

В `_captureBandsLight` и `_applyBandsLight` (в `bands.ts`) — добавить
`peqOptimizedTarget` в LightBand структуру. Это нужно чтобы Cmd+Z после
оптимизации корректно восстанавливал и состояние «stale-or-not».

---

## Логика инвалидации

### Где сетится `peqOptimizedTarget`

В `peq-optimize.ts`, после успешной оптимизации в `handleOptimizePeq` и
`handleOptimizeAll` — сохранить snapshot:

```typescript
function captureOptimizedTarget(b: BandState): PeqOptimizedTarget {
  return {
    high_pass: b.target.high_pass ? { ...b.target.high_pass } : null,
    low_pass: b.target.low_pass ? { ...b.target.low_pass } : null,
    exclusion_zones: JSON.parse(JSON.stringify(b.exclusionZones)),
  };
}

// В handleOptimizePeq:
setBandPeqBands(b.id, mergedBands);
setBandPeqOptimizedTarget(b.id, captureOptimizedTarget(b));
```

### Где сбрасывается

- `handleClearPeq` / `clearBandPeqBands` → также `setBandPeqOptimizedTarget(b.id, null)`.
- При replace measurement → пока не трогаем (отдельный вопрос).

### Computed signal `peqStale(band)`

Чистая функция в `peq-optimize.ts`:

```typescript
export function peqStale(b: BandState): boolean {
  if (!b.peqBands || b.peqBands.length === 0) return false;
  if (!b.peqOptimizedTarget) return false; // оптимизация не выполнялась
  return !targetEquals(currentTargetSnapshot(b), b.peqOptimizedTarget);
}

function currentTargetSnapshot(b: BandState): PeqOptimizedTarget {
  return {
    high_pass: b.target.high_pass ? { ...b.target.high_pass } : null,
    low_pass: b.target.low_pass ? { ...b.target.low_pass } : null,
    exclusion_zones: b.exclusionZones,
  };
}

function targetEquals(a: PeqOptimizedTarget, b: PeqOptimizedTarget): boolean {
  if (!filterEquals(a.high_pass, b.high_pass)) return false;
  if (!filterEquals(a.low_pass, b.low_pass)) return false;
  if (!exclusionZonesEquals(a.exclusion_zones, b.exclusion_zones)) return false;
  return true;
}

function filterEquals(a: FilterConfig | null, b: FilterConfig | null): boolean {
  if (a === null && b === null) return true;
  if (a === null || b === null) return false;
  return a.filter_type === b.filter_type
      && a.order === b.order
      && a.freq_hz === b.freq_hz
      && a.shape === b.shape
      && a.q === b.q;
}

function exclusionZonesEquals(a: ExclusionZone[], b: ExclusionZone[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i].startHz !== b[i].startHz) return false;
    if (a[i].endHz !== b[i].endHz) return false;
  }
  return true;
}
```

---

## UI

### 1. Индикатор в ControlPanel (PEQ блок)

В `src/components/ControlPanel.tsx`, секция PEQ — когда `peqStale(band)`
true:

- Оранжевая рамка вокруг блока PEQ (border 2px solid #d97706 или
  CSS-переменная для warning).
- Поверх блока — небольшая шапка-bar:

```
⚠ PEQ устарел: target изменён после последней оптимизации
                          [ Переоптимизировать ]  [ Очистить ]
```

Кнопки используют те же `handleOptimizePeq` и `handleClearPeq`.

### 2. Индикатор в BandTabs

В строке вкладки полосы — маленький оранжевый индикатор-точка рядом с
именем полосы если `peqStale(band)` true. Тултип: «PEQ устарел». Это
полезно когда работаешь на одной вкладке и не видишь PEQ-блок.

### 3. Проверка при FIR-экспорте

В обработчике Export FIR (найти в `fir-export.ts` или ControlPanel) —
перед запуском проверить:

```typescript
const staleBands = appState.bands.filter(b => b.measurement && peqStale(b));
if (staleBands.length > 0) {
  const proceed = await showStaleConfirmDialog(staleBands.map(b => b.name));
  if (!proceed) return;
}
```

Диалог `StalePeqExportDialog`:

```
┌─ Экспорт устаревшего PEQ ───────────────────────────┐
│                                                     │
│  На следующих полосах PEQ устарел:                  │
│  • LF                                               │
│  • MID                                              │
│                                                     │
│  Target изменён после последней оптимизации.        │
│  Экспорт даст коррекцию, не соответствующую         │
│  текущему target.                                   │
│                                                     │
│  [ Отмена ]              [ Экспортировать всё равно ]│
└─────────────────────────────────────────────────────┘
```

«Отмена» — отменить экспорт. «Экспортировать всё равно» — продолжить.
Никаких сторонних действий (не очищает, не переоптимизирует
автоматически).

Promise-API как у `UnsavedChangesDialog`:

```typescript
export function showStaleConfirmDialog(
  bandNames: string[],
): Promise<boolean>;
```

---

## Edge cases

| Сценарий | Поведение |
|---|---|
| Оптимизация выполнена, target не менялся | `peqStale=false`, индикатора нет |
| Оптимизация → HP 300→305 | `peqStale=true`, индикатор появляется |
| Оптимизация → HP 300→305 → HP 305→300 | `peqStale=false`, флаг снимается (snapshot equals current) |
| Оптимизация → HP добавлен с null | `peqStale=true` |
| Оптимизация → reference_level_db поменялся | `peqStale=false` (нормализация) |
| Загрузка старого проекта без `peq_optimized_target` | `peqOptimizedTarget=null` → `peqStale=false`. Не вводим в заблуждение пользователя на legacy-проектах |
| `clearBandPeqBands` | `peqStale=false`, snapshot тоже сбрасывается |
| Восстановление снимка через b133 Versions | `peqOptimizedTarget` восстанавливается из снимка целиком |
| Cmd+Z после Optimize | Возврат к peqBands=[] и peqOptimizedTarget=null (history восстанавливает обоих) |
| FIR-экспорт при `peqStale=false` всех полос | Диалог не появляется |

---

## Acceptance

1. После оптимизации полосы — `peqOptimizedTarget` записан, в `.pfproj`
   виден после Save.
2. Изменение HP `freq_hz` после оптимизации → оранжевая рамка вокруг PEQ
   блока + индикатор-точка во вкладке полосы.
3. Возврат HP к исходному значению → флаг снимается.
4. Изменение `reference_level_db` → флаг НЕ появляется.
5. Кнопка «Переоптимизировать» в шапке-баре → запускает обычный
   `handleOptimizePeq`, флаг гасится после успеха.
6. Кнопка «Очистить» → `handleClearPeq`, peqBands и snapshot обнуляются.
7. FIR Export при stale полосах → модалка с подтверждением. Отмена
   останавливает экспорт. «Экспортировать всё равно» продолжает.
8. Cmd+Z после Optimize → возврат к state без peqBands и без snapshot.
9. Загрузка старого `.pfproj` (без `peq_optimized_target`) →
   индикатора нет, экспорт не блокируется.

---

## Этапы

Один билд b136 — задача компактная и связная.

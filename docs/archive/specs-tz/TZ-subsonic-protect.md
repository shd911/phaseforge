# ТЗ: Защитный subsonic фильтр для Gaussian HP

**Цель:** при использовании Gaussian HP автоматически добавлять
минимально-фазовый Butterworth 48 дБ/окт на 3 октавы ниже основного
cutoff — для защиты driver от излишнего excursion в инфразвуке.
Параметры (slope, distance) **жёстко зашиты**, пользователь имеет
только чекбокс отключения.

**Out of scope:**
- Применение защиты для других типов HP (LR, Butterworth) — у них
  собственный slope уже достаточно крутой.
- Защита для LP (HF cutoff) — не относится к excursion.
- UI выбора параметров slope/distance — фиксированы 48 дБ/окт и 3 окт.

---

## Принципиальный выбор (зафиксировано)

1. Защита **только** для Gaussian HP.
2. Параметры **захардкожены**: 48 дБ/окт (Butterworth 8th order),
   3 октавы ниже основного freq_hz. Минимально-фазовый.
3. **Авто-вкл** при выборе Gaussian HP. Можно снять чекбоксом, но в
   таком состоянии под чекбоксом — предупреждение «Защита отключена,
   риск excursion на инфразвуке».
4. На графике target — **сливается** с основной кривой, отдельно не
   рисуется.
5. Edge case: HP ≤ 40 Гц → subsonic уйдёт ниже 5 Гц, бессмысленно.
   Чекбокс disabled с тултипом «HP слишком низкий, защита не требуется».

---

## DSP-обоснование выбранных параметров

| Slope | Distance | f_subsonic при HP=100 Гц | Atten в passband (200 Гц) | Suppression на 12.5 Гц | Verdict |
|---|---|---|---|---|---|
| 24 дБ/окт | 2.5 окт | 17.7 Гц | ~0.05 дБ | ~24 дБ | защита слабая |
| 24 дБ/окт | 3.0 окт | 12.5 Гц | ~0.02 дБ | ~24 дБ на 6.25 Гц | защита очень слабая |
| 48 дБ/окт | 2.5 окт | 17.7 Гц | ~0.10 дБ | ~48 дБ | компромисс |
| **48 дБ/окт** | **3.0 окт** | **12.5 Гц** | **~0.04 дБ** | **~48 дБ на 6.25 Гц** | **выбрано** |

48@3окт даёт максимум защиты при минимальном влиянии на passband (<0.05 дБ
на октаве выше HP).

---

## Структура данных

### Rust `src-tauri/src/target/mod.rs`

В `FilterConfig` добавить:

```rust
#[serde(default)]
pub subsonic_protect: Option<bool>,
```

`None` — для не-Gaussian типов и старых проектов (поведение как
раньше). `Some(true)` — защита включена. `Some(false)` — пользователь
явно отключил.

### Frontend `src/lib/types.ts`

В `FilterConfig` interface добавить:

```typescript
subsonic_protect?: boolean | null;
```

### Round-trip `project-io.ts`

Через `cloneFilter` уже копируется через spread — нужно явно прокинуть
поле `subsonic_protect`. Аналогично в обратной трансформации.

---

## Применение subsonic в Rust

### Где модифицировать

В `apply_filter` (или эквивалентной функции, обрабатывающей HP). Для
HP типа Gaussian с `subsonic_protect == Some(true)` и `freq_hz > 40`:

1. Применить Gaussian HP magnitude как обычно.
2. После этого умножить на Butterworth HP 8th order:
   - cutoff = `freq_hz / 8.0` (3 октавы ниже)
   - magnitude формула:
     ```
     subsonic_mag(f) = sqrt(1 / (1 + (f_subsonic / f)^16))
     ```
     где `16 = 2 × order` (8th order Butterworth slope).
3. **Phase не считаем** в `evaluate_target` — Gaussian HP в
   PhaseForge возвращает phase=0, frontend восстанавливает min-phase
   через Hilbert на изменённой magnitude. Это значит min-phase
   автоматически учтёт subsonic — отдельная phase logic не нужна.

Псевдокод:

```rust
// In apply_filter, when handling HP Gaussian:
let mut hp_mag = gaussian_hp_magnitude(...);  // existing
if cfg.subsonic_protect == Some(true) && cfg.freq_hz > 40.0 {
    let f_subsonic = cfg.freq_hz / 8.0;
    for i in 0..freq.len() {
        let f = freq[i];
        let ratio = (f_subsonic / f).powi(16);  // (f_sub/f)^(2*order)
        let subsonic_mag = (1.0 / (1.0 + ratio)).sqrt();
        hp_mag[i] *= subsonic_mag;
    }
}
```

Финальная magnitude target curve уже включает subsonic. Hilbert на
frontend (`compute_minimum_phase`) даст правильную phase.

### FIR export

FIR использует `evaluate_target` для построения target → subsonic
автоматически входит в FIR коэффициенты. Никаких отдельных правок в
FIR pipeline не нужно.

---

## UI (frontend)

### Чекбокс «Защитный subsonic»

Найти место где пользователь выбирает HP `filter_type` (вероятно в
`ControlPanel.tsx` или `CrossoverDialog.tsx`). Добавить чекбокс
который:

- Виден **только когда** `filter_type === "Gaussian"` для HP.
- По умолчанию ВКЛ при первом выборе Gaussian HP (см. логика дефолта
  ниже).
- Disabled когда `freq_hz <= 40`. Тултип: «HP слишком низкий
  ({freq_hz} Гц), защита не требуется».
- Когда выключен пользователем — под чекбоксом мелкий warning:
  «⚠ Защита отключена, риск excursion на инфразвуке».

### Логика дефолта

При **смене filter_type на Gaussian** (с любого другого):

```typescript
function setHpFilterType(newType: string) {
  const hp = band.target.high_pass;
  if (newType === "Gaussian") {
    hp.filter_type = "Gaussian";
    hp.subsonic_protect = true;  // авто-ВКЛ
  } else {
    hp.filter_type = newType;
    hp.subsonic_protect = null;  // не относится к не-Gaussian
  }
}
```

При смене с Gaussian на не-Gaussian: `subsonic_protect = null`.
При смене обратно на Gaussian: `subsonic_protect = true` (заново
включается). Это требование Кирилла «авто-ВКЛ» — мы не помним
предыдущее состояние пользователя.

При создании нового HP Gaussian с нуля (например через UI выбор):
тоже `subsonic_protect = true`.

### Tooltip на чекбоксе

«Минимально-фазовый Butterworth 48 дБ/окт на 3 октавы ниже HP.
Защищает driver от излишнего excursion в инфразвуке.»

---

## Связь с b136 (peqStale)

`subsonic_protect` — НЕ должен инвалидировать PEQ:

- Subsonic влияет на target ниже HP (зона ниже passband).
- Оптимизатор не работает в этой зоне (peqFloor обрезает).
- Изменение subsonic_protect не должно ставить peqStale=true.

В `peq-optimize.ts:filterEquals` (b136) — игнорировать поле
`subsonic_protect` при сравнении:

```typescript
function filterEquals(a, b) {
  if (a === null && b === null) return true;
  if (a === null || b === null) return false;
  return a.filter_type === b.filter_type
      && a.order === b.order
      && a.freq_hz === b.freq_hz
      && a.shape === b.shape
      && a.q === b.q;
  // НЕ сравниваем subsonic_protect и linear_phase
}
```

Уже сейчас не сравниваем `linear_phase` — добавляем то же исключение
для `subsonic_protect`.

---

## Edge cases

| Сценарий | Поведение |
|---|---|
| Выбран Gaussian HP, freq_hz=100 Гц | Чекбокс ВКЛ автоматом. Subsonic на 12.5 Гц применяется |
| HP freq_hz=30 Гц, Gaussian | Чекбокс disabled. Subsonic не применяется. Tooltip объясняет |
| Сменили HP с Gaussian на LR | subsonic_protect → null. Чекбокс пропадает |
| Сменили обратно с LR на Gaussian | subsonic_protect → true (авто-ВКЛ заново) |
| Загрузка старого `.pfproj` без поля | subsonic_protect = None / null. Защита НЕ применяется автоматически (legacy compat). Можно вручную включить через чекбокс |
| Изменение subsonic_protect | НЕ ставит peqStale=true (b136) |
| Подключённое FIR-export | Subsonic embedded в коэффициенты автоматически |

---

## Acceptance

1. Создание Gaussian HP с freq_hz=100 → чекбокс «Защитный subsonic» виден,
   ВКЛ по умолчанию. На графике target ниже 50 Гц виден дополнительный
   крутой спад (~48 дБ/окт ниже 12.5 Гц).
2. Снятие чекбокса → subsonic исчезает, target ниже 50 Гц возвращается
   к чистому Gaussian спаду. Под чекбоксом появляется warning.
3. Смена HP на LR → чекбокс пропадает, поведение target — стандартное LR.
4. Создание Gaussian HP с freq_hz=30 → чекбокс disabled с тултипом.
   Subsonic не применяется.
5. FIR export после применения subsonic → коэффициенты содержат
   объединённый Gaussian × Butterworth subsonic спад.
6. Изменение subsonic_protect через чекбокс → peqStale=false на полосе
   (b136 не реагирует на это поле).
7. Загрузка старого `.pfproj` без поля subsonic_protect → защита
   неактивна, существующий target не меняется.
8. Round-trip: Save → Open → состояние чекбокса и target визуально
   совпадают с моментом сохранения.

## Регрессионная проверка

- b131-b137 целы.
- Optimize / FIR export работают для всех типов HP.
- Существующие Gaussian HP проекты загружаются без изменений в UI
  (subsonic_protect = null означает поведение как до b138).
- Hilbert min-phase reconstruction (когда `linear_phase=false`) учитывает
  изменённую magnitude и даёт правильную phase для finalного FIR.

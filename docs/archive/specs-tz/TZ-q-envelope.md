# ТЗ: Частотно-зависимое ограничение и маркировка добротности PEQ

**Цель:** ограничить добротность PEQ-полос плавно изменяющимся
потолком в зависимости от частоты, и помечать в UI полосы, у которых
добротность вышла за порог «warning». Это убирает физически
неоправданные узкие полосы на средних и высоких частотах, при этом
сохраняя возможность корректировать комнатные моды на басу.

**Out of scope:**
- Запрет редактирования high-Q полос пользователем вручную (он сам
  знает что делает).
- Изменение `Q_MAX_ABOVE_LP = 2.5` для зон вне passband (отдельная
  логика, остаётся как есть).
- Авто-исправление существующих PEQ полос с Q выше нового cap.

---

## Принципиальный выбор (зафиксировано)

1. Потолок Q (cap) и порог маркировки (warn) — **плавно зависят**
   от частоты, интерполяция по `log2(f)`.
2. Опорные точки **захардкожены**, не выводятся в UI.
3. Маркировка показывается **в списке PEQ-полос И на графике**.
4. По клику на иконку `⚠` — всплывающее окно с предупреждением.
5. Существующие PEQ-полосы с Q выше нового cap при первой загрузке
   проекта в b137 — **не трогаем**, только маркируем.

---

## Формулы

```
Опорные точки:
  log2(200)  ≈ 7.64
  log2(2000) ≈ 10.97
  Δlog       = log2(2000) − log2(200) ≈ 3.32  (log2 от 10)

q_cap(f):
  если f ≤ 200:   возвращаем 12.0 (плато)
  если f ≥ 2000:  возвращаем 4.0  (плато)
  иначе:          t = (log2(f) − log2(200)) / Δlog
                  возвращаем 12.0 − 8.0 × t

q_warn(f):
  если f ≤ 200:   возвращаем 8.0
  если f ≥ 2000:  возвращаем 3.0
  иначе:          t = (log2(f) − log2(200)) / Δlog
                  возвращаем 8.0 − 5.0 × t
```

Значения для проверки:
- f=100 Гц: cap=12, warn=8
- f=200 Гц: cap=12, warn=8
- f=632 Гц: cap=8, warn=5.5 (середина по log)
- f=1000 Гц: cap≈6.07, warn≈4.06
- f=2000 Гц: cap=4, warn=3
- f=10000 Гц: cap=4, warn=3

При пересечении с `Q_MAX_ABOVE_LP = 2.5` (когда `freq > lp_freq`) —
берётся минимум: `effective_cap = q_cap(f).min(Q_MAX_ABOVE_LP)`.

---

## Rust часть

### Новый файл `src-tauri/src/peq/q_envelope.rs`

```rust
/// Frequency-dependent Q ceiling for PEQ optimization.
/// Reflects psychoacoustic reality: high Q on bass is acceptable
/// (room modes), high Q on treble causes ringing.
pub fn q_cap_at(freq_hz: f64) -> f64 {
    const F_LO: f64 = 200.0;
    const F_HI: f64 = 2000.0;
    const Q_LO: f64 = 12.0;
    const Q_HI: f64 = 4.0;
    if freq_hz <= F_LO { return Q_LO; }
    if freq_hz >= F_HI { return Q_HI; }
    let t = (freq_hz.log2() - F_LO.log2()) / (F_HI.log2() - F_LO.log2());
    Q_LO - (Q_LO - Q_HI) * t
}

pub fn q_warn_at(freq_hz: f64) -> f64 {
    const F_LO: f64 = 200.0;
    const F_HI: f64 = 2000.0;
    const Q_LO: f64 = 8.0;
    const Q_HI: f64 = 3.0;
    if freq_hz <= F_LO { return Q_LO; }
    if freq_hz >= F_HI { return Q_HI; }
    let t = (freq_hz.log2() - F_LO.log2()) / (F_HI.log2() - F_LO.log2());
    Q_LO - (Q_LO - Q_HI) * t
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cap_plateau_low() { assert_eq!(q_cap_at(50.0), 12.0); assert_eq!(q_cap_at(200.0), 12.0); }
    #[test]
    fn cap_plateau_high() { assert_eq!(q_cap_at(2000.0), 4.0); assert_eq!(q_cap_at(20000.0), 4.0); }
    #[test]
    fn cap_midpoint() {
        let q = q_cap_at(632.456);  // sqrt(200*2000)
        assert!((q - 8.0).abs() < 0.01);
    }
    #[test]
    fn warn_lower_than_cap() {
        for f in [100.0, 500.0, 1000.0, 5000.0] {
            assert!(q_warn_at(f) < q_cap_at(f));
        }
    }
}
```

Зарегистрировать в `src-tauri/src/peq/mod.rs`:
```rust
pub mod q_envelope;
pub use q_envelope::{q_cap_at, q_warn_at};
```

### Замена clamp в LMA и оптимизаторе

**`src-tauri/src/peq/lma.rs:238-243`** — текущий код:
```rust
let q_max = if params[i * 3] > self.lp_freq {
    Q_MAX_ABOVE_LP
} else {
    Q_MAX
};
params[i * 3 + 2] = params[i * 3 + 2].clamp(Q_MIN, q_max);
```

Заменить на:
```rust
let freq = params[i * 3];
let q_max = if freq > self.lp_freq {
    Q_MAX_ABOVE_LP
} else {
    crate::peq::q_cap_at(freq)
};
params[i * 3 + 2] = params[i * 3 + 2].clamp(Q_MIN, q_max);
```

**`src-tauri/src/peq/mod.rs:287-289`** — post-processing clamp:
```rust
if b.freq_hz > lp_freq {
    b.q = b.q.clamp(Q_MIN, Q_MAX_ABOVE_LP);
} else {
    b.q = b.q.clamp(Q_MIN, Q_MAX);
}
```

Заменить:
```rust
if b.freq_hz > lp_freq {
    b.q = b.q.clamp(Q_MIN, Q_MAX_ABOVE_LP);
} else {
    b.q = b.q.clamp(Q_MIN, q_envelope::q_cap_at(b.freq_hz));
}
```

**`src-tauri/src/peq/mod.rs:252`** — в seed-функции при добавлении новой
полосы. Сейчас `min(Q_MAX)`. Заменить на `min(q_envelope::q_cap_at(...))`
с учётом частоты worst_idx.

**`src-tauri/src/peq/mod.rs:1067-1069`** — это assertion в тесте, проверка
`b.q <= Q_MAX_ABOVE_LP + 0.01` для зоны выше LP. Не трогать.

**`src-tauri/src/peq/lma.rs:285`** — комментарий «penalize Q > Q_MAX_ABOVE_LP»,
если есть penalty term — оставить как есть, это другая зона. Если в
penalty используется Q_MAX для зоны внутри passband — заменить на
q_cap_at(freq).

**`src-tauri/src/peq/greedy.rs:127`** — `q.clamp(Q_MIN, Q_MAX)`. Заменить
на `q.clamp(Q_MIN, q_envelope::q_cap_at(peak_freq))`.

**`src-tauri/src/peq/greedy.rs:235`** — `group_q_min.clamp(Q_MIN, Q_MAX)`.
Это сложнее — нужно знать частоту. Если в этой точке есть `freq_hz`
доступная — использовать. Иначе оставить `Q_MAX` как абсолютный fallback,
LMA доработает дальше.

**Константа `Q_MAX = 10.0` в `peq/types.rs`** — оставить как есть (как
absolute fallback для случаев где частота недоступна).

### Q_MIN тоже учесть

Найти определение `Q_MIN` (видимо `0.3` или `0.5`). Не трогать —
нижняя граница Q не меняется.

---

## Frontend часть

### Новый файл `src/lib/peq-quality.ts`

```typescript
/**
 * Frequency-dependent warning threshold for PEQ Q. Mirrors Rust q_warn_at.
 * Returns Q above which UI shows a warning marker.
 */
export function qWarnAt(freqHz: number): number {
  const F_LO = 200;
  const F_HI = 2000;
  const Q_LO = 8;
  const Q_HI = 3;
  if (freqHz <= F_LO) return Q_LO;
  if (freqHz >= F_HI) return Q_HI;
  const t = (Math.log2(freqHz) - Math.log2(F_LO))
          / (Math.log2(F_HI) - Math.log2(F_LO));
  return Q_LO - (Q_LO - Q_HI) * t;
}

import type { PeqBand } from "./types";

/** Indices of bands whose Q exceeds the warn threshold. */
export function highQIndices(bands: PeqBand[]): Set<number> {
  const s = new Set<number>();
  for (let i = 0; i < bands.length; i++) {
    const b = bands[i];
    if (!b.enabled) continue;  // disabled bands not flagged
    if (b.q > qWarnAt(b.freq_hz)) s.add(i);
  }
  return s;
}
```

### Маркировка в списке PEQ-полос

Найти компонент, рендерящий список PEQ-полос (вероятно
`src/components/PeqList.tsx` или внутри `ControlPanel.tsx`).

Для каждой записи: если `i` в `highQIndices(band.peqBands)` —
рендерить кнопку-иконку:

```jsx
<button
  class="peq-warn-icon"
  title=""
  aria-label="Высокая добротность"
  onClick={() => openHighQPopup(band.peqBands[i], i)}
>⚠</button>
```

CSS:
```css
.peq-warn-icon {
  background: transparent;
  border: none;
  color: #d97706;
  font-size: 14px;
  cursor: pointer;
  padding: 0 4px;
  margin-left: 4px;
}
.peq-warn-icon:hover { background: rgba(217,119,6,0.18); border-radius: 3px; }
```

Иконка не имеет hover-тултипа (по требованию — попап только по клику).
Атрибут `title=""` пустой, чтобы system tooltip не вмешивался.

### Маркировка на графике

Найти где рендерятся маркеры PEQ полос в `FrequencyPlot.tsx`. Для
каждого маркера, если индекс в `highQIndices` — обводка/окантовка
жёлтым (`#d97706`) вместо текущего цвета. Внутреннюю заливку не менять.

Свойство SVG: `stroke="#d97706"` `stroke-width="2"` для маркера-кружка
(или эквивалент для его geometry).

### Компонент `src/components/HighQWarningPopup.tsx`

Promise-based API:

```typescript
interface HighQContext {
  band: PeqBand;
  index: number;
}

const [_state, _setState] = createSignal<HighQContext | null>(null);

export function openHighQPopup(band: PeqBand, index: number): void {
  _setState({ band, index });
}

export const isHighQPopupOpen = () => _state() !== null;
```

Layout:

```
┌─ Высокая добротность ────────────────────────────┐
│                                                  │
│  Полоса {index+1}: {freq_hz} Гц                  │
│  Q = {q.toFixed(2)}                              │
│  Порог предупреждения: Q ≤ {qWarnAt(freq):.1f}   │
│                                                  │
│  Узкая полоса с высокой добротностью даёт        │
│  заметный звон на импульсе и часто корректирует  │
│  артефакт замера, а не реальную особенность      │
│  системы. Рекомендуется уменьшить Q вручную или  │
│  применить notch с более широкой полосой.        │
│                                                  │
│                                       [ Закрыть ]│
└──────────────────────────────────────────────────┘
```

Закрытие: Escape / клик вне / кнопка «Закрыть». Использует
`pn-overlay` + `pn-dialog` стили как другие диалоги.

Смонтировать в `App.tsx` и добавить в `isModalOpen` guard.

---

## Edge cases

| Сценарий | Поведение |
|---|---|
| Полоса frozen (`enabled=false`) с высоким Q | Не маркируется (см. фильтр в `highQIndices`) |
| Полоса вне passband (выше LP) с Q > q_warn(f) | Маркируется (cap там жёстче, Q_MAX_ABOVE_LP=2.5, но логически — это всё равно high-Q) |
| Старый проект с PEQ полосами Q=10 на 1 кГц | После загрузки появятся маркировки. Полосы не меняются |
| Optimize → результирующие Q ≤ q_cap(f) на всех полосах | Маркировка может всё равно появиться (cap > warn). Это норма — оптимизатор не нарушил cap, но результат всё ещё «звенит» по нашей оценке. Пользователь видит warn |
| Cmd+Z после Optimize | b132 history восстанавливает peqBands → highQIndices пересчитывается реактивно |
| Изменение freq_hz одной полосы вручную (драг маркера) | После debounce — peqBands обновлены → highQIndices пересчитывается. Иконка может появиться/исчезнуть |
| Hybrid phase mode (max_boost = 60dB) | Q-cap применяется одинаково. Hybrid не освобождает от Q-cap |

---

## Acceptance

1. После Optimize Q каждой полосы внутри passband ≤ q_cap(freq).
2. На полосах с Q > q_warn(freq) видна жёлтая иконка `⚠` в списке PEQ
   и жёлтая обводка маркера на графике.
3. Клик по иконке `⚠` открывает popup с расшифровкой (частота, текущий
   Q, порог, текст рекомендации).
4. Закрытие popup по Escape / клику вне / кнопке.
5. Frozen (disabled) полоса не маркируется.
6. Загрузка старого `.pfproj` с Q=10 на 1кГц → полоса не меняется,
   но появляется маркер и иконка.
7. Cmd+Z после Optimize гасит маркеры и возвращает прежние Q.
8. Тесты `q_cap_at` и `q_warn_at` проходят: плато на краях,
   правильное значение в середине.
9. Optimize при HP=300, LP=3000 на замере с резонансом на 1кГц:
   результирующая полоса имеет Q ≤ 6.07 (cap@1000 Hz).

## Регрессионная проверка

- b131-b136 целы.
- Optimize / Optimize All — основная логика работает.
- Frozen bands механизм.
- b136 stale flag — корректно работает после изменения target.
- FIR Export.
- Q_MAX_ABOVE_LP = 2.5 для зон выше LP — не сломан.

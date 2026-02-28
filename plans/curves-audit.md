# Ревизия кривых: Band Mode vs SUM Mode

## Цветовые константы

| Константа | Значение |
|---|---|
| `MEASUREMENT_COLORS[i]` | Циклический набор (из `../lib/types`) |
| `TARGET_COLOR` | `#FFD700` (золотой) |
| `CORRECTED_COLOR` | `#22C55E` (зелёный) |
| `SMOOTHED_HALF_OCT_COLOR` | `#FF6B6B` (красный) |
| `TARGET_BAND_COLORS[i]` | `#7CB3FF, #FFB870, #6EE89A, #C98DF7, #F77070, #5CD6E8, #F7C35B, #F77DBF` |
| `CORRECTED_BAND_COLORS[i]` | `#34D399, #4ADE80, #2DD4BF, #A3E635, #86EFAC, #5EEAD4, #BEF264, #6EE7B7` |
| `FREQ_SNAP_COLORS[i]` | `#808080, #A855F7, #EC4899, #14B8A6` |

---

## Таблица 1: BAND MODE

| # | Series Label | Legend Label | Scale | Цвет | Категория | Visible | Данные | Условие |
|---|---|---|---|---|---|---|---|---|
| 1 | `{name} dB` | `Measurement` | mag | `MEAS_COLORS[0]` | measurement | `!isAlignTab` | Замер magnitude | `showMag && measurement` |
| 2 | `{name} °` | `Meas °` | phase | `MEAS_COLORS[0]` | measurement | `!isAlignTab` | Замер phase (wrapped) | `showPhase && measurement.phase` |
| 3 | `Target dB` | `Target` | mag | `#FFD700` | target | `true` | Таргет magnitude | `showTarget && targetMag` |
| 4 | `Target °` | `Target °` | phase | `#FFD700` | target | `true` | Таргет phase (wrapped; +180° если inverted) | `showTarget && showPhase && targetPhase` |
| 5 | `Meas 1/1 oct` | `Meas 1/1 oct` | mag | `#FF6B6B` | measurement | `false` | 1/1-октав сглаженный замер (PEQ intermediate) | `isAlignTab && showMag` |
| 6 | `PEQ Corrected dB` | `PEQ Corrected` | mag | `#22C55E` | corrected | `true` | **Замер + PEQ** (без XO) | `isHybrid && peqMag` |
| 7a | `Corrected + XO dB` | `Corrected + XO` | mag | `#F59E0B` (amber) | corrected | `true` | **Замер + PEQ + XO** (full) | `isHybrid && (hasPeq∥hasFilters)` |
| 7b | `Corrected dB` | `Corrected` | mag | `#22C55E` | corrected | `true` | **Замер + PEQ + XO** (full) | `!isHybrid && (hasPeq∥hasFilters)` |
| 8a | `Corrected + XO °` | `Corrected + XO °` | phase | `#F59E0B` | corrected | `true` | Full corrected phase (wrapped) | `isHybrid && showPhase && meas.phase` |
| 8b | `Corrected °` | `Corrected °` | phase | `#22C55E` | corrected | `true` | Full corrected phase (wrapped) | `!isHybrid && showPhase && meas.phase` |
| 9 | `{snap.label} dB` | `{snap.label}` | mag | `snap.color` | corrected | `true` | Снэпшот magnitude | per snapshot |
| 10 | `{snap.label} °` | `{snap.label} °` | phase | `snap.color` | corrected | `true` | Снэпшот phase | per snapshot, if has phase |

**Примечания Band Mode:**
- `isHybrid` = `exportHybridPhase()` — глобальный тогл Hybrid-φ
- 7a/7b и 8a/8b взаимоисключающие (Hybrid ИЛИ Standard)
- #6 (PEQ Corrected) только в Hybrid mode, только если есть PEQ bands
- `hasFilters` = target включён И есть HP/LP фильтры
- `hasPeq` = есть PEQ bands
- Формула corrected: `meas[i] + peqMag[i] + xsMag[i]`
  - `xsMag` из `compute_cross_section` = `filt_mag + makeup_mag`
  - `makeup_mag[i] = max(0, target[i] - (meas[i] + peq[i] + filt[i]))`

---

## Таблица 2: SUM MODE

| # | Series Label | Legend Label | Scale | Цвет | Категория | Visible | Данные | Условие |
|---|---|---|---|---|---|---|---|---|
| 1 | `{band.name} dB` | `{band.name}` | mag | `MEAS_COLORS[i]` | measurement | **false** | Per-band замер (resampled) | `showMag && resampled[i]` |
| 2 | `{band.name} °` | `{band.name} °` | phase | `MEAS_COLORS[i]` | measurement | **false** | Per-band замер phase (wrapped) | `showMag && showPhase && rm.phase` |
| 3 | `{band.name} tgt` | `{band.name} tgt` | mag | `TARGET_BAND_COLORS[i]` | target | **false** | Per-band таргет (ref = globalRef) | `showTarget && targetMag[i]` |
| 4 | `{band.name} corr+XO` | `{band.name} corr+XO` | mag | `CORRECTED_BAND_COLORS[i]` | corrected | **false** | **Per-band замер + PEQ + XO** | `showMag && resampled[i] && (hasPeq∥hasFilters)` |
| 5 | `Σ corr` | `Σ corrected` | mag | `#22C55E` | corrected | **true** | **Когерентная сумма** per-band corrected | `showMag && corrIndices.length > 0` |
| 6 | `Σ corr °` | `Σ corr °` | phase | `#22C55E` | corrected | **true** | Фаза когерентной суммы corrected | `showMag && showPhase && hasAllPhaseCorr` |
| 7 | `Σ dB` | `Σ meas` | mag | `#FFFFFF` | measurement | **true** | **Когерентная сумма** per-band замеров | `showMag && measIndices.length > 0` |
| 8 | `Σ °` | `Σ meas °` | phase | `#FFFFFF` | measurement | **true** | Фаза когерентной суммы замеров | `showMag && showPhase && hasAllPhase` |
| 9 | `Σ tgt` | `Σ target` | mag | `#FFD700` | target | **true** | **Когерентная сумма** нормализованных таргетов | `showTarget && enabledNorm.length > 0` |
| 10 | `Σ tgt °` | `Σ target °` | phase | `#FFD700` | target | **true** | Фаза когерентной суммы таргетов | `showTarget && showPhase && enabledNorm.length > 0` |

**Примечания SUM Mode:**
- Ряды 1-4 повторяются для КАЖДОГО бэнда (до N бэндов)
- Per-band (1-4) по умолчанию **скрыты**, Σ (5-10) по умолчанию **видны**
- **Нет per-band corrected phase** (только Σ corrected phase)
- **Нет PEQ Corrected** (отдельной кривой meas+PEQ без XO)
- **Нет снэпшотов** в SUM
- **Нет Hybrid/Standard разделения** — всегда одна corrected кривая per band
- Формула per-band corrected **идентична** band mode: `meas[i] + peqMag[i] + xsMag[i]`
- Σ corrected = когерентная комплексная сумма всех per-band corrected
- Σ meas = когерентная комплексная сумма всех per-band замеров
- Σ target = когерентная комплексная сумма нормализованных таргетов + avgRef

---

## Ключевые различия

| Аспект | Band Mode | SUM Mode |
|---|---|---|
| Замер | 1 кривая (текущая полоса) | Per-band (скрыты) + Σ meas (видна) |
| Таргет | 1 кривая (текущая полоса) | Per-band (скрыты) + Σ target (видна) |
| Corrected | PEQ Corrected¹ + Full Corrected | Per-band corr+XO (скрыты) + Σ corrected (видна) |
| Phase | Замер°, таргет°, corrected° | Per-band замер°, Σ meas°, Σ corr°, Σ target° |
| Снэпшоты | Да | Нет |
| 1/1 oct | Да (Align tab) | Нет |
| Hybrid split | Да (PEQ Corrected + Corrected+XO) | Нет |
| Видимость | Почти всё видно | Per-band скрыты, Σ видны |

¹ PEQ Corrected — только в Hybrid mode

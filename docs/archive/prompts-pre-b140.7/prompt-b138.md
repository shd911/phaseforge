# Промт для Code: b138 — защитный subsonic для Gaussian HP

ТЗ целиком: `docs/TZ-subsonic-protect.md`.
Текущий билд: 0.1.0-b137 → bump до 0.1.0-b138.

## Принципиальный выбор (зафиксировано)

- Защита **только для Gaussian HP**.
- Параметры **захардкожены**: 48 дБ/окт (Butterworth 8th order), 3 окт
  ниже основного freq_hz, минимально-фазовый.
- **Авто-вкл** при выборе Gaussian HP. Чекбокс отключения с warning
  под ним.
- Не применяется при `freq_hz ≤ 40` (subsonic уйдёт ниже 5 Гц).
- На графике сливается с основной target curve.
- **Не инвалидирует PEQ** (b136 stale flag игнорирует это поле).

## Что нужно сделать

### 1. Структура данных

**Rust `src-tauri/src/target/mod.rs`** — в `FilterConfig`:

```rust
#[serde(default)]
pub subsonic_protect: Option<bool>,
```

**Frontend `src/lib/types.ts`** — в `FilterConfig` interface:

```typescript
subsonic_protect?: boolean | null;
```

**Round-trip**: проверить `cloneFilter` в `project-io.ts:333` — добавить
поле в копирование. Аналогично в `mapBandToProject` / `mapBandFromProject`
если фильтры там разворачиваются вручную (вероятно нет — spread
работает).

### 2. Применение subsonic в Rust

В `apply_filter` (или функции, обрабатывающей HP — найти точное место,
вероятно `target/mod.rs:143`).

После расчёта Gaussian HP magnitude — если выполнены условия
`cfg.filter_type == FilterType::Gaussian && cfg.subsonic_protect == Some(true) && cfg.freq_hz > 40.0`:

```rust
let f_subsonic = cfg.freq_hz / 8.0;  // 3 octaves below
for i in 0..freq.len() {
    let f = freq[i];
    let ratio = (f_subsonic / f).powi(16);  // Butterworth 8th order: (f_c/f)^(2*N)
    let subsonic_mag = (1.0 / (1.0 + ratio)).sqrt();
    hp_mag[i] *= subsonic_mag;
}
```

**Phase** в evaluate_target для Gaussian уже = 0 (frontend применяет
Hilbert при `linear_phase=false`). Subsonic меняет magnitude → Hilbert
автоматически вернёт правильную min-phase. Никаких отдельных правок
phase в Rust не требуется.

Unit-тесты в том же файле:
- `subsonic_off_doesnt_affect_passband`: вход = HP Gaussian 100Hz без
  subsonic, выход в полосе 200…20000 Гц совпадает с baseline ±0.001 дБ.
- `subsonic_on_attenuates_infrasound`: HP Gaussian 100Hz + subsonic.
  На 6.25 Гц attenuation ≥ 40 дБ относительно passband.
- `subsonic_on_passband_unchanged`: HP Gaussian 100Hz + subsonic.
  На 200 Гц разница с baseline < 0.05 дБ.
- `subsonic_skipped_for_low_hp`: HP Gaussian 30Hz + subsonic_protect=true.
  Magnitude совпадает с baseline (защита не применилась).

### 3. UI: чекбокс

Найти где живёт выбор HP filter_type (вероятно `ControlPanel.tsx`,
секция target). Добавить чекбокс рядом или под полем выбора:

```jsx
<Show when={hp().filter_type === "Gaussian"}>
  <div class="subsonic-protect-row">
    <label>
      <input
        type="checkbox"
        checked={hp().subsonic_protect ?? false}
        disabled={hp().freq_hz <= 40}
        onChange={(e) => setSubsonicProtect(e.currentTarget.checked)}
      />
      <span>Защитный subsonic фильтр</span>
    </label>
    <Show when={hp().freq_hz <= 40}>
      <span class="hint" title="HP слишком низкий, защита не требуется">⊘</span>
    </Show>
    <Show when={hp().subsonic_protect === false && hp().freq_hz > 40}>
      <div class="warn">⚠ Защита отключена, риск excursion на инфразвуке</div>
    </Show>
  </div>
</Show>
```

CSS:
```css
.subsonic-protect-row {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  margin-top: var(--space-sm);
  font-size: 13px;
}
.subsonic-protect-row .warn {
  color: #d97706;
  font-size: 12px;
}
```

Tooltip на лейбле/чекбоксе:
«Минимально-фазовый Butterworth 48 дБ/окт на 3 октавы ниже HP.
Защищает driver от излишнего excursion в инфразвуке.»

### 4. Логика дефолта

При смене `filter_type` для HP:

```typescript
function setHpFilterType(newType: FilterType) {
  if (newType === "Gaussian") {
    hp.subsonic_protect = true;  // auto-ON
  } else {
    hp.subsonic_protect = null;
  }
}
```

Это вызывается в существующем handler смены filter_type. Найти его и
расширить логику.

При создании нового HP Gaussian (через UI с нуля) — тоже `subsonic_protect = true`.

### 5. Связь с b136

В `src/stores/peq-optimize.ts` — функция `filterEquals` (часть `peqStale`
логики). Убедиться что **не** сравнивает `subsonic_protect`:

```typescript
function filterEquals(a, b) {
  if (a === null && b === null) return true;
  if (a === null || b === null) return false;
  return a.filter_type === b.filter_type
      && a.order === b.order
      && a.freq_hz === b.freq_hz
      && a.shape === b.shape
      && a.q === b.q;
}
```

Если уже так — не трогать. Если случайно `subsonic_protect` попадает в
сравнение (например через JSON.stringify) — исключить.

### 6. Bump версии

- `src-tauri/tauri.conf.json` — version + productName/title до b138.
- `src-tauri/src/lib.rs` — startup-лог.
- После билда — skill `build-version`.

## Acceptance

1. Создаём Gaussian HP 100 Гц → чекбокс «Защитный subsonic» виден и
   включён. На графике target ниже 50 Гц видим более крутой спад,
   чем без защиты.
2. Снимаем чекбокс → spike ниже 50 Гц исчезает, target ниже плавный
   как в чистом Gaussian. Под чекбоксом появляется warning.
3. Меняем HP на LR4 → чекбокс пропадает, target — стандартный LR4.
4. Возвращаем HP на Gaussian → чекбокс снова появляется и ВКЛ.
5. Меняем HP freq_hz на 30 Гц → чекбокс disabled, иконка ⊘ с тултипом.
6. FIR export с subsonic_protect=true → файл .wav содержит коррекцию
   с включённым subsonic spade (визуально на графике exported FIR
   виден дополнительный крутой rolloff ниже cutoff/8).
7. Optimize при HP=100 Гц с subsonic_protect=true → результат не
   отличается от случая без subsonic в passband (оптимизатор работает
   выше peqLow, subsonic в зоне ниже peqFloor).
8. Изменение subsonic_protect (вкл/выкл) → peqStale остаётся false
   (b136 не реагирует).
9. Save → Open → чекбокс и target curve восстанавливаются точно.
10. Загрузка старого `.pfproj` без поля subsonic_protect → защита
    неактивна. Существующий проект ведёт себя как до b138.
11. Cargo unit-тесты проходят: 4 новых теста для subsonic + все
    существующие.

## Регрессионная проверка

- b131-b137 целы. Особое внимание b137 Q-cap (LMA penalty не должен
  деградировать).
- Optimize для HP типа LR / Butterworth — без изменений (subsonic не
  применяется).
- FIR export для всех типов HP (LR / Butterworth / Gaussian с/без
  subsonic).
- Hilbert min-phase reconstruction для Gaussian + subsonic
  (`linear_phase=false`) — phase в нижней зоне корректно отражает
  изменённую magnitude.

## Что НЕ трогать

- Существующая Gaussian математика (формулы LP/HP, shape coefficient).
- LR / Butterworth HP логика — без изменений.
- LP filter handling — subsonic не применяется к LP.
- `Q_MAX_ABOVE_LP`, `q_cap_at` (b137) — без изменений.

## Тестировать на `.dmg`

После сборки запустить
`src-tauri/target/release/bundle/dmg/PhaseForge_0.1.138_aarch64.dmg`
и пройти acceptance pp. 1-10. Не полагаться только на `cargo tauri dev`.

## Правила (CLAUDE.md)

- Один коммит: `feat: subsonic protection filter for Gaussian HP (b138)`
  + Co-Authored-By.
- 7-vector review.
- Без нарратива прогресса.
- Unit-тесты для Rust subsonic — обязательны.
- `cargo tauri build` для финальной сборки.

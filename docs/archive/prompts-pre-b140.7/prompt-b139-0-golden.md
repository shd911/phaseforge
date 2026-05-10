# Промт для Code: b139.0 — golden snapshot инфраструктура

ТЗ целиком: `docs/TZ-unified-evaluation.md`.
Текущий билд: 0.1.0-b138.4 → bump до 0.1.0-b139.0.

## Контекст

Запускается архитектурный refactor unified evaluation pipeline.
**Этап 0** — обязательная инфраструктура **до** любых изменений в
production коде. Создаётся golden snapshot reference на текущий output
pipeline (b138.4). Будущие этапы будут сравнивать новый код с этими
снимками — это гарантия что refactor не меняет наблюдаемое поведение.

В этом этапе **никаких изменений в production коде**. Только новые
тестовые файлы.

## Что нужно сделать

### 1. Manual regression checklist

Создать `docs/regression-checklist.md` с фиксированным списком из
10 пунктов которые прогоняются на `.dmg` после каждого этапа b139.x:

```markdown
# Regression Checklist для b139.x этапов

После каждого этапа запускать соответствующий PhaseForge_<version>.dmg
и проверить:

1. New Project с двумя полосами — Cmd+S сохраняет, Cmd+O открывает.
2. Импорт измерения (любой .txt) → analysis dialog появляется.
3. Gaussian HP=632 + linear_phase=true + subsonic ON: на SPL phase
   крутится в 5–40 Гц, в 200–2000 Гц = 0.
4. Gaussian HP=632 + linear_phase=false + subsonic ON: phase
   крутится в обеих зонах.
5. Optimize PEQ: получаем полосы с Q ≤ q_cap по формуле b137.
6. Изменение HP freq → оранжевый банер «PEQ устарел» (b136).
7. Cmd+Z после Optimize → возврат состояния (b132).
8. File → Versions → создать версию → восстановить (b133).
9. Cmd+Q при unsaved → диалог Save/Don't Save/Cancel (b131).
10. FIR Export → файл .wav сохранён, импульс в нём не пустой.

Все 10 пунктов должны проходить на каждом этапе b139.x. Любое
отклонение — diagnostic, не слепой фикс.
```

### 2. Reference fixtures для golden тестов

Создать `src/lib/__tests__/fixtures/eval-fixtures.ts`:

Эта библиотека генерирует **детерминированные** входные данные для
6 эталонных конфигураций. Никакого случайного шума, никаких
зависимостей от системного времени.

```typescript
import type { BandState, FilterConfig, TargetCurve } from "../../types";
// или соответствующие импорты

/** Synthetic measurement: log-spaced freq grid, smooth response. */
export function fixtureMeasurement(): { freq: number[]; magnitude: number[]; phase: number[] } {
  const freq: number[] = [];
  const magnitude: number[] = [];
  const phase: number[] = [];
  const n = 512;
  for (let i = 0; i < n; i++) {
    const f = 20 * Math.pow(20000 / 20, i / (n - 1));
    freq.push(f);
    // Smooth response: -3 dB at 50 Hz, flat to 5kHz, -6 dB/oct above
    let mag = 0;
    if (f < 50) mag = -3 - 12 * Math.log2(50 / f);
    else if (f > 5000) mag = -6 * Math.log2(f / 5000);
    magnitude.push(mag);
    phase.push(0);
  }
  return { freq, magnitude, phase };
}

export function fixtureGaussianHP(linearPhase: boolean, subsonicProtect: boolean | null): FilterConfig {
  return {
    filter_type: "Gaussian",
    order: 4,
    freq_hz: 632,
    shape: 1.0,
    linear_phase: linearPhase,
    q: null,
    subsonic_protect: subsonicProtect,
  };
}

export function fixtureLR4HP(): FilterConfig {
  return {
    filter_type: "LinkwitzRiley",
    order: 4,
    freq_hz: 80,
    shape: null,
    linear_phase: false,
    q: null,
    subsonic_protect: null,
  };
}

/** Six canonical configurations referenced in TZ. */
export const FIXTURE_CONFIGS = [
  { name: "gaussian_lin_subsonic_off", hp: fixtureGaussianHP(true, false), label: "1. Gaussian linear, subsonic OFF" },
  { name: "gaussian_lin_subsonic_on",  hp: fixtureGaussianHP(true, true),  label: "2. Gaussian linear, subsonic ON" },
  { name: "gaussian_min_subsonic_off", hp: fixtureGaussianHP(false, false), label: "3. Gaussian min-phase, subsonic OFF" },
  { name: "gaussian_min_subsonic_on",  hp: fixtureGaussianHP(false, true),  label: "4. Gaussian min-phase, subsonic ON" },
  { name: "lr4_baseline",              hp: fixtureLR4HP(),                  label: "5. LR4 baseline (не Gaussian)" },
  { name: "no_hp_fullrange",           hp: null,                            label: "6. No HP (full-range)" },
] as const;
```

### 3. Golden snapshot тест на `evaluateBand` + `addGaussianMinPhase`

Создать `src/lib/__tests__/golden-pipeline.test.ts`:

Тест прогоняет каждую из 6 конфигураций через **существующий** pipeline
(`evaluateBand` или его эквивалент с `addGaussianMinPhase`) и сравнивает
result с золотым снимком.

Поскольку `evaluateBand` требует измерение и invoke calls — мокаем
invoke и используем helpers напрямую:

```typescript
import { describe, it, expect, vi } from "vitest";
import { fixtureMeasurement, FIXTURE_CONFIGS } from "./fixtures/eval-fixtures";
import { gaussianFilterMagDb, subsonicMagDb } from "../plot-helpers";

// Что именно проверяем — pipeline phase reconstruction для Gaussian +
// subsonic. Это фокус b138 каскада, и регрессии здесь критичны.

describe("golden pipeline (b139.0 reference for refactor)", () => {
  for (const cfg of FIXTURE_CONFIGS) {
    it(`${cfg.label} — magnitude snapshot`, () => {
      const meas = fixtureMeasurement();

      // Compute reference magnitude using current b138.4 helpers
      let mag = [...meas.magnitude];

      if (cfg.hp && cfg.hp.filter_type === "Gaussian") {
        const hpMag = gaussianFilterMagDb(meas.freq, cfg.hp, false);
        mag = mag.map((m, i) => m + hpMag[i]);

        if (cfg.hp.subsonic_protect === true && cfg.hp.freq_hz > 40) {
          const subDb = subsonicMagDb(meas.freq, cfg.hp.freq_hz / 8);
          mag = mag.map((m, i) => m + subDb[i]);
        }
      }

      // Round to fixed precision to make snapshots stable
      const rounded = mag.map(m => Math.round(m * 1e6) / 1e6);
      expect(rounded).toMatchSnapshot();
    });
  }
});
```

### 4. Cargo snapshot тест на `evaluate_target`

В `src-tauri/src/target/mod.rs` (внутри `mod tests`) добавить тест
который вычисляет magnitude для 6 эталонных HP конфигураций на
фиксированной freq grid и сравнивает с захардкоженным reference
массивом.

Пример формата:

```rust
#[test]
fn evaluate_target_b139_golden_gaussian_lin_subsonic_off() {
    let target = TargetCurve {
        reference_level_db: 0.0,
        tilt_db_per_oct: 0.0,
        high_pass: Some(FilterConfig {
            filter_type: FilterType::Gaussian,
            order: 4,
            freq_hz: 632.0,
            shape: Some(1.0),
            linear_phase: true,
            q: None,
            subsonic_protect: Some(false),
        }),
        low_pass: None,
        // shelves если есть — все zero
    };
    // Standard log grid 20 Hz – 20 kHz, 32 точек (компактно)
    let freq: Vec<f64> = (0..32).map(|i| 20.0 * (1000f64).powf(i as f64 / 31.0)).collect();
    let resp = evaluate(&target, &freq);

    // Reference values: захардкожены из текущего (b138.4) output.
    // При написании теста — запустить один раз, скопировать output,
    // округлить до 6 знаков, вставить сюда.
    let expected_mag: Vec<f64> = vec![/* 32 значения с 6 знаками */];

    for (i, (a, b)) in resp.magnitude.iter().zip(expected_mag.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5,
            "mag mismatch at i={}, freq={:.1}: got {}, expected {}",
            i, freq[i], a, b);
    }
}
```

Аналогично 5 других тестов для остальных конфигураций.

**Важно по процессу:** при первом запуске Code-сессия выводит actual
значения через `eprintln!`, копирует их обратно в `expected_mag`, потом
`assert` подтверждается. Это и есть «фиксация reference».

### 5. SHA-256 hash для FIR reference

Файл `src-tauri/src/fir/mod.rs` — добавить тест который:

1. Генерирует FIR для одной эталонной конфигурации (LR4 HP=80, без
   subsonic, без PEQ, sample_rate=48000, taps=8192).
2. Применяет FFT к impulse, берёт magnitude.
3. Округляет до 6 знаков, JSON-stringify, SHA-256 hash.
4. Сравнивает с reference hash (тоже захардкожен из b138.4 output).

```rust
#[test]
fn generate_fir_b139_golden_lr4_baseline_spectrum_hash() {
    let target_mag: Vec<f64> = /* эталонная LR4 HP=80 magnitude на freq grid 5–40k, 512 точек */;
    let freq: Vec<f64> = /* freq grid */;
    let config = /* стандарт */;
    let result = generate_model_fir(/* args */);

    // FFT magnitude → round → hash
    let spectrum_hash = hash_spectrum(&result.impulse);
    let expected_hash = "abc123..." /* зафиксировано из первого прогона */;
    assert_eq!(spectrum_hash, expected_hash);
}

fn hash_spectrum(impulse: &[f64]) -> String {
    // FFT, magnitude, round, serialize, sha256
    // Use sha2 crate (вероятно уже подключён) или ring
}
```

Если `sha2` crate не подключён — добавить в `Cargo.toml`. Если
подключение проблемно — использовать встроенный
`std::collections::hash_map::DefaultHasher` (менее надёжно но не
требует deps).

### 6. Bump версии

- `src-tauri/tauri.conf.json` — version + productName/title до b139.0.
- `src-tauri/src/lib.rs` — startup-лог.
- После билда — skill `build-version`.

## Acceptance

1. `docs/regression-checklist.md` создан с 10 пунктами.
2. `src/lib/__tests__/fixtures/eval-fixtures.ts` создан с 6 конфигурациями.
3. `src/lib/__tests__/golden-pipeline.test.ts` создан, все snapshots
   сгенерированы (`npm test -- -u` для первого прогона), сохранены
   в `__snapshots__` и закоммичены.
4. `npm test` зелёный — все snapshots совпадают.
5. В `src-tauri/src/target/mod.rs` добавлены 6 cargo тестов
   `evaluate_target_b139_golden_*`, все проходят.
6. В `src-tauri/src/fir/mod.rs` добавлен `generate_fir_b139_golden_*`
   spectrum hash test, проходит.
7. `cargo test` зелёный.
8. **`.dmg b139.0` собран и regression-checklist (10 пунктов) пройден
   на нём.**

## Регрессионная проверка

Поскольку **никаких изменений в production коде нет**, regression
сводится к проверке что:
- `cargo test` зелёный (включая 7 новых тестов)
- `npm test` зелёный (включая новые snapshot тесты)
- `.dmg` собирается
- regression-checklist.md из 10 пунктов проходит на `.dmg b139.0`

## Что НЕ трогать

- Никакой логики в `src/lib/band-evaluation.ts`, `fir-export.ts`,
  `peq-optimize.ts`, `bands.ts` — только новые тестовые файлы.
- Никаких изменений в Rust commands (только новые тесты внутри
  `mod tests`).
- Никакого нового functionality на frontend — только tests.

## Учёт уроков b138

1. **Audit before write.** Перед созданием fixture файла grep на
   существующие fixture файлы (`find src -name "fixtures*"`) — если
   уже есть, не создавать дубликат.
2. **Diagnostic-first при провале.** Если cargo тест fail при первом
   прогоне — это ожидаемо (нужно зафиксировать reference). НЕ pre-fix
   значения вслепую — собрать actual из теста через `eprintln!`,
   проверить разумность диапазонов, потом захардкодить.
3. **Версия в заголовке** = b139.0 — обязательно проверить на `.dmg`
   перед прогоном checklist.

## Тестировать на `.dmg`

После сборки — запустить
`src-tauri/target/release/bundle/dmg/PhaseForge_0.1.139_aarch64.dmg`,
проверить версию в заголовке = b139.0, прогнать
`docs/regression-checklist.md` все 10 пунктов.

## Правила (CLAUDE.md)

- Один коммит: `test: golden snapshots for unified eval refactor (b139.0)`
  + Co-Authored-By.
- 7-vector review.
- Без нарратива прогресса.
- `cargo tauri build` для финальной сборки.

# Промт для Code: b139.5.3 — FIR использует свою standalone сетку, не measurement

**Тип:** регрессионный фикс. Bump до 0.1.0-b139.5.3.

## Контекст

В b139.4c унификация на BandEvaluator заставила FIR pipeline
использовать ту же freq grid что и SPL view (сетку измерения).

Логи реального экспорта:
```
evaluate_target: 484 points  ← measurement grid, 20 Hz – 20 kHz
generate_model_fir: 484 points, taps=65536, sr=176400, phase_mode=Composite
realized_max=25.71 dB → normalizing by 25.71 dB
```

Раньше (legacy `generateBandImpulse`):
```rust
evaluate_target_standalone: nPoints=512, fMin=5, fMax=40000
```

То есть legacy FIR использовал **standalone grid 5–40000 Гц, 512
точек, равномерный по log**. Сейчас FIR получает measurement grid
(20 Гц – 20 кГц, неравномерный).

Это нарушает FIR generation:
- Нижняя граница 20 Гц вместо 5 → subsonic rolloff обрезается.
- Верхняя 20 кГц вместо 40 кГц → нет резерва выше 22.05 кГц для
  sr=176.4 кГц, что критично для anti-aliasing.
- Неравномерный шаг → искажение Gaussian rolloff.

Регрессия: основной фильтр (Gaussian HP) на экспорте не
реализуется правильно.

## Что нужно сделать

### 1. В `band-evaluator.ts:evaluateBandFull`

Когда задан `req.fir`, для FIR ветки использовать **separate**
freq grid, не measurement. Вариант реализации:

```typescript
// Если запрошен FIR — используем standalone grid 5–40k для FIR pipeline,
// независимо от measurement. Это нужно для корректной реализации
// HP/LP rolloff и anti-aliasing запаса до Найквиста.
let firFreq: number[] | null = null;
let firTargetMag: number[] | null = null;
let firPeqMag: number[] | null = null;
let firCombinedPhase: number[] | null = null;

if (req.fir) {
  const sr = req.fir.sampleRate;
  const fMax = Math.min(40000, sr / 2 * 0.95);  // не выше Nyquist*0.95
  const [standaloneFreq, response] = await invoke<[number[], TargetResponse]>(
    "evaluate_target_standalone",
    { target: targetCurve, nPoints: 512, fMin: 5, fMax },
  );
  firFreq = standaloneFreq;
  firTargetMag = response.magnitude;
  const firTargetPhase = await reconstructTargetPhase(
    firFreq, response.phase, band.target.high_pass, band.target.low_pass,
  );

  // PEQ contribution на FIR grid
  if (enabledPeq.length > 0) {
    const [pm, pp] = await invoke<[number[], number[]]>("compute_peq_complex", {
      freq: firFreq, bands: enabledPeq,
    });
    firPeqMag = pm;
    firCombinedPhase = firTargetPhase.map((p, i) => p + pp[i]);
  } else {
    firPeqMag = new Array(firFreq.length).fill(0);
    firCombinedPhase = firTargetPhase;
  }

  // ...generate_model_fir с firFreq, firTargetMag, firPeqMag, firCombinedPhase
}
```

То есть FIR pipeline ВНУТРИ evaluateBandFull использует свой freq
grid через дополнительный invoke `evaluate_target_standalone`.
Display pipeline (target/corrected/peq на SPL) остаётся на
measurement grid.

Это **правильное** разделение: SPL plot показывает что слышит
человек (на сетке измерения), FIR строится на инженерной grid с
запасом ниже 20 Hz и выше 20 кГц.

### 2. Аудит legacy generateBandImpulse

Сравнить параметры старого вызова `evaluate_target_standalone` с
новым:

```
legacy: nPoints=512, fMin=5, fMax=40000
new:    nPoints=512, fMin=5, fMax=min(40000, sr/2 * 0.95)
```

Если что-то ещё передавалось в legacy (например другие настройки
target) — учесть.

### 3. Тесты

#### Cargo

Добавить cargo тест который воспроизводит сценарий:

```rust
#[test]
fn fir_uses_wide_grid_for_proper_rolloff() {
    // Gaussian HP=632 на standalone grid 5-40k.
    // FIR должен реализовать magnitude из target в зоне 5-40000 Hz.
    // Проверить что magnitude FIR в зоне 5-15 Hz сильно ослаблена.
    let freq: Vec<f64> = dsp::generate_log_freq_grid(512, 5.0, 40000.0);
    // ... evaluate target с Gaussian HP ...
    // ... generate FIR ...
    // ... assert magnitude_at(7 Hz) < -30 dB ...
}
```

#### Vitest

Snapshot тест на `evaluateBandFull({band, fir: {...}})` —
result.fir.realizedMag должен иметь ожидаемый rolloff.

### 4. Bump

- `src-tauri/tauri.conf.json` → `0.1.139` (numeric).
- `src-tauri/src/lib.rs` startup-лог: b139.5.3.
- `src/lib/version.ts` → `0.1.0-b139.5.3`.
- skill `build-version`.

## Acceptance

1. На `.dmg b139.5.3` импортировать flat measurement, поставить
   Gaussian HP=1000, defaults, Export FIR.
2. В Rust логах:
   ```
   evaluate_target_standalone: 512 points    ← FIR pipeline
   generate_model_fir: 512 points
   ```
3. Импортировать сгенерированный `.wav` обратно через Import.
4. На SPL plot импортированной FIR-кривой видно Gaussian HP rolloff
   ниже 1000 Hz (-3 dB на 1000, -20 dB на 300, и т.д.) — то что
   было задано в target.
5. Раньше (b139.5.2) на этом импорте rolloff не был виден.

## Регрессионная проверка

- 168+ existing cargo тестов PASS.
- vitest 140+ PASS.
- regression-checklist 5 manual UI пунктов.
- b139.5.1 (CR-only parser) и b139.5.2 (self-copy fix) не сломаны.

## Что НЕ делать

- Не менять SPL pipeline — он должен оставаться на measurement grid.
- Не менять PEQ optimizer (он использует measurement grid отдельно).
- Не удалять `evaluate_target_standalone` — теперь используется в
  evaluateBandFull для FIR ветки.

## Правила

- Один коммит: `fix: FIR uses standalone wide grid, not measurement (b139.5.3)`
  + Co-Authored-By.
- 7-vector review.
- Cargo + vitest тесты обязательны.
- Без нарратива.

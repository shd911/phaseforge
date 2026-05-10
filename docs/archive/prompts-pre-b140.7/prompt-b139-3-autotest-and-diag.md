# Промт для Code: b139.3-test — автоматизация регрессии + диагностика

**Тип:** расширение test suite + диагностика бага. Bump до 0.1.0-b139.3.1.

## Контекст и цели

1. **Баг (приоритет):** в b139.3 защитный subsonic переводит весь
   экспорт в min-phase, включая основной фильтр. Flat test
   measurement не даёт identity FIR — exported кривая не совпадает
   с target.

2. **Автоматизация:** ручной regression-checklist (10 пунктов)
   слишком долгий. Перевести максимум на автотесты, manual оставить
   только то что требует UI workflow.

Подход: написать **автотесты которые воспроизводят описанный баг**.
Если они fail — баг локализован без UI. Заодно эти тесты становятся
постоянной защитой от регрессии.

## Что нужно сделать

### 1. Cargo: identity FIR test (главный детектор бага)

В `src-tauri/src/fir/mod.rs` добавить:

```rust
#[test]
fn fir_identity_for_flat_input_no_filters() {
    // Flat measurement (0 dB) + flat target (no HP, LP, shelves, tilt) +
    // no PEQ → FIR должен быть identity (одна точка в позиции delay).
    let n = 512;
    let freq: Vec<f64> = log_grid(n, 5.0, 40000.0);
    let target_mag: Vec<f64> = vec![0.0; n];      // flat 0 dB
    let target_phase: Vec<f64> = vec![0.0; n];    // zero phase
    let peq_mag: Vec<f64> = vec![0.0; n];

    let cfg = FirConfig {
        taps: 8192, sample_rate: 48000.0,
        max_boost_db: 24.0, noise_floor_db: -150.0,
        window: WindowType::Blackman,
        phase_mode: PhaseMode::LinearPhase,
        iterations: 3, freq_weighting: true,
        narrowband_limit: false,
        nb_smoothing_oct: 0.333,
        nb_max_excess_db: 6.0,
    };
    let result = generate_model_fir(&freq, &target_mag, &peq_mag, &target_phase, &cfg);

    // Identity check: один пик ~1.0, остальное ~0.
    let peak_idx = result.impulse.iter().enumerate()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
        .unwrap().0;
    let peak_val = result.impulse[peak_idx];
    let energy_off_peak: f64 = result.impulse.iter().enumerate()
        .filter(|(i, _)| *i != peak_idx)
        .map(|(_, v)| v * v).sum();

    assert!((peak_val.abs() - 1.0).abs() < 0.01,
        "Identity FIR должна иметь peak ≈1.0, got {}", peak_val);
    assert!(energy_off_peak < 0.01,
        "Identity FIR должна иметь почти всю энергию в peak, got off-peak energy {}", energy_off_peak);
}

#[test]
fn fir_identity_with_min_phase_mode() {
    // То же что выше, но MinimumPhase mode.
    // Должно быть identity (один пик в начале).
    // ... аналогично, но phase_mode: MinimumPhase
}
```

Если эти тесты **fail** — это автоматический детектор бага «не
identity для flat input». Если **pass** — баг где-то в frontend
chain (evaluateBandFull / fir-export.ts передаёт неправильную phase).

### 2. Cargo: subsonic не должен делать FIR минфазовым целиком

```rust
#[test]
fn fir_linear_gaussian_with_subsonic_keeps_passband_linear_phase() {
    // Linear-phase Gaussian HP=632 + subsonic ON.
    // В passband (1k-10k Hz) FIR должен оставаться linear-phase
    // (центрированный, симметричный относительно середины taps).
    // В infrasound зоне (5-40 Hz) — допустима min-phase rotation от subsonic.

    // Сгенерировать target_mag с Gaussian HP применённым.
    // Сгенерировать target_phase: 0 в passband (linear), Hilbert(subsonic) в infra.
    // Generate FIR.
    // Проверить:
    //   FFT(impulse) → magnitude в passband ≈ target_mag в passband (с допуском)
    //   FFT(impulse) → phase в passband ≈ 0 (linear) или строго линейная по freq
    //   FFT(impulse) → phase в зоне 5-40 Hz: нелинейная (subsonic min-phase)
}
```

### 3. Vitest: snapshot тесты на полный output evaluateBandFull с FIR

В `src/lib/__tests__/band-evaluator.test.ts` (или новый
`band-evaluator-fir.test.ts`) — добавить тесты:

```typescript
describe("evaluateBandFull with FIR — flat input", () => {
  it("flat measurement, no filters, no PEQ → identity FIR", async () => {
    const band = makeFlatBand();  // synthetic flat measurement
    const result = await evaluateBandFull({
      band,
      includeFir: true,
    });

    expect(result.fir).toBeDefined();
    const impulse = result.fir!.impulse;

    // Find peak
    let peakIdx = 0, peakVal = 0;
    for (let i = 0; i < impulse.length; i++) {
      if (Math.abs(impulse[i]) > Math.abs(peakVal)) {
        peakVal = impulse[i];
        peakIdx = i;
      }
    }
    expect(Math.abs(peakVal - 1.0)).toBeLessThan(0.01);

    // Off-peak energy
    let offPeakEnergy = 0;
    for (let i = 0; i < impulse.length; i++) {
      if (i !== peakIdx) offPeakEnergy += impulse[i] * impulse[i];
    }
    expect(offPeakEnergy).toBeLessThan(0.01);
  });
});
```

Если этот тест fail — это **frontend**-чейн виноват
(evaluateBandFull передаёт неправильную phase или mag в Rust). Если
pass — баг в Rust (test 1 cargo тоже fail).

### 4. Сокращение manual regression-checklist

Обновить `docs/regression-checklist.md` — оставить только UI
workflow пункты (которые нельзя автоматизировать без e2e framework):

```markdown
# Regression Checklist (manual UI workflow)

DSP-логика покрыта автотестами (cargo + vitest). Этот checklist —
только UI workflow которые требуют запуска приложения.

После сборки `.dmg`:

1. Версия в заголовке окна совпадает с релизом.
2. New Project → две полосы → Cmd+S → Cmd+O возвращает то же состояние.
3. Cmd+Q при unsaved → диалог Save/Don't Save/Cancel (b131).
4. Импорт измерения → analysis dialog появляется (b135).
5. File → Versions → создать версию → восстановить (b133).

Все пункты которые требуют DSP-валидации (phase reconstruction,
FIR коэффициенты, Q envelope, etc.) — автоматизированы. Локально
запустить:
- `cargo test` — все Rust DSP тесты
- `npm test` — все frontend тесты

Если автотесты падают — DSP regression. Если manual UI checklist
падает — UI/state regression.
```

### 5. Диагностический лог в evaluateBandFull (временно, для текущего бага)

В `src/lib/band-evaluator.ts` внутри `evaluateBandFull` если
`includeFir`:

```typescript
console.log("[BandEval:FIR]", {
  bandName: band.name,
  hp: band.target.high_pass,
  hasSubsonic: hasActiveSubsonicProtect(band.target.high_pass),
  targetMag_at_1kHz: targetMag[freq.findIndex(f => f >= 1000)],
  targetPhase_at_1kHz: combinedTargetPhase[freq.findIndex(f => f >= 1000)],
  peqMag_nonzero: peqMag.some(v => v !== 0),
  phaseMode: /* что передаётся в Rust */,
});
```

После того как cargo и vitest тесты подтвердят источник проблемы —
эти логи откатить (отдельным mini-патчем перед коммитом b139.3.2).

### 6. Bump версии

- `src-tauri/tauri.conf.json` — version + productName/title до b139.3.1.
- `src-tauri/src/lib.rs` — startup-лог.
- После билда — skill `build-version`.

## Acceptance

1. **Cargo тест `fir_identity_for_flat_input_no_filters`** добавлен.
   Если pass — Rust pipeline корректен для identity case. Если fail —
   баг в Rust (требует фикс в b139.3.2).

2. **Cargo тест `fir_linear_gaussian_with_subsonic_keeps_passband_linear_phase`**
   добавлен. Должен pass — это закрывает заявленный Кирилла баг
   автотестом.

3. **Vitest тест на flat measurement → identity FIR через
   evaluateBandFull** добавлен. Pass или fail подскажет где баг
   (frontend vs Rust).

4. **`docs/regression-checklist.md` сокращён** до 5 UI пунктов.
   Старые DSP-пункты заменены автотестами.

5. **Диагностический лог в evaluateBandFull** добавлен (временно).

6. После прогона `cargo test` + `npm test` — отчитаться **какие
   тесты прошли, какие упали**. Это и есть локализация бага.

## Что НЕ делать

- Не фиксить баг вслепую. Сначала тесты должны его поймать.
- Не удалять старый regression-checklist полностью — только заменить
  его новой UI-only версией.
- Не менять Rust generate_model_fir пока тесты не покажут что и как.
- Не делать e2e UI testing инфраструктуру (большой scope, отложить).

## Что прислать обратно

После применения промта и прогона тестов — отчёт:

```
cargo test:
  fir_identity_for_flat_input_no_filters: PASS / FAIL (reason)
  fir_linear_gaussian_with_subsonic_keeps_passband_linear_phase: PASS / FAIL
  fir_identity_with_min_phase_mode: PASS / FAIL
  (existing 154 tests): PASS / FAIL count

vitest:
  evaluateBandFull flat → identity: PASS / FAIL
  (existing 135 tests): PASS / FAIL count
```

Если что-то fail — приложить **конкретные числовые расхождения**
(actual vs expected). Это даст следующий промт диагностики или фикса
точечный.

## Тестировать на `.dmg`

После сборки запустить
`PhaseForge_0.1.139-3-1_aarch64.dmg`, версия = b139.3.1.

Воспроизвести Кирилла сценарий:
- Импорт flat measurement (например `0_db_flat.txt` если есть, или
  создать synthetic).
- Default target (no HP/LP).
- Export FIR.
- Импортировать .wav обратно через Import.
- Сравнить magnitude и phase импортированного FIR с измерением.

Должны совпадать (FIR не должен менять flat input). Если не
совпадают — это и есть баг которые автотесты должны поймать.

## Правила (CLAUDE.md)

- Один коммит: `test: identity FIR + subsonic regression tests (b139.3.1)`
  + Co-Authored-By.
- 7-vector review.
- Без нарратива прогресса.
- Cargo и vitest тесты — основа промта, не дополнение.

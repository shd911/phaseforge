# Промт для Code: b140.3.4 — corrected IR на widе grid с extension

Текущий билд: 0.1.0-b140.3.3 → bump до 0.1.0-b140.3.4.

## Самооценка эффективности

| Критерий | Оценка | Комментарий |
|---|---|---|
| Размер | ⚠️ средний | Расширение IR ветки + повторное вычисление PEQ/XS на широком grid |
| Pre-flight audit | ✅ | Аналог b140.3.3 для corrected. Extension механизм отлажен (b140.3.1.5 / b140.3.2) |
| Гипотезы без данных | ✅ | DSP стандарт |

## Контекст

В b140.3.3 target IR использует широкий standalone grid 5–40k для
видимости subsonic в impulse. Для corrected IR такой же фикс нужен —
сейчас corrected IR на measurement grid (20–22000 Гц), subsonic
эффект не виден.

Corrected IR требует extension measurement через target shape
(аналог b140.3.1.5 / b140.3.2) и пересчёт PEQ + cross-section на
широком grid.

## Что нужно сделать

### В `src/lib/band-evaluator.ts`, IR ветка (~ строка 398, после target IR из b140.3.3)

```typescript
// b140.3.4: corrected IR на том же широком grid что target IR.
// extension measurement через target shape + Hilbert phase.
if (band.targetEnabled && measurement) {
  // irFreq уже определён из target IR ветки (5-40k log grid)

  // Extension measurement на irFreq через target shape (helper computeExtension)
  const extMeas = await computeExtension(
    measurement.freq, measurement.magnitude, measurement.phase ?? null,
    irFreq, irTargetResp.magnitude,
  );

  // PEQ и cross-section на широком grid
  const enabledPeq = (band.peqBands ?? []).filter(p => p.enabled);
  const [irPeqMag, irPeqPhase] = enabledPeq.length > 0
    ? await invoke<[number[], number[]]>("compute_peq_complex",
        { freq: irFreq, bands: enabledPeq })
    : [new Array(irFreq.length).fill(0), new Array(irFreq.length).fill(0)];

  let irXsMag = new Array(irFreq.length).fill(0);
  let irXsPhase = new Array(irFreq.length).fill(0);
  if (band.target.high_pass || band.target.low_pass) {
    const xs = await invoke<{magnitude: number[]; phase: number[]}>(
      "compute_cross_section",
      { freq: irFreq, target: band.target, normDb: 0 },
    );
    irXsMag = xs.magnitude;
    irXsPhase = xs.phase;
  }

  // Corrected = extended_meas + peq + xs
  const irCorrMag = extMeas.mag.map((m, i) =>
    m + irPeqMag[i] + irXsMag[i]);
  const irCorrPhase = (extMeas.phase ?? new Array(irFreq.length).fill(0))
    .map((p, i) => p + irPeqPhase[i] + irXsPhase[i]);

  const r = await invoke<{time, impulse, step}>("compute_impulse",
    { freq: irFreq, magnitude: irCorrMag, phase: irCorrPhase, sampleRate: sr });
  ir.corrected = { time: r.time, impulse: r.impulse, step: r.step };
}
```

`computeExtension` — уже существует (b140.3.1.5 / b140.3.2).
`irFreq`, `irTargetResp` — уже вычислены в target IR ветке b140.3.3.

### Если у полосы НЕТ measurement

Corrected IR не считаем (corrected требует measurement).
`ir.corrected = undefined`.

### Vitest

```typescript
describe("evaluateBandFull — corrected IR uses wide grid", () => {
  it("subsonic toggle changes corrected IR", async () => {
    // Band с measurement + Gaussian HP + subsonic
    // ON vs OFF: corrected impulse должен отличаться
    const irOff = await evaluateBandFull({ band: bSubOff, includeIr: true });
    const irOn = await evaluateBandFull({ band: bSubOn, includeIr: true });
    expect(maxAbsDiff(irOff.ir!.corrected!.impulse,
                      irOn.ir!.corrected!.impulse)).toBeGreaterThan(0.01);
  });

  it("PEQ contribution visible in corrected IR", async () => {
    // Band с PEQ полосой → impulse меняется относительно без PEQ
  });

  it("no measurement → no corrected IR", async () => {
    // Band без measurement → ir.corrected undefined
  });
});
```

### Bump

- `src-tauri/tauri.conf.json` → `0.1.140`.
- `src-tauri/src/lib.rs` startup → b140.3.4.
- `src/lib/version.ts` → b140.3.4.

## Acceptance

1. На band view IR/Step при включении защитного subsonic →
   **corrected** impulse и step меняются (subsonic виден).
2. PEQ изменения отражаются в corrected IR.
3. Linear-phase target + subsonic → corrected impulse асимметричный
   (linear main + min-phase subsonic).
4. Min-phase target + subsonic → corrected — длинный low-freq decay.
5. Measurement IR не меняется (на measurement grid, как было).
6. existing 64+ vitest + 3 новых corrected IR PASS.

## Что НЕ делать

- Не менять measurement IR pipeline (на measurement grid, как было).
- Не трогать SUM IR (отдельная задача).

## End-of-prompt автозапуск dev

```
osascript -e 'tell application "PhaseForge" to quit' 2>/dev/null || true
pkill -9 -f -i "phaseforge" 2>/dev/null || true
pkill -9 -f "tauri dev" 2>/dev/null || true
pkill -9 -f "tauri-driver" 2>/dev/null || true
sleep 1
lsof -ti:1420 | xargs kill -9 2>/dev/null || true
lsof -ti:5173 | xargs kill -9 2>/dev/null || true
sleep 1
cd /Users/olegryzhikov/phaseforge && nohup cargo tauri dev > /tmp/phaseforge-dev.log 2>&1 &
```

## Правила

- Один коммит: `fix: corrected IR on wide grid with extension (b140.3.4)` + Co-Authored-By.
- Без нарратива.

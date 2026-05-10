# Промт для Code: b140.3.3 — target IR на widе standalone grid

Текущий билд: 0.1.0-b140.3.2 → bump до 0.1.0-b140.3.3.

## Самооценка эффективности

| Критерий | Оценка | Комментарий |
|---|---|---|
| Размер | ✅ малый | Точечная правка в `evaluateBandFull`, IR ветка |
| Pre-flight audit | ✅ | Аудит сделан, точно знаем где правка |
| Гипотезы без данных | ✅ | Тот же подход что для FIR в b139.5.3 |

## Контекст

Аудит подтвердил: target IR в band view не реагирует на включение
защитного subsonic. Корень — target IR использует measurement freq
grid (20–22 кГц), а subsonic работает в зоне 5–79 Гц (cutoff/8).
Большая часть subsonic эффекта (40+ dB attenuation на 5–15 Гц) не
входит в grid → impulse визуально идентичен.

Решение: для target IR использовать widе standalone grid 5–min(40k, sr/2·0.95)
**всегда**, независимо от measurement. Target — это model, она не должна
зависеть от диапазона замера. Measurement IR и Corrected IR — на
measurement grid (там реальные данные).

## Что нужно сделать

### В `src/lib/band-evaluator.ts`, IR ветка (~ строка 398)

Сейчас target IR использует общий `freq` (== measurement.freq):

```typescript
if (targetMag && targetPhase) {
  const r = await invoke("compute_impulse", {
    freq, magnitude: targetMag, phase: targetPhase, sampleRate: sr,
  });
  ir.target = { time: r.time, impulse: r.impulse, step: r.step };
}
```

Заменить на отдельный standalone grid для target IR:

```typescript
// b140.3.3: target IR использует widе standalone grid (5-40k или Nyquist*0.95)
// независимо от measurement, чтобы subsonic / supersonic поведение
// корректно отображалось в impulse.
if (band.targetEnabled) {
  const irSr = sr;  // sample_rate из measurement или default 48000
  const irFMax = Math.min(40000, irSr / 2 * 0.95);
  const irFreq = buildLogGrid(512, 5, irFMax);

  const targetWithRef = {
    ...JSON.parse(JSON.stringify(band.target)),
    reference_level_db: (band.target.reference_level_db ?? 0) + (refLevelOverride ?? autoRef),
  };

  const irTargetResp = await invoke<TargetResponse>("evaluate_target", {
    target: targetWithRef, freq: irFreq,
  });
  const irTargetPhase = await reconstructTargetPhase(
    irFreq, irTargetResp.phase,
    band.target.high_pass, band.target.low_pass,
  );

  const r = await invoke<{ time: number[]; impulse: number[]; step: number[] }>(
    "compute_impulse",
    { freq: irFreq, magnitude: irTargetResp.magnitude, phase: irTargetPhase, sampleRate: irSr },
  );
  ir.target = { time: r.time, impulse: r.impulse, step: r.step };
}
```

`buildLogGrid` — уже существует в band-evaluator.ts.
`reconstructTargetPhase` — уже существует.

`refLevelOverride ?? autoRef` — переиспользуем существующую логику
вычисления refLevel. Если код уже хранит `refLevel` локально —
использовать.

**Measurement IR и Corrected IR оставляем на measurement grid**
(текущий `freq` параметр) — там реальные замеры.

### Vitest

```typescript
describe("evaluateBandFull — target IR uses wide grid", () => {
  it("target IR computed on 5-40k grid regardless of measurement", async () => {
    // Band с measurement freq 20-22000, target Gaussian HP=632 + subsonic
    // result.ir.target.impulse длина соответствует sr × time, time покрывает 5 Hz rolloff
  });

  it("subsonic toggle changes target IR shape", async () => {
    // Band с HP Gaussian linear_phase=true
    // ON vs OFF: impulse должен отличаться
    const irOff = await evaluateBandFull({ band: bandSubsonicOff, includeIr: true });
    const irOn = await evaluateBandFull({ band: bandSubsonicOn, includeIr: true });
    const diff = maxAbsDiff(irOff.ir!.target!.impulse, irOn.ir!.target!.impulse);
    expect(diff).toBeGreaterThan(0.01);  // не идентичны
  });
});
```

### Bump

- `src-tauri/tauri.conf.json` → `0.1.140`.
- `src-tauri/src/lib.rs` startup → b140.3.3.
- `src/lib/version.ts` → b140.3.3.

## Acceptance

1. На band view вкладка IR/Step при включении защитного subsonic →
   target impulse и step **меняются** (виден subsonic rolloff в IR
   shape).
2. Linear-phase Gaussian + subsonic ON: target impulse становится
   асимметричным (linear-phase main + min-phase subsonic).
3. Min-phase Gaussian + subsonic ON: target impulse — длинный low-freq
   decay.
4. Measurement IR и Corrected IR не меняются (они на measurement grid).
5. existing 64+ vitest + 2 новых PASS.

## Что НЕ делать

- Не менять measurement IR / corrected IR pipeline (они должны
  оставаться на measurement grid).
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

- Один коммит: `fix: target IR uses wide standalone grid for subsonic visibility (b140.3.3)` + Co-Authored-By.
- Без нарратива.

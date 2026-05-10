# Промт для Code: b140.6 — FIR realized_mag/phase resample на evalRes.freq

**Тип:** bug fix. Bump до b140.6, коммит.

## Root cause (подтверждён диагностикой)

Расхождение Model vs FIR на графике Export при sr=44.1/48 кГц — из-за
двух разных log-сеток:

- **Model curve** на `evalRes.freq` = 5..40000 Hz (512 точек, всегда).
- **FIR generate + realized_mag** на `firFreq` = 5..`fMaxFir` Hz, где
  `fMaxFir = min(40000, sr·0.95/2)`. На 44.1/48 кГц fMaxFir = 20947/22800.

Обе массива по 512 точек попадают в `expData` с x-axis = evalRes.freq.
uPlot индексирует позиционно → значение realized_mag на firFreq[i]
рисуется на x=evalRes.freq[i] → видимый сдвиг до 0.8 октавы на 48k.

На sr ≥ 88200 fMaxFir = 40000 = evalRes.freq → сетки совпадают → нет
сдвига. Точно матчит пользовательский отчёт.

DIAG-REFINE подтвердил: realized_mag матчит target_mag на firFreq до
0.001 dB. Bug не в DSP, а в plot path.

## Что нужно сделать

### 1. Bump

```
src-tauri/tauri.conf.json:    version → 0.1.141 (если используется)
src-tauri/tauri.conf.json:    title → "PhaseForge — b140.6"
src-tauri/src/lib.rs:         startup log → b140.6
```

### 2. Resample realized_mag/phase на evalRes.freq в evaluateBandFull

`src/lib/band-evaluator.ts`, в блоке после получения `result` от
`generate_model_fir` (около строки 405–416):

```ts
const result = await invoke<{ ... }>("generate_model_fir", { ... });

// b140.6: resample realized_mag/phase from firFreq onto req.freq so
// callers (Export, SUM) plot FIR on the same grid as Model. Prior bug:
// at sr=44.1/48 kHz fMaxFir<40000 → firFreq compresses 0..fMaxFir into
// 512 positions while evalRes.freq covers 0..40000 → visual shift up
// to ~0.8 oct on rolloff.
const realizedMag = resampleLogGrid(firFreq, result.realized_mag, freq);
const realizedPhase = resampleLogGrid(firFreq, result.realized_phase, freq);

fir = {
  impulse: result.impulse,
  timeMs: result.time_ms,
  realizedMag,
  realizedPhase,
  taps: result.taps,
  sampleRate: result.sample_rate,
  normDb: result.norm_db,
  causality: result.causality,
};
```

`resampleLogGrid` — линейная интерполяция в log-частотном домене,
clamp на boundaries (вне диапазона firFreq → noise_floor).

Возможно уже есть utility `resampleOntoGrid` в band-evaluator.ts (был
упомянут в b140.3.x). Проверить:

```
grep -n "resampleOntoGrid\|resampleLog\|interpolate.*log" src/lib/band-evaluator.ts
```

Если есть — переиспользовать. Если нет — написать пару строк inline:

```ts
function resampleLogGrid(srcF: number[], srcV: number[], dstF: number[]): number[] {
  return dstF.map(f => {
    if (f <= srcF[0]) return srcV[0];
    if (f >= srcF[srcF.length - 1]) return srcV[srcV.length - 1];
    // binary search
    let lo = 0, hi = srcF.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (srcF[mid] <= f) lo = mid; else hi = mid;
    }
    const lf = Math.log(srcF[lo]), hf = Math.log(srcF[hi]);
    const t = (Math.log(f) - lf) / (hf - lf);
    return srcV[lo] + t * (srcV[hi] - srcV[lo]);
  });
}
```

### 3. Проверка SUM tab тоже использует resample

evaluateSum агрегирует через evaluateBandFull → realized_mag уже на
правильном grid после фикса. Но проверить grep:

```
grep -n "fir\.realizedMag\|fir.realized_mag" src/components/FrequencyPlot.tsx src/lib/band-evaluator.ts
```

Все consumers получают realized_mag на evalRes.freq grid (после фикса)
без дополнительной интерполяции.

### 4. Откат b140.5 noise floor extension (опционально)

`appendNoiseFloorTail` добавлял хвост только до Nyquist — не решал
основную проблему (разные грид-длины 512 vs 512 на разных диапазонах).
После b140.6 расширение не нужно — но оставить как защиту от boundary
clamp в Rust на linear FFT bins выше fMaxFir.

Решить: оставить (defence-in-depth) или убрать (упрощение). Спросить
если не уверен — лучше оставить пока.

### 5. Тесты

Cargo: убедиться что 179 тестов всё ещё проходят (FIR generation не
меняется).

Vitest: добавить тест на evaluateBandFull для sr=48000:
- realized_mag.length === req.freq.length
- realized_mag в passband (HP=200..LP=2000) совпадает с targetMag±0.5dB
  на тех же позициях массива.

### 6. Acceptance

Запустить dev. Открыть flat проект, HP=200 LR4 + LP=2000 LR4.
- sr=48000: Model и FIR кривые **совпадают** в passband и rolloff.
- sr=44100: то же.
- sr=88200, sr=176400: то же (без регрессии).

Скриншоты Export для подтверждения.

### 7. Коммит

```
fix: resample FIR realized_mag/phase onto eval grid (b140.6)

Cause of model/FIR mismatch on Export at sr=44.1/48 kHz: realized_mag
returned from generate_model_fir lives on firFreq (5..fMaxFir<40000)
while plot x-axis uses evalRes.freq (5..40000). Both 512 pts → uPlot
indexed positionally → ~0.8 oct visual shift.

Resample realized_mag/phase onto evalRes.freq inside evaluateBandFull.

Co-Authored-By: Claude <noreply@anthropic.com>
```

## Что НЕ делать

- Не трогать iterative_refine — он работает корректно (DIAG-REFINE
  подтвердил матч до 0.001 dB).
- Не убирать b140.5 noise floor tail без отдельного решения.
- Не менять API generate_model_fir.

## Правила

- Без нарратива.
- End-of-prompt: kill all + dev restart per CLAUDE.md.
- Один step, один промт.

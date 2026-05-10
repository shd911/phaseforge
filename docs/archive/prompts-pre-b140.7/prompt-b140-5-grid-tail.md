# Промт для Code: b140.5 — расширение grid до Nyquist с noise_floor tail

Текущий билд: 0.1.0-b140.4 → bump до 0.1.0-b140.5.

## Самооценка эффективности

| Критерий | Оценка | Комментарий |
|---|---|---|
| Размер | ✅ малый | Точечная правка в FIR ветке evaluateBandFull |
| Pre-flight audit | ✅ | Локализация чёткая |
| Гипотезы без данных | ✅ | DSP стандарт — explicit noise floor tail |

## Контекст

Регрессия на низких sr (44.1/48 кГц): apparent rolloff в FIR
сдвинут на ~1/2 окт.

Корень: FIR grid в evaluateBandFull обрезается на
`f_max = min(40000, sr/2 × 0.95)`. На 48 кГц это 22.8 кГц, на 176 кГц
40 кГц. Linear FFT grid в Rust расширяется до Nyquist; bins 22.8–24
кГц получают **constant clamp** последнего значения log grid (boundary
extrapolation `interp_single`).

Для LR4 LP=2000 Гц target_mag на 22.8 кГц ≈ −85 dB, а constant tail
−85 dB в bins 22.8–24 кГц вместо продолжения rolloff к −150 dB.
В IFFT эта "полка" даёт artifact → apparent shift rolloff.

На 176 кГц f_max=40 кГц, target_mag там ≈ −145 dB (близко к noise
floor) — clamp безвреден.

## Фикс

Дополнить FIR grid bins до Nyquist значением `noise_floor_db`. Это
заменяет constant boundary clamp на explicit silent tail.

## Что нужно сделать

### В `src/lib/band-evaluator.ts`, FIR ветка (~ строка где строится `irFreq`)

Сейчас:
```typescript
const irFMax = Math.min(40000, irSr / 2 * 0.95);
const irFreq = buildLogGrid(512, 5, irFMax);
const irTargetResp = await invoke<TargetResponse>(
  "evaluate_target", { target: targetWithRef, freq: irFreq });
const irTargetPhase = await reconstructTargetPhase(...);
```

Расширить irFreq и target до Nyquist с noise_floor tail:

```typescript
const irFMax = Math.min(40000, irSr / 2 * 0.95);
let irFreq = buildLogGrid(512, 5, irFMax);
let irTargetMag = (await invoke<TargetResponse>(
  "evaluate_target", { target: targetWithRef, freq: irFreq })).magnitude;
let irTargetPhase = await reconstructTargetPhase(...);

// b140.5: extend grid up to Nyquist with noise_floor tail
// чтобы избежать constant boundary clamp в Rust interp на linear FFT grid.
const nyquist = irSr / 2;
if (irFMax < nyquist * 0.999) {
  const noiseFloor = -150;  // соответствует firCfg.noise_floor_db
  const extraBins = 32;
  const extraFreqs: number[] = [];
  const extraMags: number[] = [];
  const extraPhases: number[] = [];
  for (let i = 1; i <= extraBins; i++) {
    const t = i / extraBins;
    // log spacing от irFMax до Nyquist*0.999
    const f = irFMax * Math.pow(nyquist * 0.999 / irFMax, t);
    extraFreqs.push(f);
    extraMags.push(noiseFloor);
    extraPhases.push(0);
  }
  irFreq = [...irFreq, ...extraFreqs];
  irTargetMag = [...irTargetMag, ...extraMags];
  irTargetPhase = [...irTargetPhase, ...extraPhases];
}

// Также для FIR pipeline на freq/mag/phase
// continue с extended arrays...
```

Так же расширить `peq_mag` если он используется (заполнить нулями).

То же при необходимости для нижней границы (если log grid f_min > некоторого порога), но обычно freq[0]=5 Hz уже близко к 0 → не критично.

### Аналогично для target IR ветки (b140.3.3)

В блоке вычисления target IR — также extend grid до Nyquist с noise_floor tail.

И для corrected IR (b140.3.4).

И для SUM IR (b140.3.5).

Везде где строится `irFreq` для compute_impulse — добавлять noise_floor tail до Nyquist.

### Vitest тест

```typescript
it("FIR grid extended to Nyquist with noise floor tail", async () => {
  // На sr=48000, irFMax=22800. После extend grid должен идти до ~24000
  // c магнитудой -150 dB.
  const band = makeBandWithLR4Bandpass();
  const result = await evaluateBandFull({ band, fir: { sampleRate: 48000, ... } });
  // FIR realized magnitude near Nyquist должна быть ≤ -120 dB (noise floor приближение)
  // Не constant boundary clamp от target.
});

it("48k vs 176k FIR rolloff matches", async () => {
  // На 48k и 176k для одного bandpass target rolloff shape должна совпадать
  // в зоне до 20 kHz (где есть данные на обоих).
});
```

### Bump

- `src-tauri/tauri.conf.json` → `0.1.140`.
- `src-tauri/src/lib.rs` startup → b140.5.
- `src/lib/version.ts` → b140.5.

## Acceptance

1. На 48 кГц LR4 200-2000 bandpass — rolloff FIR совпадает с Model
   (без 1/2 окт сдвига).
2. На 176 кГц — без изменений (там grid и так покрывал до 40 кГц).
3. existing 179+ cargo / 102+ vitest PASS.

## Что НЕ делать

- Не менять Rust `interp_single` / `interpolate_log` (используется
  широко).
- Не менять noise_floor_db default config.
- Не trogat sample_rate selection logic в UI.

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

- Один коммит: `fix: extend FIR grid to Nyquist with noise floor tail (b140.5)` + Co-Authored-By.
- Без нарратива.

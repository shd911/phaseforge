# Промт для Code: b140.7.5 — readback diag + Python compare WAV vs realized_mag

**Тип:** diagnostic. Bump до b140.7.5. Без правок DSP.

## Step 0

```
cd /Users/olegryzhikov/phaseforge
git status
git checkout -- src-tauri/src/lib.rs   # revert f32 swap → обратно f64
git status
```

## Bump до b140.7.5

- `tauri.conf.json` → b140.7.5.
- `lib.rs` → b140.7.5.
- `version.ts` → b140.7.5.

## Diag 1: readback после записи WAV

В `src-tauri/src/lib.rs::export_fir_wav` (или где вызывается
`export_wav_f64`) — после записи WAV открыть его обратно и сравнить:

```rust
// После fir::export_wav_f64(...)?;
// Read back and log first 5 samples + peak
use hound::WavReader;
let mut reader = WavReader::open(&path)
    .map_err(|e| format!("readback open: {}", e))?;
let spec = reader.spec();
let samples_back: Vec<f64> = reader.samples::<f32>()
    .filter_map(|r| r.ok().map(|v| v as f64))
    .collect();
// или для f64:
// let samples_back: Vec<f64> = reader.samples::<f64>()...
// зависит от того что fir::export_wav_f64 пишет (надо проверить spec.bits_per_sample)

let peak_back = samples_back.iter().fold(0f64, |m, &v| m.max(v.abs()));
let sum_back: f64 = samples_back.iter().sum();
tracing::info!(
    "[READBACK DIAG] file: sr={} ch={} bps={} format={:?} count={} \
     impulse[0..5]={:?} peak_abs={:e} sum={:e}",
    spec.sample_rate, spec.channels, spec.bits_per_sample, spec.sample_format,
    samples_back.len(), &samples_back[0..5.min(samples_back.len())],
    peak_back, sum_back
);
```

Если crate hound не подходит для f64 — использовать что-то совместимое.

### Что покажут эти значения

- Если `[READBACK DIAG]` impulse[0..5] === `[EXPORT WAV DIAG] entry`:
  WAV byte-perfect. Bug не в writing.
- Если различаются: bug в writing. Конкретные различия покажут где.

## Diag 2: Python compare WAV vs PhaseForge realized_mag

Создать `docs/wav-fft-compare.py`:

```python
#!/usr/bin/env python3
"""Compare FFT of exported WAV with PhaseForge realized_mag.

Usage:
    python3 wav-fft-compare.py <path_to_wav>

Loads WAV (any sample format), FFTs the impulse, prints magnitude (dB) at
key frequencies. User compares with what PhaseForge UI shows on Export tab.
"""
import sys
import numpy as np
import scipy.io.wavfile as wav

def main():
    if len(sys.argv) != 2:
        print("Usage: wav-fft-compare.py <path>")
        sys.exit(1)

    sr, data = wav.read(sys.argv[1])
    # Handle multi-channel by taking first channel
    if data.ndim > 1:
        data = data[:, 0]
    # Convert to float64 if integer
    if np.issubdtype(data.dtype, np.integer):
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float64) / max_val

    print(f"WAV: sr={sr}, samples={len(data)}, dtype={data.dtype}")
    print(f"impulse[0..5]={data[:5]}")
    print(f"peak_abs={np.max(np.abs(data)):.6e}")
    print(f"sum={np.sum(data):.6e}")

    # FFT magnitude
    n = len(data)
    spec = np.fft.rfft(data)
    mag_db = 20 * np.log10(np.maximum(np.abs(spec), 1e-300))

    # Sample at key freqs
    freqs = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
    print("\nFreq (Hz) → FFT mag (dB)")
    for f in freqs:
        if f >= sr / 2:
            print(f"  {f:>6}  > Nyquist, skip")
            continue
        idx = int(f * n / sr)
        print(f"  {f:>6}  {mag_db[idx]:+8.2f}")

if __name__ == "__main__":
    main()
```

Сделать executable: `chmod +x docs/wav-fft-compare.py`.

## Build + run

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

Сообщить user-у:
- Title bar = **b140.7.5**.
- Действия для diag:
  1. Открыть Band 3 HP=2000 sr=48k → Export WAV.
  2. В терминале:
     ```bash
     awk -v t="$(date '+%H:%M:%S')" '$0 > t' /tmp/phaseforge-dev.log | grep -E "DIAG|export"
     ```
     (или просто tail -50 после export)
  3. Прислать `[IIR PATH DIAG]`, `[EXPORT WAV DIAG]`, `[READBACK DIAG]`
     блоки.
  4. Запустить Python compare:
     ```bash
     cd /Users/olegryzhikov/phaseforge
     python3 docs/wav-fft-compare.py "/path/to/flat_48000_65536_Blackman.wav"
     ```
     Прислать вывод.
  5. То же для sr=176.4k WAV.
  6. Также прислать что PhaseForge UI показывает на Export tab для тех
     же конфигов (числа курсора при f=50, 200, 1000, 2000, 5000 Hz).

## Что покажут результаты

- **Если [READBACK] === [EXPORT WAV DIAG] entry**: WAV byte-perfect.
- **Если Python FFT(48k WAV) === PhaseForge realized_mag (48k UI)**:
  WAV содержит правильный импульс. Bug в REW reading.
- **Если Python FFT(48k WAV) показывает notch и **отличается** от
  PhaseForge realized_mag**: PhaseForge UI лжёт — реально импульс
  в IIR cascade сломан, UI plot его извращает.

## Что НЕ делать

- Не патчить DSP.
- Не убирать diag.
- Не коммитить.

## Правила

- Без нарратива.
- Один collected report после verify.

# Промт для Code: b140.7.2 — Export WAV как Float32 вместо Float64

**Тип:** bug fix. Bump до b140.7.2, коммит после verify.

## Step 0

```
cd /Users/olegryzhikov/phaseforge
git status
```

## Контекст

Diagnostic data b140.7.1 подтвердил: impulse корректно проходит
IIR → frontend → Rust → entry export_fir_wav (3 блока [*DIAG*]
идентичны). Bug **после** entry — в WAV writing.

`afinfo` показал что WAV сохраняется как **Float64** (1 ch, 8 bytes
per sample). Это нестандартный формат для audio — REW и другие
analyzers могут читать его некорректно (например интерпретировать
8 bytes как два Float32 samples → корраптинг шейпа FIR).

Standard WAV для FIR audio = **Float32** (4 bytes per sample) или
int24. Float32 имеет 7 decimal digits точности — достаточно для
impulse с peak ~0.01 (precision ~1e-9, безопасно).

## Что нужно сделать

### 1. Bump до b140.7.2

- `src-tauri/tauri.conf.json` → title `"PhaseForge — b140.7.2"`.
- `src-tauri/src/lib.rs` → startup `b140.7.2`.
- `src/lib/version.ts` → b140.7.2.

### 2. Изменить формат WAV на Float32

Найти `export_fir_wav` или функцию записи WAV:
```
grep -n "export_fir_wav\|hound\|wav::write\|WavWriter\|write_sample" src-tauri/src
```

Скорее всего используется crate `hound` или `wav`. Изменить:

```rust
// Было: Float64
let spec = hound::WavSpec {
    channels: 1,
    sample_rate: sr as u32,
    bits_per_sample: 64,
    sample_format: hound::SampleFormat::Float, // 64-bit float
};

// Должно быть: Float32
let spec = hound::WavSpec {
    channels: 1,
    sample_rate: sr as u32,
    bits_per_sample: 32,
    sample_format: hound::SampleFormat::Float, // 32-bit float
};

// При записи:
for &sample in impulse.iter() {
    writer.write_sample(sample as f32)?; // f64 → f32 cast
}
```

Точная синтаксис зависит от crate. Проверь Cargo.toml на используемый
audio crate.

### 3. Update startup diag marker

В `src-tauri/src/lib.rs` обновить:
```rust
tracing::info!("[DIAG ACTIVE] export-wav: tracing impulse data flow IIR → WAV (Float32)");
```

(Оставить diag tracing на месте на этот раз — после verify уберём
вместе с ним.)

### 4. Cargo + vitest

```
cd src-tauri && cargo test --lib 2>&1 | tail -10
cd /Users/olegryzhikov/phaseforge && npm run test 2>&1 | tail -10
```

185+ cargo / 104+ vitest должно остаться.

### 5. Запуск + UI verify

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
- В title bar **b140.7.2** — proof что новый код применён.
- Открыть проект, sr=48k, экспортировать Band 1 LP=200 как WAV.
- В терминале:
  ```
  afinfo "<path to WAV>"
  ```
  Должно показать `Float32` или `F32` в bit depth.
- Загрузить WAV в REW — phase и mag должны совпасть с моделью
  (плавный LR4 LP=200 rolloff, не notch).
- Также Band 3 HP=2000 sr=48k re-export → REW должна показать корректный
  HP вместо notch.

### 6. Если REW всё ещё broken после Float32

Возможные причины:
- WAV всё равно содержит wrong data (но diag уже подтвердил correct
  impulse в memory — bug в writing path).
- REW issue — попробовать Audacity или sox конверсию.

В таком случае: dump first 10 samples из WAV через `sox` или Python
script и сравнить с `impulse[0..5]` из diag log:

```bash
sox "<wav>" -t raw -e float -b 32 - | xxd | head -3
```

или Python:
```python
import scipy.io.wavfile as wav
rate, data = wav.read("<path>")
print(rate, data.dtype, data[:5])
```

### 7. Commit (только после REW PASS)

```
git add -A
git commit -m "$(cat <<'EOF'
fix: export FIR WAV as Float32 instead of Float64 (b140.7.2)

WAV files saved as Float64 (8 bytes per sample) are non-standard
for audio. REW and other analyzers may misinterpret Float64 as
Float32 with doubled sample count, corrupting impulse shape.

Symptom: REW showed notch-at-corner instead of HP/LP rolloff for
sr=48k WAV exports despite UI plot being correct (b140.7 IIR path
generated correct impulse, confirmed by [EXPORT WAV DIAG] match
with [IIR PATH DIAG] output).

Fix: WAV writer now uses Float32 (4 bytes per sample) — standard
for audio FIR. Precision is sufficient for impulse with peak ~0.01
(Float32 precision ~1e-9 at this magnitude).

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

## Что НЕ делать

- Не убирать diag tracing до подтверждения REW PASS.
- Не менять IIR path code.
- Не bump в b140.8 — это всё ещё инкремент внутри 140.7.x.

## Acceptance

- afinfo показывает `Float32` / `F32` в WAV.
- REW для 48k WAV показывает корректный shape (HP for HP=2000,
  LP for LP=200) совпадающий с моделью.
- 176k WAV всё ещё работает.
- 185+ cargo / 104+ vitest PASS.

## Правила

- Без нарратива.
- Один report после verify.

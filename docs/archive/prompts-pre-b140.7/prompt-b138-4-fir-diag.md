# Промт для Code: диагностика b138.4 — FIR не реагирует на subsonic

**Тип:** временный диагностический патч. Без bump версии. Без коммита.

## Контекст

В b138.4 SPL phase target корректно крутится от subsonic в обоих
режимах linear_phase (true/false). Но **FIR импульс, step и phase
response экспорта остаются такими же, как без subsonic** — будто
defense-фильтр не входит в финальные коэффициенты.

В коде `fir-export.ts:generateBandImpulse`:
1. `evaluate_target_standalone` возвращает targetMag — **должен**
   включать subsonic (та же `target::evaluate` что и в `evaluate_target`).
2. `phase_mode = MinimumPhase` когда `hasActiveSubsonicProtect(hp)` —
   значит Rust должен сам Hilbert делать от targetMag.
3. `modelPhase = new Array(freq.length).fill(0)` — phase нулевая, в
   MinimumPhase Rust её игнорирует (предположительно).

Где-то в этой цепочке subsonic теряется. Не знаем где — нужна
диагностика на каждом шаге.

## Что нужно сделать

### 1. Frontend — лог в `fir-export.ts:generateBandImpulse`

Перед invoke `generate_model_fir`:

```typescript
const subFreqIdx = freq.findIndex(f => f >= 12 && f <= 13);
const passbandIdx = freq.findIndex(f => f >= 1000);
console.log("[FIR] inputs", {
  bandName: b.name,
  hp: b.target.high_pass,
  hasSubsonic: hasActiveSubsonicProtect(b.target.high_pass),
  isLinHP: !b.target.high_pass || (b.target.high_pass.linear_phase && !hasActiveSubsonicProtect(b.target.high_pass)),
  phaseMode: (isLin(b.target.high_pass) && isLin(b.target.low_pass)) ? "LinearPhase" : "MinimumPhase",
  targetMag_at_12Hz: targetMag[subFreqIdx],
  targetMag_at_1kHz: targetMag[passbandIdx],
  freqLen: freq.length,
});
```

После invoke:

```typescript
console.log("[FIR] impulse stats", {
  impulseLen: fir.impulse.length,
  peakIdx: fir.impulse.reduce((maxI, v, i, arr) => Math.abs(v) > Math.abs(arr[maxI]) ? i : maxI, 0),
  energy: fir.impulse.reduce((s, v) => s + v * v, 0).toFixed(4),
  first10: fir.impulse.slice(0, 10).map(v => v.toFixed(4)),
});
```

### 2. Rust — лог в `generate_model_fir`

Найти команду в `src-tauri/src/fir/mod.rs` (или близко). В начале:

```rust
tracing::info!(
    "generate_model_fir: phase_mode={:?}, taps={}, target_mag len={}, target_mag[0]={:.2}dB, target_mag[~12Hz]={:.2}dB",
    config.phase_mode, config.taps,
    target_mag.len(),
    target_mag.first().copied().unwrap_or(0.0),
    /* найти индекс ближайший к 12 Hz по freq[] и логнуть target_mag там */
);
```

Перед Hilbert reconstruction (если он явный) — лог:

```rust
tracing::info!("generate_model_fir: about to compute Hilbert from target_mag");
```

После Hilbert — лог:

```rust
tracing::info!("generate_model_fir: Hilbert done, phase[~12Hz]={:.2} rad", reconstructed_phase[idx_12hz]);
```

Если в коде вместо MinimumPhase / LinearPhase ветвление — лог
"taking MinimumPhase branch" / "taking LinearPhase branch".

### 3. Поднять уровень логов

В `src-tauri/src/lib.rs` env_filter временно:

```rust
.with_env_filter("phaseforge_lib::fir=debug,info")
```

(подставить корректное имя модуля).

## Запуск

```
cd /Users/olegryzhikov/phaseforge
cargo tauri dev
```

Создать полосу, поставить Gaussian HP=632 Hz, `linear_phase=false`,
subsonic ВКЛ. Открыть DevTools console.

**Сценарий:**
1. Открой проект, импорт измерения если нужно.
2. Запусти Export FIR (кнопкой). Можно отменить save dialog — нам
   нужны только логи.

## Что прислать обратно

Из DevTools Console — все строки `[FIR]`.

Из терминала с `cargo tauri dev` — все строки `generate_model_fir:`.

## Что НЕ делать

- Не предлагать гипотезы.
- Не менять логику (только логи).
- Не коммитить.
- Не делать bump.

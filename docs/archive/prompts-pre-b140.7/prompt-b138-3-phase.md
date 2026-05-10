# Промт для Code: b138.3 — phase реконструкция учитывает subsonic

**Тип:** функциональный фикс. Bump до 0.1.0-b138.3.

## Контекст

В b138.2 чекбокс работает, magnitude target правильно отражает subsonic
spike. Но phase target в зоне ниже cutoff/8 **не крутится** — выглядит
как для чистого Gaussian.

Причина: на фронтенде `addGaussianMinPhase` (band-evaluation.ts:21)
считает Hilbert от чистой Gaussian magnitude (`gaussianFilterMagDb`).
Subsonic к этой magnitude не применяется, поэтому Hilbert не учитывает
защитный фильтр и phase в зоне subsonic = 0.

Активно только когда `linear_phase=false` для Gaussian HP. В режиме
`linear_phase=true` phase везде = 0 by design — это корректное
поведение, не трогаем.

## Что нужно сделать

### Точечная правка `src/lib/band-evaluation.ts`

Функция `addGaussianMinPhase`. Сейчас:

```typescript
if (isGaussianMinPhase(hp)) {
  const hpMag = gaussianFilterMagDb(freq, hp!, false);
  const hpPh = await invoke<number[]>("compute_minimum_phase", { freq, magnitude: hpMag });
  result = result.map((v, i) => v + hpPh[i]);
}
```

Заменить на:

```typescript
if (isGaussianMinPhase(hp)) {
  let hpMag = gaussianFilterMagDb(freq, hp!, false);
  // Apply subsonic protect to magnitude before Hilbert reconstruction,
  // so the resulting min-phase reflects total (Gaussian × Subsonic) target.
  if (hp!.subsonic_protect === true && hp!.freq_hz > 40) {
    const fSub = hp!.freq_hz / 8;
    hpMag = hpMag.map((db, i) => {
      const f = freq[i];
      if (f <= 0) return db;
      const ratio = Math.pow(fSub / f, 16);  // Butterworth 8th order
      const subsonicLin = Math.sqrt(1 / (1 + ratio));
      const subsonicDb = subsonicLin > 1e-20 ? 20 * Math.log10(subsonicLin) : -400;
      return db + subsonicDb;
    });
  }
  const hpPh = await invoke<number[]>("compute_minimum_phase", { freq, magnitude: hpMag });
  result = result.map((v, i) => v + hpPh[i]);
}
```

LP-ветка не меняется — subsonic применяется только к HP.

Формула subsonic Butterworth 8th order — точно та же что в Rust
`apply_filter`: `(1 / (1 + (f_sub / f)^16))^0.5`. Это обеспечивает
консистентность magnitude в Hilbert input и в evaluate_target output.

### Проверка FIR export

После фикса проверить (не править если нет проблемы) — где живёт
построение FIR target curve. Если FIR использует `evaluate_target`
напрямую через invoke и берёт phase из его response — то Rust сейчас
возвращает phase=0 для Gaussian (без subsonic). Это значит FIR
со subsonic будет линейно-фазовым по subsonic части (только magnitude
применена), а Gaussian min-phase реконструкция в FIR pipeline
делается другим путём.

Если FIR pipeline вызывает `addGaussianMinPhase` (или аналог) — фикс
выше автоматически работает для FIR.

Если FIR строится отдельным путём, не вызывая `addGaussianMinPhase` —
там нужен такой же фикс. Найти точку построения FIR target и
проверить.

**Не делать слепо** — сначала глянуть код FIR-pipeline, понять,
дублируется ли логика или используется общая функция. Если
дублируется — применить аналогичную правку. Если используется общая —
пропустить.

### Bump версии

- `src-tauri/tauri.conf.json` — version + productName/title до b138.3.
- `src-tauri/src/lib.rs` — startup-лог.
- После билда — skill `build-version`.

## Acceptance

1. Gaussian HP=632 Гц с `linear_phase=false` (min-phase), subsonic ВКЛ:
   - Phase target в зоне 50…200 Гц крутится примерно как Gaussian min-phase.
   - Phase target в зоне 5…40 Гц **дополнительно** крутится от subsonic
     Butterworth (около 90° на каждые 12 дБ затухания).
2. Снять subsonic → phase в зоне 5…40 Гц возвращается к 0 (или к
   слабому Gaussian tail).
3. `linear_phase=true` Gaussian HP — phase везде 0 (by design),
   subsonic не влияет на phase. Это корректное поведение.
4. FIR export со subsonic ВКЛ:
   - Если FIR использует `addGaussianMinPhase` → работает автоматически.
   - Если FIR имеет свою логику построения target → проверить и
     применить аналогичный subsonic mod (если нужно).

## Регрессионная проверка

- b131-b138.2 целы.
- Magnitude target с subsonic — без изменений.
- Чекбокс ON/OFF переключается корректно (b138.2 не сломан).
- LP филтр без subsonic_protect — phase реконструкция работает как
  раньше.
- linear_phase=true — phase target = 0, как было.

## Что НЕ трогать

- Rust `apply_filter` — magnitude формула subsonic уже корректна.
- `gaussianFilterMagDb` — оставить чистым Gaussian, subsonic
  накладывается локально в `addGaussianMinPhase`.
- `isGaussianMinPhase` логика — без изменений.

## Тестировать на `.dmg`

После сборки запустить
`src-tauri/target/release/bundle/dmg/PhaseForge_0.1.138-3_aarch64.dmg`
и пройти acceptance pp. 1-4.

## Правила (CLAUDE.md)

- Один коммит: `fix: subsonic protect contributes to min-phase
  reconstruction (b138.3)` + Co-Authored-By.
- 7-vector review.
- Без нарратива прогресса.
- `cargo tauri build` для финальной сборки.

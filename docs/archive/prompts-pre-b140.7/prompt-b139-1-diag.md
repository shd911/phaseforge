# Промт для Code: диагностика — linear-phase Gaussian + subsonic не крутит phase

**Тип:** временный диагностический патч. Без bump версии. Без коммита.

## Контекст

В b139.0 regression-checklist обнаружил что пункт 3 (Gaussian
HP=632 + linear_phase=true + subsonic ON: phase должна крутиться в
5–40 Гц) НЕ работает. b139.0 не менял production код, значит дефект
существовал в b138.4 — мы пропустили проверку этой комбинации.

Логика в `band-evaluation.ts` визуально правильная:
- `evaluateBand` ловит `needsLinearPhaseSubsonic` и зовёт
  `addGaussianMinPhase`.
- `addGaussianMinPhase` имеет ветку `else if (hasActiveSubsonicProtect(hp) && hp!.linear_phase === true)`,
  которая делает Hilbert от subsonic-only magnitude.

Но фактически phase не крутится. Гипотезы:
1. **FrequencyPlot не использует `evaluateBand`** для phase plot —
   имеет inline invoke цепь.
2. Hilbert от subsonic-only magnitude **возвращает ≈0** (magnitude
   плоская в верхней части grid → Hilbert даёт мало phase).
3. Phase где-то **перезаписывается** перед рендером.
4. **Условие** `needsLinearPhaseSubsonic` не срабатывает (поле
   subsonic_protect не доходит).

Нужна точечная диагностика чтобы определить какая.

## Что нужно сделать

### 1. Frontend: лог в `band-evaluation.ts:evaluateBand`

Перед условием `if (targetPhase && freq && ...)` (строка ~126):

```typescript
console.log("[evalBand] state", {
  bandName: band.name,
  hp: band.target.high_pass,
  isGaussMinPhaseHP: isGaussianMinPhase(band.target.high_pass),
  isGaussMinPhaseLP: isGaussianMinPhase(band.target.low_pass),
  needsLinearPhaseSubsonic,
  hasSubsonic: hasActiveSubsonicProtect(band.target.high_pass),
});
```

Внутри `addGaussianMinPhase`, после ветвления:

```typescript
if (isGaussianMinPhase(hp)) {
  console.log("[addGMP] taking branch: Gaussian min-phase HP");
  // ... existing logic
} else if (hasActiveSubsonicProtect(hp) && hp!.linear_phase === true) {
  console.log("[addGMP] taking branch: linear-phase Gaussian + subsonic only");
  const subDb = subsonicMagDb(freq, hp!.freq_hz / 8);
  console.log("[addGMP] subDb sample", {
    fSub: hp!.freq_hz / 8,
    at_5Hz: subDb[freq.findIndex(f => f >= 5)],
    at_10Hz: subDb[freq.findIndex(f => f >= 10)],
    at_50Hz: subDb[freq.findIndex(f => f >= 50)],
    at_500Hz: subDb[freq.findIndex(f => f >= 500)],
  });
  const subPh = await invoke<number[]>("compute_minimum_phase", { freq, magnitude: subDb });
  console.log("[addGMP] subPh from Hilbert", {
    at_5Hz: subPh[freq.findIndex(f => f >= 5)],
    at_10Hz: subPh[freq.findIndex(f => f >= 10)],
    at_50Hz: subPh[freq.findIndex(f => f >= 50)],
    at_500Hz: subPh[freq.findIndex(f => f >= 500)],
  });
  result = result.map((v, i) => v + subPh[i]);
}
```

### 2. Найти точку рендера phase в FrequencyPlot

Запустить:

```
grep -n "evaluateBand\|targetPhase\|target.*[Pp]hase" src/components/FrequencyPlot.tsx | head -30
```

Перечислить все места где phase plot читает данные. Если `FrequencyPlot.tsx`
**не вызывает** `evaluateBand` (имеет свой pipeline через inline invoke
`evaluate_target`) — это и есть корень проблемы (Гипотеза 1).

В каждой такой inline точке добавить лог:

```typescript
console.log("[FreqPlot:phasePoint]", {
  callsite: "<какая функция или строка>",
  hp: <какой HP в этом контексте>,
  // если phase уже посчитана — sample на 5/10/50/500 Hz
});
```

### 3. Запуск

```
cd /Users/olegryzhikov/phaseforge
cargo tauri dev
```

В UI:
1. Создать или открыть полосу с измерением.
2. Поставить HP: Gaussian, freq=632 Hz, **linear_phase=TRUE**, subsonic ON.
3. Перейти на вкладку SPL/phase plot, дождаться рендера.

Открыть DevTools Console.

## Что прислать обратно

Все строки `[evalBand]`, `[addGMP]`, `[FreqPlot:phasePoint]` —
скопировать в чат.

Также скопировать **результат grep** из шага 2 — список найденных
строк в FrequencyPlot.tsx с phase / evaluateBand.

## Что НЕ делать

- Не предлагать гипотезы о причине.
- Не менять логику.
- Не коммитить — это временный патч.
- Не делать bump.

После diagnostic — точечный фикс на основе полученных данных.

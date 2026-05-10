# Промт для Code: b140.3.8 — fix divergence regression + cleanup

Текущий билд: 0.1.0-b140.3.7 → bump до 0.1.0-b140.3.8.

## Самооценка эффективности

| Критерий | Оценка | Комментарий |
|---|---|---|
| Размер | ✅ малый | Восстановление существовавшего фикса в Composite ветке |
| Pre-flight audit | ✅ | Аудит локализовал регрессию |
| Гипотезы без данных | ✅ | Известное поведение, фикс уже был в b139.3.4 |

## Контекст

Глобальный аудит показал блокер для релиза b140.4: cargo тест
`iterative_refine_converges_with_min_phase_subsonic` падает с
DIVERGENCE (max_err 11.222 dB на iter 2 vs 0.418 на iter 1).

Тест был зафикшен в b139.3.4 — `phase_rad` пересчитывался на каждой
итерации для MinimumPhase mode (исключая double-recompute). После
добавления Composite mode (b139.4a) и phase-split (b140.1) фикс
отменён или Composite обходит recompute.

## Что нужно сделать

### 1. Восстановить per-iteration phase recompute для Composite

В `src-tauri/src/fir/helpers.rs` функция `iterative_refine`:

Текущая логика (после b139.3.4) делала recompute для
`PhaseMode::MinimumPhase`. Нужно убедиться что аналогичный recompute
работает для `PhaseMode::Composite` — он внутри использует
`composite_phase_inner` (объединение main + peq + subsonic phase
sources).

Алгоритм:
- В loop iteration → пересчитать phase для refined_db через
  `composite_phase_inner` с теми же параметрами что и в initial pass.
- Не использовать сохранённую phase_rad от прошлой итерации.

```rust
let recompute_min_phase = matches!(config.phase_mode, PhaseMode::MinimumPhase);
let recompute_composite = matches!(config.phase_mode, PhaseMode::Composite);

for iter in 0..config.iterations {
    // ... refined_db computation ...

    let iter_phase = if recompute_min_phase {
        minimum_phase_from_magnitude(&refined_db, n_fft)
    } else if recompute_composite {
        let subsonic = subsonic_mag.as_ref()
            .expect("subsonic_mag_fixed must exist in Composite mode");
        composite_phase_inner(
            &refined_db, subsonic, peq_mag,
            n_fft, config.linear_phase_main, config.noise_floor_db,
        )
    } else {
        phase_rad.to_vec()  // LinearPhase / Hybrid: фиксированная phase
    };

    // assemble_complex_spectrum(&refined_db, &iter_phase) ...
}
```

Точная сигнатура `composite_phase_inner` — проверить в коде.
В b139.4a / b140.1 она принимает 3 источника (main_mag через subtraction,
subsonic_mag отдельно, peq_mag отдельно). Применить ту же сигнатуру.

### 2. Убедиться что cargo тест PASS

После фикса:
```
cargo test fir::tests::iterative_refine_converges_with_min_phase_subsonic
```

Должен PASS — convergence iter 1 → ≤ iter 1 → ≤ iter 2.

### 3. Обновить CLAUDE.md Last reviewed

Через `date '+%Y-%m-%d'` получить текущую дату, обновить:

```
> **Last reviewed:** YYYY-MM-DD (после b140.3.x clean SUM rebuild + b140.3.8 divergence fix).
```

### 4. Убрать устаревшую ссылку в комментарии

`band-evaluator.ts:118` — комментарий `"Mirrors band-evaluation.ts:addGaussianMinPhase"` —
функция `addGaussianMinPhase` удалена в b139.4c. Обновить комментарий
или удалить ссылку:

```typescript
// b139.4c: единая phase reconstruction (Gaussian min-phase + subsonic).
// Заменила legacy addGaussianMinPhase из band-evaluation.ts.
```

### 5. Bump

- `src-tauri/tauri.conf.json` → `0.1.140`.
- `src-tauri/src/lib.rs` startup → b140.3.8.
- `src/lib/version.ts` → b140.3.8.

## Acceptance

1. `cargo test fir::tests::iterative_refine_converges_with_min_phase_subsonic`
   PASS.
2. Все 178+ cargo тестов PASS.
3. Vitest 101+ PASS.
4. CLAUDE.md `Last reviewed` обновлено через `date`.
5. Комментарий в band-evaluator.ts:118 обновлён.

## Что НЕ делать

- Не менять Composite mode logic — только phase recompute path.
- Не трогать LinearPhase / Hybrid — там phase фиксирована, recompute
  не нужен.
- Не делать другие fixes — релиз готовится сразу после.

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

- Один коммит: `fix: per-iteration phase recompute for Composite mode (b140.3.8)` + Co-Authored-By.
- Без нарратива.
- После passing — готовность к релизу b140.4.

# PhaseForge b140.4 — Composite phase mode + unified SUM pipeline

Большой пакет архитектурных улучшений с предыдущего релиза b139.5.3.

## Composite phase mode для FIR экспорта (b140.1)

При включённом защитном subsonic фильтре + линейно-фазовом основном
Gaussian — экспорт работает в **составном режиме**: основной фильтр
сохраняет линейную фазу в полосе пропускания, защитный subsonic вносит
минимально-фазовую rotation только в инфразвуке. Это убрало проблему
из b138.x когда включение защитного делало весь FIR минимально-фазовым.

## Unified SUM pipeline (b140.3.x)

После каскада b140.2.x (11 итераций без паритета через копирование
legacy) — **полная перестройка с чистого листа** через `evaluateSum`.

- **Σ Target**: coherent sum target curves каждой полосы с polarity
  и alignment_delay.
- **Σ Corrected**: per-band normalize к собственному target в его
  passband, width-aware excess limiter (узкие резонансы сохраняются,
  широкие превышения > 0.1 dB лимитируются), global shift вниз при
  обнаружении wide bumps.
- **Σ Measurement**: extension через target shape для magnitude и
  Hilbert reconstruction для phase — корректное продление за пределы
  native freq диапазона полос.
- **SUM IR/Step**: через единый источник `evaluateSum.ir` на широкой
  freq grid (5 Гц – 40 кГц) — subsonic виден в impulse.

Переключатель **Legacy/New** на вкладке общей суммы — старый pipeline
сохранён для совместимости.

## Band IR/Step на широкой grid (b140.3.3, b140.3.4)

Target IR и Corrected IR для отдельной полосы строятся на standalone
grid 5–40 кГц (не на узком grid замера). Subsonic фильтр и другие
низкочастотные эффекты теперь видны в импульсе и step response.

## Extension в evaluateBandFull (b140.3.2)

Логика расширения measurement за пределы native freq диапазона
(через target shape + Hilbert phase) перенесена на per-band уровень.
На band view синтетическая часть отображается **бледным цветом** —
пользователь видит где реальный замер, где математическая модель.

## Полировка (b140.3.x)

- Crossover handles + drag-drop в New SUM (b140.3.1.6).
- Корректная связь impulse↔step после resample (cumsum derive вместо
  независимой интерполяции, b140.3.6).
- Стабильные labels для legend (без variable suffix), auto-align не
  сбрасывает visibility (b140.3.7).

## Под капотом

- Test harness fix для cargo iterative_refine convergence (thread-local
  vs Mutex pollution, b140.3.8).
- 179 cargo + 102 vitest тестов проходят.
- Глобальный аудит pipeline перед релизом (только legacy/UX inline
  invoke остались, все мигрировано).

## Известные ограничения

- Default SUM режим — Legacy. New работает но требует больше
  тестирования на разных проектах.
- TODO: sr=None при импорте .txt (с b135) — отдельная задача.

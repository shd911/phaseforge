# Промт для Code: диагностика b135 — анализ не показывается на живых замерах

**Тип задачи:** диагностический логгинг, без bump версии, без полного билда.
Только `cargo tauri dev`. Коммитить **не нужно**, изменения временные.

## Контекст

После импорта живых замеров пользователь не видит модалку анализа.
Гипотеза: детекторы возвращают пустой `findings = []` (пороги слишком
строги для реальных room-данных), и модалка по коду не открывается
вообще (нет реализации сообщения «проблем не обнаружено»).

## Что нужно сделать

### 1. В `src/lib/measurement-actions.ts`

Функция `runAnalysis`, между строкой 52 и 53. После

```typescript
const result = await invoke<AnalysisResult>("analyze_measurement", { measurement: m });
```

вставить:

```typescript
console.log("[Analysis] result for", m.name, {
  n_findings: result.findings.length,
  ids: result.findings.map(f => f.id),
  full: result,
});
```

### 2. В `src-tauri/src/analysis/mod.rs`

Внутри `analyze_measurement`, перед `Ok(AnalysisResult { ... })`,
добавить лог:

```rust
tracing::info!(
    "analyze_measurement: name={}, len={}, sr={:?}, findings={}",
    measurement.name,
    measurement.freq.len(),
    measurement.sample_rate,
    findings.len(),
);
```

И для каждого детектора, который возвращает `None`, добавить debug-лог
с причиной (одна строка):

- `detect_noise_floor_low`: при возврате `None` — `tracing::debug!("nf_low: rejected reason={...}, last_flat={:?}, drop={:?}")`.
- `detect_lf_rolloff`: при возврате `None` — `tracing::debug!("lf_rolloff: slope={:.1}, no_resonance={}, f_hi={:?}")`.
- `detect_hf_cliff`: при возврате `None` после цикла — `tracing::debug!("hf_cliff: no candidate found in range")`.

Использовать существующий `tracing` инфраструктуру (она уже есть в lib.rs).
Уровень `info` для итогов, `debug` для причин отказов — чтобы не засорять
вывод.

### 3. Убедиться что debug логи видны

В `src-tauri/src/lib.rs` найти инициализацию `tracing_subscriber`. Если
текущий уровень `info` — временно поднять до `debug` для модуля
`phaseforge::analysis`. Минимально:

```rust
.with_env_filter("phaseforge::analysis=debug,info")
```

(подставить корректное имя модуля по актуальному lib.rs).

## Что прислать обратно

После запуска `cargo tauri dev` импортировать живой замер. Скопировать:

1. Из терминала где запущен dev: строки `analyze_measurement: name=...`
   и debug-логи каждого детектора.
2. Из DevTools Console (Cmd+Opt+I в окне приложения): строку
   `[Analysis] result for ... { n_findings: ... }`.

Прислать оба блока сюда — на основе данных решим, что фиксить:
пороги детекторов, реализацию «no findings» модалки, или оба.

## Что НЕ делать

- Не менять пороги детекторов на этом шаге.
- Не реализовывать модалку «проблем не обнаружено».
- Не делать bump версии.
- Не коммитить — это временный диагностический патч.

После сбора данных — диагностические логи откатить (`git checkout`),
получить точечный промт b135.1 на основе того что увидим.

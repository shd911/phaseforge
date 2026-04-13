# Audit Pre-Release b115
**Date:** 2026-04-13  
**Scope:** `git diff v0.1.91..HEAD` — 68 файлов, ~15k строк изменений  
**Agents:** 7 параллельных (bugs, logic, simplification, security, performance, data-compat, global adversarial)

---

## Сводная таблица (дедуплицированная, по severity)

| # | Severity | File:Line | Issue | Agent(s) |
|---|----------|-----------|-------|---------|
| 1 | CRITICAL | `auto-align.ts:88–90`, `FrequencyPlot.tsx:3503` | PEQ фаза возвращается wrapped `[-180°,+180°]` и прибавляется к unwrapped фазе измерения → корруптированная cost function в авто-align | Global, Bugs |
| 2 | HIGH | `project.rs:74`, `project-io.ts:257` | `alignment_delay` отсутствует в Rust `BandData` — поле silently dropped serde при каждом сохранении → все задержки теряются на reload | Data compat |
| 3 | HIGH | `FrequencyPlot.tsx:3310–3394` | 8+ `await invoke()` в `renderSumMode` без `gen !== renderGen` guard → concurrent renders пишут в shared chart state | Global |
| 4 | HIGH | `auto-align.ts:125–131` | HP/LP источники XO пары перепутаны: `sorted[i].HP` + `sorted[i+1].LP` — правильно должно быть `sorted[i].LP` + `sorted[i+1].HP`; при отсутствии фильтра у сабвуфера вся пара **silently SKIP** | Bugs |
| 5 | HIGH | `FrequencyPlot.tsx:3602, 3692` | Guard `perBandCorrPhase[ci]!.length === nPts` использует pre-resample `nPts` вместо `freq.length` → при resampling NaN в когерентной сумме | Bugs, Logic |
| 6 | HIGH | `auto-align.ts:88–100` | Нет length assertion после `invoke("compute_peq_complex")` и `invoke("compute_cross_section")` → массивы меньшей длины дают silent `undefined → 0` | Bugs |
| 7 | HIGH | `tauri.conf.json:25` | CSP отключён (`null`) — нет XSS mitigation. Crafted измерение с именем-скриптом выполнится без ограничений | Security |
| 8 | HIGH | `project.rs:140,151`, `lib.rs:481` | Arbitrary path write/read — нет directory scope enforcement; `save_project` / `export_fir_wav` принимают любой путь с фронтенда | Security |
| 9 | HIGH | `auto-align.ts:127` | Центр XO вычисляется arithmetic mean `(lp+hp)/2` вместо geometric `√(lp×hp)` — на log scale смещение оценки XO региона | Logic, Simplification |
| 10 | HIGH | `auto-align.ts:229–246` | Gradient descent: `newDelay` не clamp'ится к `[-scanRange, scanRange]` → теоретически может выйти за физически осмысленный диапазон | Logic |
| 11 | HIGH | `auto-align.ts:23–35` | `interpOnGrid` с `srcFreq.length === 0`: `srcFreq[-1] = undefined` → NaN без ошибки; с `length === 1` возвращает NaN на всём диапазоне кроме одной точки | Logic |
| 12 | HIGH | `FrequencyPlot.tsx:3326–3344` | Sequential `await invoke("interpolate_log")` per band в for-loop вместо `Promise.all` → N×IPC latency | Performance |
| 13 | HIGH | `FrequencyPlot.tsx:3371–3393` | 2× `evaluate_target` per band (reference + norm) sequential → можно заменить scalar offset, убрать 50% вызовов | Performance |
| 14 | HIGH | `FrequencyPlot.tsx:3452–3473` | Sequential PEQ + XO invokes per band в for-loop → можно параллелизировать `Promise.all([peq, xo])` per band | Performance |
| 15 | MEDIUM | `FrequencyPlot.tsx:3637–3648` | Per-frequency `avgDelay` для компенсации фазы даёт скачки на XO частотах (где доминирование полосы меняется) | Logic |
| 16 | MEDIUM | `auto-align.ts:49–53` | Early return при `validBands < 2` сбрасывает только validBands → у остальных полос остаются старые задержки | Global |
| 17 | MEDIUM | `project.rs:174` | `project_name` используется напрямую в `PathBuf::join` без sanitization → `../../../.ssh/authorized_keys` escape | Security |
| 18 | MEDIUM | `project-io.ts:365,391` | Path traversal guard на фронтенде; `mergeSource.nfPath/ffPath` идут в invoke без проверки | Security |
| 19 | MEDIUM | `project-io.ts:437–450` | `m.phase` может быть `null` при вызове `invoke("remove_measurement_delay")`, где Rust ожидает `Vec<f64>` | Data compat |
| 20 | MEDIUM | `FrequencyPlot.tsx:3329` | Strict `===` float comparison для grid fast-path → ненужные IPC re-interpolations | Bugs |
| 21 | MEDIUM | `auto-align.ts:232–238` | Redundant `cost(currentDelay)` re-eval в каждой итерации GD; constant amplitudes пересчитываются внутри cost() | Performance |
| 22 | MEDIUM | `FrequencyPlot.tsx:3636, 3724` | O(n_freq × n_bands) `Math.pow(10, v/10)` в phase display loops; можно reuse уже вычисленные amplitude arrays | Performance |
| 23 | LOW | `auto-align.ts:250–254` | `bandCenterFreq` — dead code, ни разу не вызывается; sort tiebreaker нестабилен при нескольких полосах без HP | Simplification, Global |
| 24 | LOW | `project-io.ts:809` | `load_project` result без runtime schema validation; `exclusion_zones` как `Vec<serde_json::Value>` в Rust | Data compat |
| 25 | LOW | `plot-helpers.ts:14,32` | `gaussianLpMagDb` / `gaussianHpMagDb` экспортируются без внешних caller'ов | Simplification |
| 26 | LOW | `band-evaluation.ts:48` | Параметр `_showPhase` никогда не читается в теле функции | Simplification |

---

## Детали по CRITICAL/HIGH

### #1 CRITICAL — PEQ фаза wrapped + unwrapped measurement phase

**Root cause:** `compute_peq_complex` (Rust `biquad.rs:48`) возвращает фазу wrapped в `[-180°, 180°]`. В `auto-align.ts:88–90`:
```typescript
ph = ph.map((v, i) => v + (pp[i] ?? 0));  // unwrapped + wrapped = corrupted
```
`ph` из измерения REW — unwrapped (может быть ±несколько тысяч градусов). `pp` — wrapped per-biquad. После сложения фаза в XO-регионе содержит ±360°·N ошибку → cost function видит неверную фазу → неверные задержки.

То же самое в `FrequencyPlot.tsx:3503`:
```typescript
let corrPhase = rm.phase.map((v, j) => v + (peqPhase ? peqPhase[j] : 0) + (xsPhase ? xsPhase[j] : 0));
```

**Fix:** После прибавления PEQ phase — unwrap результат:
```typescript
// После: ph = ph.map((v, i) => v + (pp[i] ?? 0));
ph = unwrapDegrees(ph);  // добавить функцию unwrap
```
Или изменить Rust, чтобы возвращал accumulated фазу без финального wrap.

---

### #2 HIGH — `alignment_delay` не сохраняется

**Root cause:** Поля нет в `BandData` struct в `project.rs`. Serde игнорирует неизвестные поля при десериализации → при `save_project` → Rust re-serializes → поле отсутствует.

**Fix (одна строка в Rust):**
```rust
// project.rs, в struct BandData:
#[serde(default)]
pub alignment_delay: Option<f64>,
```

---

### #3 HIGH — Missing gen guards в renderSumMode

**Fix:** Добавить `if (gen !== renderGen) return;` после:
- строки 3314 (после `interpolate_log` для newFreq)
- строки 3386 и 3393 (после каждого `evaluate_target` в for-loop)

---

### #4 HIGH — XO пары HP/LP перепутаны

Текущий код (неверно):
```typescript
const hpFreq = sorted[i].target?.high_pass?.freq_hz;    // HP верхней полосы
const lpFreq = sorted[i + 1].target?.low_pass?.freq_hz; // LP нижней полосы
```
Для XO между полосами нужно: LP верхней с HP нижней:
```typescript
const lpFreq = sorted[i].target?.low_pass?.freq_hz;     // LP верхней полосы
const hpFreq = sorted[i + 1].target?.high_pass?.freq_hz; // HP нижней полосы
```
При текущем коде сабвуфер (нет HP) → `hpFreq = undefined` → пара silently SKIP.

---

### #9 HIGH — Geometric mean для XO центра

```typescript
// Было:
const xoFreq = (lpFreq + hpFreq) / 2;
// Правильно:
const xoFreq = Math.sqrt(lpFreq * hpFreq);
```

---

## Приоритет фиксов

**Немедленно (блокируют корректность):**
1. `#2` — alignment_delay не сохраняется (1 строка Rust)
2. `#4` — HP/LP перепутаны в XO detection (swap двух переменных)
3. `#5` — `nPts` → `freq.length` в guards когерентной суммы

**Важно (снижают качество результата):**
4. `#1` — PEQ phase wrapping corruption
5. `#16` — early return сбрасывает не все полосы
6. `#10` — gradient descent clamp
7. `#11` — interpOnGrid guard для пустого массива

**Безопасность (до публичного релиза):**
8. `#7` — CSP
9. `#8` — path traversal в file operations
10. `#17` — project_name sanitization

**Performance (можно после релиза):**
11. `#12`, `#13`, `#14` — Promise.all в renderSumMode

**Tech debt (backlog):**
12. `#23` — удалить `bandCenterFreq`
13. `#25`, `#26` — cleanup exports/params

---

## Что НЕ исправлять сейчас

- `#15` (avgDelay скачки) — это display-only, не crasher; сложный трейдофф
- `#24` (exclusion_zones типизация) — архитектурное решение требует обсуждения
- `#20` (float strict equality) — minor latency, не correctness

---

## Resolution (b116)
- #2 serde default for alignment_delay — APPLIED
- #4 HP/LP swap — REVERTED in b116 (sort is HF→LF, original code correct)
- #5 freq.length guards in FrequencyPlot — APPLIED
- #9 geometric mean for XO center — APPLIED
- #1 unwrapDegrees after PEQ/XO phase addition — APPLIED
- #11 interpOnGrid guard on degenerate input — APPLIED
- Tested on 4-way: delays 0.00/1.16/0.94/0.92 ms, save/reopen OK
- Remaining: #7, #8, #17 (security) — separate TZ after b116
- Performance #12-14 — backlog

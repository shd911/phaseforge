# Regression Checklist (manual UI workflow)

DSP-логика покрыта автотестами (cargo + vitest). Этот checklist — только UI
workflow которые требуют запуска приложения и не автоматизированы без
e2e-фреймворка.

## Перед запуском

```
cargo test --manifest-path src-tauri/Cargo.toml
npm test
```

Если что-то падает — DSP regression. Останавливаемся и чиним до сборки.

## Manual smoke на `.dmg`

1. **Версия** — заголовок окна совпадает с релизом (`PhaseForge v0.1.0-bXXX.X`).
2. **Project lifecycle** — New Project → две полосы → Cmd+S → Cmd+O возвращает
   то же состояние; Cmd+Q при unsaved → диалог Save / Don't Save / Cancel
   (b131); File → Versions → Save Version → Restore (b133).
3. **Импорт измерения** (любой `.txt`) → сразу появляется analysis dialog
   (b135). Закрытие через ✓/Закрыть.
4. **Optimize PEQ + Stale flag** — после Optimize изменить HP freq → PEQ-вкладка
   получает оранжевый банер «PEQ устарел» (b136). Cmd+Z откатывает оба
   изменения (b132).
5. **FIR Export** — выбрать активную полосу → Export WAV → файл сохраняется,
   импульс непустой (импортируй обратно через Import — должно совпадать с
   target в пределах нормализации).

## DSP-логика, покрытая автотестами

Не дублировать вручную — fail в этих ветках = провал автотеста.

- evaluate_target для всех 6 канонических HP конфигураций
  (`evaluate_target_b139_golden_*` cargo)
- FIR identity для flat input (LinearPhase + MinimumPhase) —
  `fir_identity_for_flat_input_no_filters`,
  `fir_identity_with_min_phase_mode`
- FIR magnitude в passband для Gaussian linear + subsonic —
  `fir_linear_gaussian_with_subsonic_keeps_passband_intact`
- Lock-in golden hashes для b138.4 поведения —
  `generate_fir_b139_3_*`
- Phase reconstruction для всех 4 Gaussian × subsonic комбинаций —
  vitest `band-evaluator.test.ts` snapshot + equivalence
- evaluateBandFull → identity FIR для flat band — vitest
  `band-evaluator-fir.test.ts`
- Q envelope, peq stale, subsonic-protect math, q_warn_at —
  cargo `q_envelope::tests::*`, frontend `peq-quality.ts`

Если автотесты падают — DSP regression. Если manual UI checklist падает —
UI/state regression.

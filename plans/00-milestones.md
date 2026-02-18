# Milestone Roadmap

## M0 — Scaffold & Infrastructure (Week 1)

**Goal:** Tauri 2 + SolidJS project boots, IPC roundtrip works, CI builds macOS `.dmg`.

- [ ] `cargo tauri init` с SolidJS template
- [ ] Базовая Rust-структура: модули `io`, `dsp`, `peq`, `fir`, `phase`, `target`, `export`
- [ ] Tauri IPC: фронт вызывает Rust-функцию, получает ответ
- [ ] GitHub Actions: `cargo tauri build --target aarch64-apple-darwin`
- [ ] Error handling strategy: `thiserror` в Rust, typed errors через IPC
- [ ] Logging: `tracing` в Rust, console bridge в SolidJS

**Deliverable:** Пустое окно с "Hello PhaseForge", версия в title bar, .dmg собирается.

---

## M1 — Measurement I/O + Plot (Weeks 2–3)

**Goal:** Импорт REW `.txt`, отрисовка АЧХ и фазы на интерактивном графике.

- [ ] Парсинг REW `.txt` (freq / SPL / phase)
- [ ] Парсинг `.frd` (frequency response data)
- [ ] Внутренняя структура `Measurement { freq: Vec<f64>, mag: Vec<f64>, phase: Vec<f64> }`
- [ ] Интерполяция на логарифмическую сетку (для отображения) и линейную (для FFT)
- [ ] uPlot компонент: dual-axis (dB + degrees), log-X, zoom/pan
- [ ] File dialog через Tauri API
- [ ] Multi-measurement overlay (разные цвета)

**Deliverable:** Открываешь REW файл → видишь АЧХ + фазу на графике, можно зумить.

→ Details: [01-measurement-io.md](01-measurement-io.md)

---

## M2 — Target Curve Engine (Week 4)

**Goal:** Дизайнер целевой кривой с визуальным предпросмотром.

- [ ] Flat target (reference)
- [ ] High-pass / Low-pass фильтры (Butterworth, Bessel, Linkwitz-Riley 2nd–8th order)
- [ ] Super-Gaussian аппроксимация для плавных rolloff
- [ ] Tilt (dB/octave наклон)
- [ ] Low/High shelf с настраиваемой частотой и Q
- [ ] Целевая кривая рисуется поверх измерения на графике
- [ ] Сохранение/загрузка target presets (JSON)

**Deliverable:** Настраиваешь кроссовер + tilt → целевая кривая обновляется в реальном времени.

→ Details: [02-target-curve.md](02-target-curve.md)

---

## M3 — PEQ Engine (Weeks 5–7)

**Goal:** Авто-выравнивание: greedy → pruning → export списка PEQ фильтров.

- [ ] Психоакустическое сглаживание (variable smoothing: 1/48, 1/12, 1/3 oct)
- [ ] Greedy итеративный алгоритм (до 60 итераций)
- [ ] Приоритет пиков над провалами (peak bias)
- [ ] Эвристический Q: шире на НЧ, уже на ВЧ
- [ ] Pruning: последовательное удаление с проверкой tolerance
- [ ] **[Advanced]** Joint L-BFGS оптимизация (freq, gain, Q) после greedy
- [ ] PEQ band editor UI: таблица + drag на графике
- [ ] Manual add/remove/edit PEQ bands
- [ ] Export: список фильтров для miniDSP / APO / Roon формат

**Deliverable:** Нажал "Auto EQ" → появились PEQ фильтры на графике, таблица параметров, можно руками подправить.

→ Details: [03-peq-engine.md](03-peq-engine.md)

---

## M4 — FIR Engine (Weeks 8–10)

**Goal:** Генерация FIR correction impulse WAV файла.

- [ ] Линеаризация сетки (log → linear interpolation для FFT)
- [ ] Correction spectrum: `target_dB - current_dB`
- [ ] Boost limiting (+18dB cap, noise floor check)
- [ ] Cut application (minimal limiting)
- [ ] Phase correction strategies:
  - [ ] Linear phase (`correction = -measured_phase`)
  - [ ] Minimum phase (Hilbert transform)
  - [ ] Mixed phase (linear < crossover freq, minimum > crossover freq)
- [ ] Complex spectrum assembly + conjugate symmetry
- [ ] IFFT → cyclic rotation → windowing (Blackman / Kaiser / Tukey)
- [ ] Настройка tap count: 4096 – 262144
- [ ] WAV export: IEEE Float 64-bit и 32-bit
- [ ] Pre-ringing visualization для linear phase

**Deliverable:** Нажал "Generate FIR" → скачиваешь WAV impulse файл, готовый для convolver.

→ Details: [04-fir-engine.md](04-fir-engine.md)

---

## M5 — Advanced Features (Weeks 11–13)

**Goal:** Продвинутые возможности, выделяющие PhaseForge.

- [ ] Multi-point averaging (spatially weighted measurement fusion)
- [ ] Excess group delay targeting
- [ ] A/B comparison: bypass / corrected toggle
- [ ] Undo/redo (command pattern)
- [ ] Project save/load (`.phaseforge` JSON)
- [ ] Batch processing: папка измерений → папка FIR файлов
- [ ] Impulse viewer: time-domain plot с zoom

→ Details: [05-phase-engine.md](05-phase-engine.md)

---

## M6 — Polish & Release (Week 14+)

**Goal:** Production-ready macOS app.

- [ ] macOS code signing + notarization
- [ ] Auto-updater (Tauri updater plugin)
- [ ] Keyboard shortcuts (Cmd+O, Cmd+S, Cmd+Z, etc.)
- [ ] Dark/light theme
- [ ] Onboarding: первый запуск с demo измерением
- [ ] Performance profiling: FFT 256k taps < 500ms
- [ ] Documentation / user guide
- [ ] Landing page

---

## Dependency Graph

```
M0 (Scaffold)
 └─► M1 (Measurement I/O)
      ├─► M2 (Target Curve)
      │    └─► M3 (PEQ Engine)
      │         └─► M4 (FIR Engine)
      │              └─► M5 (Advanced)
      └─────────────────────► M6 (Polish)
```

## Risk Log

| Risk | Impact | Mitigation |
|------|--------|------------|
| FFT 256k taps slow on M1 Mac | Medium | `realfft` + NEON SIMD, benchmark early in M1 |
| REW format variations | Low | Collect 20+ sample files, fuzzy parser |
| uPlot 10k points laggy | Medium | Decimation for display, full data for computation |
| L-BFGS convergence on PEQ | Low | Greedy alone is sufficient, L-BFGS is nice-to-have |
| Tauri 2 macOS WebView quirks | Medium | Pin WebView version, test on Sonoma + Sequoia |

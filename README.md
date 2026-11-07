# PhaseForge

**Free FIR crossover and correction designer for active multi-way loudspeakers.**
macOS (Apple Silicon) and Windows. Beta. English and Russian UI.

You load your measurements, design every band of the speaker in one project (target,
crossover, PEQ, phase), see what the bands do together, and export one FIR filter
(WAV) per band for your convolver — HQPlayer, CamillaDSP, Equalizer APO and the like.

> **Non-commercial hobby project. No warranty. Use at your own risk.**
> Always check the generated filters before playing music through them: load the WAV
> into REW (or similar) and look at magnitude, phase and impulse, then start at low
> volume. A wrong filter — a tweeter without its high-pass, a big boost, a polarity or
> delay mistake — can damage drivers and ears. Much of the code was written with an AI
> coding assistant under the author's direction; that is one more reason to verify
> the output rather than trust it.

## Why

Doing a multi-way speaker in REPhase means one filter per band, a separate tool for PEQ,
a spreadsheet for delays, and no view of the sum until the files are loaded and measured
again. PhaseForge keeps all of it in one place:

- target, crossover, PEQ and phase of every band are designed together;
- the sum of all bands (SPL, phase, impulse, step, group delay) is visible live;
- what the plot shows is what the exported WAV contains — the plot and the file are
  built by the same code.

## Features

**Measurements**
- REW `.txt` and `.frd` import, with phase when present; propagation-delay removal.
- Near-field / far-field merge, baffle-step helper.
- Measurement analysis: noise floor, low-frequency roll-off and high-frequency cliff
  detection, with suggested optimization limits.

**Target per band**
- High-pass / low-pass: Linkwitz-Riley, Butterworth, Bessel, Gaussian, custom Q.
- Tilt, low/high shelves, Linkwitz transform.
- Complementary «1 − H» sections and nested crossovers for linear-phase splits that
  sum flat.
- Phase per section: minimum, linear or mixed (e.g. min-phase HP + linear-phase LP).

**PEQ**
- Manual filters or automatic optimization against the measurement.
- The optimizer limits Q and ringing; every filter shows its ringing time, and filters
  that lengthen the measured decay beyond what the ear masks are flagged.

**FIR export**
- 44.1–384 kHz, 4K to 1M taps, 20 window types, one WAV per band.
- Minimum-phase bands are built as an exact digital IIR cascade; everything else via a
  cepstral / FFT route. Above 88.2 kHz a zero-phase ultrasonic low-pass is applied.
- Per-band metrics: pre-ringing, pre-peak noise floor, energy after the peak, and the
  exact delay to enter in the convolver.
- Polarity is written into the file; level and alignment delay stay host-side settings.

**Sum view**
- Level, delay (positive = plays later, as in HQPlayer) and polarity per band.
- Summed measurement, target and corrected response: SPL, phase, IR/step, GD.

**Projects**
- Save/open, versions, undo/redo.

## How it is checked

The DSP is covered by automated tests (Rust and TypeScript), among them:

- comparison against REPhase reference filters and analytic filter responses;
- a 4-way export matrix — every filter family × phase mode × sample rate × tap count ×
  window: the WAVs are written, read back from disk, FFT'd and compared with the
  design, including the sum of the four bands;
- a release gate on a real 4-way project.

That catches a lot, not everything. See `CHANGELOG.md` for what changed and why.

## Download

[Releases](https://github.com/shd911/phaseforge/releases) — macOS `.dmg`
(Apple Silicon) and Windows `.exe` / `.msi`.

The builds are not code-signed. On macOS, if the app is reported as damaged or from an
unidentified developer, right-click → Open, or run
`xattr -dr com.apple.quarantine /Applications/PhaseForge.app`. On Windows, SmartScreen
may ask you to confirm («More info» → «Run anyway»).

## Building from source

Prerequisites: Rust (stable), Node.js 20+, Tauri CLI.

```bash
cargo install tauri-cli
npm install
cargo tauri dev      # development
cargo tauri build    # release bundle
```

Tests: `npx vitest run`, `npx tsc --noEmit`, and `cargo test` in `src-tauri/`.

Stack: Tauri 2, Rust (DSP), SolidJS + uPlot (UI).

## Feedback

Bug reports are welcome — ideally with the measurement files or the project
(File → Save As…) and the PhaseForge version from the window title.

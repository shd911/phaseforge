# PhaseForge — Modern DSP Room/Speaker Correction Tool

> Offline macOS-native successor to [RePhase](http://www.rephase.org), built with Tauri 2 + Rust + SolidJS.

## Vision

A precision tool for audiophiles and acoustic engineers to generate PEQ and FIR correction filters for loudspeaker/room systems. Combines the parametric EQ workflow of RePhase with modern auto-alignment, mixed-phase correction, and multi-point measurement support.

## Core Capabilities

| Feature | Description |
|---------|-------------|
| **Measurement Import** | REW `.txt`, `.frd`, `.mdat`, ARTA, HolmImpulse |
| **Auto PEQ Alignment** | Greedy iterative → pruning → optional joint L-BFGS optimization |
| **Target Curve Designer** | Crossovers (Bessel/Butterworth/LR), tilt, shelves, Super-Gaussian rolloff |
| **FIR Generation** | Correction spectrum → phase strategy → IFFT → windowing → WAV export |
| **Phase Strategies** | Linear phase, minimum phase, mixed phase (freq-dependent) |
| **Multi-point Averaging** | Spatially-weighted measurement fusion |
| **WAV Export** | IEEE Float 64-bit for lossless chaining; 32-bit for playback |

## Architecture

```
┌─────────────────────────────────┐
│   Frontend (Tauri WebView)      │
│   SolidJS + uPlot/Canvas        │
│   ├── FR/Phase plot (interactive)│
│   ├── PEQ band editor           │
│   ├── Target curve designer     │
│   ├── FIR config panel          │
│   └── Import / Export UI        │
└──────────┬──────────────────────┘
           │  Tauri IPC (invoke / events)
┌──────────▼──────────────────────┐
│   Rust Core (src-tauri/)        │
│   ├── io/                       │  Measurement & project I/O
│   ├── target/                   │  Target curve engine
│   ├── peq/                      │  Parametric EQ engine
│   ├── fir/                      │  FIR correction engine
│   ├── phase/                    │  Phase analysis & correction
│   ├── dsp/                      │  Shared DSP primitives
│   └── export/                   │  WAV / config export
└─────────────────────────────────┘
```

## Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Shell | Tauri 2 | Native macOS binary, ~5MB, no Electron bloat |
| Backend | Rust | FFT performance on 64k–256k taps, memory safety |
| Frontend | SolidJS | Fine-grained reactivity for real-time slider updates |
| Plots | uPlot + Canvas | 10k+ points at 60fps, interactive filter dragging |
| FFT | `rustfft` / `realfft` | Real-to-complex optimization (2× memory savings) |
| WAV I/O | `hound` | f32/f64 IEEE float support |
| Optimization | `argmin` | L-BFGS for joint PEQ optimization |
| Serialization | `serde` + JSON | Project save/load |

## Project Structure

```
phaseforge/
├── README.md
├── plans/                        # Detailed planning docs
│   ├── 00-milestones.md          # Milestone roadmap
│   ├── 01-measurement-io.md      # Measurement import/parsing
│   ├── 02-target-curve.md        # Target curve engine
│   ├── 03-peq-engine.md          # Auto PEQ alignment
│   ├── 04-fir-engine.md          # FIR correction generation
│   ├── 05-phase-engine.md        # Phase strategies
│   ├── 06-ui-architecture.md     # Frontend & interaction design
│   └── 07-export.md              # Export formats & pipeline
├── src-tauri/                    # Rust backend
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs
│       ├── io/
│       ├── target/
│       ├── peq/
│       ├── fir/
│       ├── phase/
│       ├── dsp/
│       └── export/
├── src/                          # SolidJS frontend
│   ├── App.tsx
│   ├── components/
│   ├── stores/
│   └── lib/
├── package.json
└── tauri.conf.json
```

## Quick Start (Development)

```bash
# Prerequisites: Rust toolchain, Node.js 20+
cargo install tauri-cli
npm install
cargo tauri dev
```

## License

TBD

## See Also

- `plans/00-milestones.md` — Development roadmap
- `plans/01-measurement-io.md` — Measurement format specs
- `plans/03-peq-engine.md` — Auto-alignment algorithm details
- `plans/04-fir-engine.md` — FIR generation pipeline

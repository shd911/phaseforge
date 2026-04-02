# PhaseForge — Project Rules

## Project Architecture
- **Stack**: Rust/Tauri 2 backend + SolidJS/TypeScript frontend. Do NOT assume Python/numpy for DSP code.
- **Versioning**: bNN (b89, b89.1, etc.) — increment on each release
- **Key dirs**: `src-tauri/src/` (Rust DSP, FIR, target eval), `src/components/` (SolidJS UI), `src/stores/` (state), `src/lib/` (helpers)
- **Version files**: `src-tauri/tauri.conf.json` (version + title), `src-tauri/src/lib.rs` (startup log)

## Debugging Rules
- **NEVER guess at root cause.** Always add diagnostic logging first, get user's console output, then fix based on evidence.
- Do not attempt more than 2 fix iterations without diagnostic data.
- Before fixing chart/rendering bugs, read the relevant render function END TO END — don't patch blindly.
- SolidJS gotcha: signals inside async functions are NOT tracked by createEffect. Read them synchronously before any await.

## Testing / Verification
- After every fix, verify that existing functionality on OTHER tabs/views still works.
- Fixes to one component must not regress others (Export fix must not break IR, snapshot fix must not break legends).
- Before committing, mentally trace: "what else reads/writes this state?" — if unsure, grep for the variable name.

## Workflow
- Commit after EACH completed block of changes — not after a giant batch.
- Format: `type: description (bNN)` — feat/fix/cleanup/chore
- Run code review (7-vector audit) before marking a feature complete.
- Co-Authored-By trailer in every commit.

## Build & Development
- **Dev**: `cargo tauri dev`
- **Release**: `cargo tauri build` -> launch from `target/release/bundle/macos/PhaseForge.app`
- **NEVER**: `cargo build --release` — it does NOT embed frontend assets correctly
- If white/black screen after build: kill all processes on port 1420, rebuild clean

## SolidJS Patterns
- `batch()`: wrap multiple signal updates to prevent intermediate effects
- PEQ drag: debounce via `peqDragging` signal + `setTimeout(150ms)`
- Store proxies: deep clone via `JSON.parse(JSON.stringify(obj))` before passing to async
- Gaussian min-phase: Rust returns 0 phase for Gaussian filters; frontend calls `compute_minimum_phase` (Hilbert) when `linear_phase=false`

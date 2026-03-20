# 01 — UI Specification

## 1. Layout Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  .top-bar (32px)                                                │
│  [Logo] PhaseForge v0.1.0 │ File │ [Standard│Hybrid] │ OptAll │ Info │
├─────────────────────────────────────────────────────────────────┤
│  .band-tabs (28px)                                              │
│  [SUM] [Band 1 ×] [Band 2 ×] ... [+]                           │
├───────────────────────────────────────────┬─────────────────────┤
│  .main-content-col (flex: 1)              │ .peq-sidebar (260px)│
│  ┌─ .freq-plot-area ────────────────────┐ │ (Target tab only)   │
│  │  FrequencyPlot (mag + phase)         │ │ ┌─────────────────┐ │
│  │  readout bar + axis controls         │ │ │ PEQ Bands [+]   │ │
│  └──────────────────────────────────────┘ │ │ Tolerance / Max  │ │
│  ── .resize-handle-plots (4px) ──         │ │ [Optimize][Clear]│ │
│  ┌─ bottom plot ────────────────────────┐ │ │ ┌─────────────┐ │ │
│  │  ImpulseResponsePlot (Meas/Target)   │ │ │ │ # F  G  Q × │ │ │
│  │  — or PeqResponsePlot (Target tab)   │ │ │ │ 1 63 -4  2  │ │ │
│  │  — or ExportImpulsePlot (Export tab)  │ │ │ │ 2 125 -7 3  │ │ │
│  └──────────────────────────────────────┘ │ │ └─────────────┘ │ │
│  ── .resize-handle (4px) ──               │ │ Mini IR plot    │ │
│  ┌─ .ctrl-wrap ─────────────────────────┐ │ └─────────────────┘ │
│  │  [Measurement] [Target] [Export]     │ │                     │
│  │  Control Panel body                  │ │                     │
│  └──────────────────────────────────────┘ │                     │
├───────────────────────────────────────────┴─────────────────────┤
│  .status-bar                                                    │
└─────────────────────────────────────────────────────────────────┘
```

### Responsive Modes

| Context | Top Plot | Bottom Plot | ControlPanel | PEQ Sidebar |
|---------|----------|-------------|--------------|-------------|
| SUM tab | FrequencyPlot (single) | — | Hidden | Hidden |
| Band → Measurement | FrequencyPlot | ImpulseResponsePlot | Visible | Hidden |
| Band → Target | FrequencyPlot | PeqResponsePlot | Visible | Visible (260px) |
| Band → Export | ExportPlot | ExportImpulsePlot | Visible (78px fixed) | Hidden |

- Plot split ratio: user-draggable, default 35% bottom
- Control panel height: draggable (min 80px, max 500px)
- X-scale sync: FrequencyPlot ↔ PeqResponsePlot share axis zoom/pan

---

## 2. Top Bar

Height: 32px. Background: `var(--bg-secondary)` (#26262e). Border-bottom: 1px solid `var(--border)`.

### Elements (left → right)

| # | Element | Details |
|---|---------|---------|
| 1 | Logo | `logo.png` (18px height) + "PhaseForge" (bold 13px) + version "v0.1.0-b86" (10px muted) |
| 2 | Separator | 1px vertical divider |
| 3 | FileMenu | Dropdown button "File", dot indicator when `isDirty()` |
| 4 | Separator | |
| 5 | Project Name | `projectName()` or filename or "Untitled", "(modified)" suffix when dirty, tooltip = full path |
| 6 | Separator | |
| 7 | Strategy Toggle | Two buttons: **Standard** \| **Hybrid**. Mutually exclusive. Toggles `exportHybridPhase` signal |
| 8 | Optimize All | Primary style (accent blue), disabled when `computing()`, shows "..." spinner, tooltip "Optimize PEQ for all bands" |
| 9 | Info (right-aligned) | SUM: "SUM — all bands". Band: "{BandName} · {MeasName} · {pts} pts" or "{BandName} · no measurement" |

### Strategy Toggle Behavior

| Mode | `exportHybridPhase` | PEQ Constraints | Phase Mode |
|------|---------------------|-----------------|------------|
| **Standard** | `false` | boost ≤ 6dB, cut ≤ 18dB, peak_bias 1.5 | MinimumPhase / LinearPhase |
| **Hybrid** | `true` | boost ≤ 60dB, cut ≤ 60dB, peak_bias 1.0 | HybridPhase |

Clicking either button triggers `handleOptimizeAll()` to re-optimize all bands with the new constraints.

---

## 3. File Menu

Dropdown appears on click. Items:

| Action | Shortcut | IPC Command | Notes |
|--------|----------|-------------|-------|
| New Project | Cmd+N | — | Shows ProjectNameDialog, creates folder |
| Open... | Cmd+O | `dialog.open` | File picker → `.pfproj` |
| Recent Projects | — | — | Submenu (hover delay 150ms), click → `loadProjectFromPath()` |
| Save | Cmd+S | `save_project` | Save to current path, or prompt if new |
| Save As... | Shift+Cmd+S | `save_project` | Always shows ProjectNameDialog |

### Save Flow
1. Copy pending measurements to project `inbox/` folder
2. Convert AppState → ProjectFile (camelCase → snake_case)
3. Write JSON to `.pfproj`
4. Set `isDirty(false)`

### Load Flow
1. Parse ProjectFile (snake_case → camelCase)
2. Re-import measurements from relative paths
3. Restore AppState, signals, tab state
4. Set `isDirty(false)`

### Project File Format (V2)
```json
{
  "version": 2,
  "app_name": "PhaseForge",
  "project_name": "My Project",
  "bands": [...],
  "active_band_id": "uuid",
  "show_phase": true,
  "show_mag": true,
  "show_target": true,
  "next_band_num": 4,
  "export_sample_rate": 48000,
  "export_taps": 65536,
  "export_window": "Blackman",
  "active_tab": "target"
}
```

Measurements stored as relative paths (`measurementFile`), not embedded.

---

## 4. Band Tabs

Height: 28px. Layout: `[SUM] [Band1] [Band2] ... [BandN] [+]`

### SUM Tab
- Always first position
- Shows `projectName()` or "SUM"
- Active when `activeBandId === "__sum__"`
- Styling: yellow/warning color, uppercase 10px, class `band-tab-sum`

### Band Tabs
- Display band name
- Close button (×) — visible on hover, only if `bands.length > 1`
- Confirmation dialog: "Delete band '{name}'?"
- Active underline animation, class `band-tab`

### Add Button (+)
- Rightmost position
- Creates band with auto-incremented name "Band {nextBandNum}"
- Sets new band as active

### Drag-and-Drop Reorder
- **Threshold:** 5px movement before drag starts
- **Visual feedback:** `dragging` class on source, `drag-over` on target
- **Implementation:** PointerEvent-based (not MouseEvent)

### `moveBand(fromIdx, toIdx)` Logic
After reorder:
1. First band: remove `high_pass` filter
2. Last band: remove `low_pass` filter
3. Rebuild linked connections:
   - Auto-link if `LP[i].freq ≈ HP[i+1].freq` within ±1%
   - Propagate filter params (type, order, linear_phase)

---

## 5. Plot Area

### 5.1 Frequency Plot (Top)

Component: `FrequencyPlot.tsx`

#### Band Mode Curves

| # | Series | Scale | Color | Category | Default Visible | Condition |
|---|--------|-------|-------|----------|----------------|-----------|
| 1 | `{name} dB` (Measurement) | mag | `MEAS_COLORS[0]` (#4A9EFF) | measurement | yes (non-Align) | `showMag && measurement` |
| 2 | `{name} °` (Meas phase) | phase | `MEAS_COLORS[0]` | measurement | yes (non-Align) | `showPhase && measurement.phase` |
| 3 | `Target dB` | mag | `#FFD700` | target | yes | `showTarget && targetMag` |
| 4 | `Target °` | phase | `#FFD700` | target | yes | `showTarget && showPhase && targetPhase` |
| 5 | `Meas 1/1 oct` | mag | `#FF6B6B` | measurement | no | `isAlignTab && showMag` |
| 6 | `PEQ Corrected dB` | mag | `#22C55E` | corrected | yes | `isHybrid && peqMag` |
| 7a | `Corrected + XO dB` | mag | `#F59E0B` (amber) | corrected | yes | `isHybrid && (hasPeq \|\| hasFilters)` |
| 7b | `Corrected dB` | mag | `#22C55E` | corrected | yes | `!isHybrid && (hasPeq \|\| hasFilters)` |
| 8a | `Corrected + XO °` | phase | `#F59E0B` | corrected | yes | `isHybrid && showPhase && meas.phase` |
| 8b | `Corrected °` | phase | `#22C55E` | corrected | yes | `!isHybrid && showPhase && meas.phase` |
| 9 | `{snap.label} dB` | mag | `snap.color` | corrected | yes | per snapshot |
| 10 | `{snap.label} °` | phase | `snap.color` | corrected | yes | per snapshot, if has phase |

- 7a/7b and 8a/8b are mutually exclusive (Hybrid OR Standard)
- #6 only in Hybrid mode when PEQ bands exist
- Corrected formula: `meas[i] + peqMag[i] + xsMag[i]`
- `xsMag` from `compute_cross_section`: `filt_mag + makeup_mag`

#### SUM Mode Curves

| # | Series | Scale | Color | Category | Default Visible | Condition |
|---|--------|-------|-------|----------|----------------|-----------|
| 1 | `{band.name} dB` | mag | `MEAS_COLORS[i]` | measurement | **no** | per band |
| 2 | `{band.name} °` | phase | `MEAS_COLORS[i]` | measurement | **no** | per band |
| 3 | `{band.name} tgt` | mag | `TARGET_BAND_COLORS[i]` | target | **no** | per band |
| 4 | `{band.name} corr+XO` | mag | `CORRECTED_BAND_COLORS[i]` | corrected | **no** | per band |
| 5 | `Sigma corr` | mag | `#22C55E` | corrected | **yes** | coherent sum corrected |
| 6 | `Sigma corr °` | phase | `#22C55E` | corrected | **yes** | coherent sum corrected phase |
| 7 | `Sigma dB` | mag | `#FFFFFF` | measurement | **yes** | coherent sum measurements |
| 8 | `Sigma °` | phase | `#FFFFFF` | measurement | **yes** | coherent sum meas phase |
| 9 | `Sigma tgt` | mag | `#FFD700` | target | **yes** | coherent sum normalized targets |
| 10 | `Sigma tgt °` | phase | `#FFD700` | target | **yes** | coherent sum target phase |

- Per-band series (1-4) repeat for each band, hidden by default
- Sigma series (5-10) visible by default
- No snapshots in SUM, no PEQ Corrected split, no Hybrid/Standard distinction

#### Readout Bar
- Cursor frequency (Hz), SPL (dB), phase (deg)
- Separator lines between items
- SUM mode: right-aligned legend with per-band/curve toggle checkboxes

#### Axis Controls (top-right corner)
- `+` / `−` vertical: zoom dB (centered on passband)
- `▲` / `▼`: scroll up/down
- `+` / `−` horizontal: zoom frequency
- `◀` / `▶`: scroll left/right
- `FIT`: reset to default scale

#### Y-Scale Anchor
- Center on passband reference level (adaptive to HP/LP filter range)
- Or 0 dB if no measurement

#### Mouse Interaction
- Wheel: Y-zoom (centered on cursor)
- Shift+Wheel: Y-scroll
- Double-click: fit data
- Click-drag: uPlot native zoom

#### Phase Wrapping
All displayed phase wrapped to [−180°, +180°].

### 5.2 Impulse Response Plot (Bottom, Non-SUM, Measurement tab)

Component: `ImpulseResponsePlot.tsx`

| Curve | Color | Toggle |
|-------|-------|--------|
| Impulse | `#4A9EFF` | Impulse mode |
| Step | `#22C55E` | Step mode |
| Target IR | `#FFD700` | — |
| Target Step | `#FFD700` | — |

- Readout: cursor time (ms), amplitude
- View mode selector: Impulse / Step
- Peak detection at ~25% from left edge
- Anchored zoom around peak

### 5.3 PEQ Response Plot (Bottom, Target tab)

Component: `PeqResponsePlot.tsx`

| Curve | Color |
|-------|-------|
| PEQ magnitude | `#38BDF8` (light blue) |
| PEQ phase | `#F59E0B` (amber) |

- X-scale synced with FrequencyPlot via `sharedXScale`
- Y-scale centered at 0 dB
- Readout: cursor frequency, magnitude, phase

### 5.4 Export Plot (Top, Export tab)

Component: `ExportPlot.tsx`

| Curve | Color |
|-------|-------|
| Model magnitude | `#FF9F43` (orange) |
| Model phase | `#FFCB80` (light orange) |
| FIR magnitude | `#38BDF8` (light blue) |
| FIR phase | `#7DD3FC` (lighter blue) |
| Snapshots | Cycling colors |

- Readout: cursor frequency, Model Mag, FIR Mag
- Y-scale persisted in `exportYScale` signal

### 5.5 Export Impulse Plot (Bottom, Export tab)

Component: `ExportImpulsePlot.tsx`

- FIR impulse response in `#38BDF8`
- Normalized scale (dB)
- Time range depends on tap count

---

## 6. Control Panel

Hidden on SUM tab. Three tabs: **Measurement** | **Target** | **Export**.

### 6.1 Measurement Tab

Badge indicator if measurement exists.

#### Table Columns
| Column | Content |
|--------|---------|
| Color dot | Measurement overlay color |
| Name | Clickable to rename |
| Points | Point count |
| Phase | Checkbox (if phase data exists) |
| Smoothing | Dropdown: off / 1/3 / 1/6 / 1/12 / 1/24 / var |
| Distance | Meters, from delay removal |
| Actions | Import, Merge, Delay Remove/Restore, Remove |

#### Actions
- **Import:** File dialog → `.txt` / `.frd` → `import_measurement` IPC
- **Merge NF+FF:** Opens MergeDialog
- **Delay Remove:** `remove_measurement_delay` IPC, saves `originalPhase`
- **Delay Restore:** Restores from `originalPhase`
- **Remove:** Clears measurement and settings

#### Smoothing Modes
| Mode | Description |
|------|-------------|
| `off` | No smoothing |
| `1/3` | 1/3 octave |
| `1/6` | 1/6 octave |
| `1/12` | 1/12 octave |
| `1/24` | 1/24 octave |
| `var` | Variable (psychoacoustic: 1/48 < 100Hz, 1/12 100-500Hz, 1/3 > 500Hz) |

### 6.2 Target Tab

Horizontal layout of filter blocks (`.filters-row`, flex gap 8px, each block min-width 170px).

#### General Block
| Control | Range | Default |
|---------|-------|---------|
| Toggle | ON / OFF | ON |
| Preset | flat / harman / bk / x-curve / custom | flat |
| Level (dB) | −20 to +20 | 0 |
| Tilt (dB/oct) | −6 to +6 | 0 |
| Invert | NOR / INV (+180°) | NOR |
| Export Target | button | — |

#### Presets

| Preset | Tilt | Low Shelf | High Shelf |
|--------|------|-----------|------------|
| Flat | 0 | — | — |
| Harman | −0.4 dB/oct | 200 Hz, +4 dB, Q 0.7 | 8000 Hz, −2 dB, Q 0.7 |
| B&K | −1.0 dB/oct | — | — |
| X-Curve | 0 | — | 2000 Hz, −6 dB, Q 0.5 |
| Custom | user | user | user |

#### High-Pass Block
| Control | Range | Default |
|---------|-------|---------|
| Frequency (Hz) | 10–20000 | — |
| Type | Butterworth / Bessel / LinkwitzRiley / Gaussian | Butterworth |
| Order | 1–8 (LR: 2,4,8 only) | 4 |
| Linear Phase | checkbox | false |
| Shape (M) | Gaussian only | — |

Link indicator: shows if HP linked from previous band's LP.

#### Low-Pass Block
Same controls as HP, plus:
- **Link button (🔗):** Links this LP to next band's HP
- On link: propagates freq, type, order, linear_phase to next band's HP

#### Linked Filter Behavior
- LP freq, type, order, linear_phase → HP of next band
- Uses `_propagating` flag to prevent infinite cycles
- Validation: HP freq ≤ LP freq

#### Low Shelf / High Shelf
| Control | Range | Default |
|---------|-------|---------|
| Frequency (Hz) | 10–20000 | — |
| Gain (dB) | −20 to +20 | 0 |
| Q | 0.1–10 | 0.707 |

### 6.3 Export Tab

Auto-sizes control panel to ~78px height.

#### FIR Config

| Control | Options / Range | Default |
|---------|-----------------|---------|
| Sample Rate | 44100, 48000, 88200, 96000, 176400, 192000 | 48000 |
| Taps | 4096–262144 (powers of 2) | 65536 |
| Window | 20+ types (see WindowType enum) | Blackman |

#### Window Types
Rectangular, Bartlett, Hann, Hamming, Blackman, ExactBlackman, BlackmanHarris, Nuttall3, Nuttall4, FlatTop, Kaiser, DolphChebyshev, Gaussian, Tukey, Lanczos, Poisson, HannPoisson, Bohman, Cauchy, Riesz

#### Phase Badge Logic
Phase mode auto-determined:
- All filters linear-phase → `LinearPhase`
- Mixed → depends on Strategy Toggle (Standard → MinimumPhase, Hybrid → HybridPhase)

#### Export Snapshots
- **Take Snapshot:** Captures current FIR curve
- **Clear Snapshots:** Removes all
- Labels: "Snap 1", "Snap 2", ...
- Colors: cycle through `EXPORT_SNAP_COLORS`

#### Export Flow
1. User configures SR / Taps / Window
2. Clicks **Export WAV**
3. File dialog → choose path
4. Backend generates FIR → writes WAV (IEEE Float 64-bit)

---

## 7. PEQ Sidebar

Width: 260px. Visible only on **Target tab**, non-SUM mode.

### Header
- "PEQ Bands" title (11px, uppercase)
- "+" button to add band manually

### Auto-Fit Section (if measurement exists)

| Control | Range | Default |
|---------|-------|---------|
| Tolerance (dB) | 0.5–3.0 | 1.0 |
| Max Bands | 1–60 | 20 |

- **Frequency range:** Auto-calculated `"{minHz}–{maxHz} Hz"` from HP/LP filters
- **Optimize button:** Primary style, disabled when `computing()`
- **Clear button:** Removes all PEQ bands
- **Status line:** `"{count} band(s) · max: {error}dB · {iters}it"`
- **Error display:** Red text on failure

### PEQ Table

| Column | Content |
|--------|---------|
| ☑ | Enable/disable checkbox |
| # | Row number |
| Freq (Hz) | 20–20000, wheel-scrollable |
| Gain (dB) | Color-coded: red < 0 (cut), green > 0 (boost), wheel-scrollable |
| Q | Quality factor, wheel-scrollable |
| × | Delete button |

#### Row States
- `peq-row-selected` — click to select/deselect
- `peq-row-pending` — newly added (until committed)
- `peq-row-disabled` — unchecked band

#### Wheel Handler
- Non-passive wheel events
- Shift+Wheel = 10× multiplier
- Adaptive step: < 100 Hz → 1, < 1000 Hz → 10, else → 100

#### Auto-sort
After edit, bands auto-sort by frequency. New bands inserted at index 0.

### Mini Impulse Plot
Below table, ~180px height. Shows corrected impulse response with PEQ contribution.

### Pending Workflow
1. User clicks "+" → new PeqBand added with `pending` flag
2. User edits freq/gain/Q
3. On commit (blur or Enter) → `commitPeqBand()` sorts by frequency, clears pending

---

## 8. Dialogs

### 8.1 Welcome Dialog
Component: `WelcomeDialog.tsx`

- **Shows when:** `currentProjectPath() === null`
- Logo + "PhaseForge" + "DSP Room & Speaker Correction" subtitle
- Buttons: **New Project** (primary), **Open Project**
- Recent Projects list (clickable)
- Dark overlay, centered dialog

### 8.2 Project Name Dialog
Component: `ProjectNameDialog.tsx`

| Mode | Controls |
|------|----------|
| New Project | Name input + Band count selector (−/count/+, range 1–8) |
| Save As | Name input only |

- Enter → create/save
- Escape → cancel
- Input auto-focused

### 8.3 Crossover Dialog
Component: `CrossoverDialog.tsx`

**Triggered:** Double-click on crossover marker in SUM plot.

| Control | Options |
|---------|---------|
| Frequency (Hz) | 20–20000 |
| Filter Type | LinkwitzRiley / Butterworth / Bessel / Gaussian |
| Order | 1–8 (LR: 2,4,8 only) |
| Linear Phase | checkbox |
| Shape (M) | Gaussian only |

Buttons: **Apply**, **Cancel**

### 8.4 Merge Dialog
Component: `MergeDialog.tsx`

**Purpose:** Blend near-field + far-field measurements.

| Section | Controls |
|---------|----------|
| File Selection | Near-Field button, Far-Field button, filename display |
| Parameters | Splice Freq (Hz, default 300), Blend Octaves (0.5–3.0, default 1.0), Level Offset (auto/manual), Match Range |
| Baffle Step | Enable checkbox, "Baffle Settings..." → opens BaffleStepDialog |

Buttons: **Merge**, **Close**

### 8.5 Baffle Step Dialog
Component: `BaffleStepDialog.tsx`

| Control | Range |
|---------|-------|
| Baffle Width (m) | 0.1–1.0 |
| Baffle Height (m) | 0.1–1.0 |
| Driver Offset X (m) | — |
| Driver Offset Y (m) | — |

Live preview chart: frequency response (log scale, −7 to +1 dB) showing F3 and edge frequencies.

---

## 9. SUM View

Active when `activeBandId === "__sum__"`.

### What's Hidden
- Control panel (all 3 tabs)
- PEQ sidebar
- Bottom plot (impulse/PEQ/export)
- Per-band curves (default hidden, togglable)

### What's Shown
- Single FrequencyPlot (full height)
- Sigma (Σ) curves: corrected, measurement, target — all with magnitude + phase
- Visibility matrix above chart: rows = bands, columns = categories (toggle checkboxes)
- Crossover markers (orange circles) on plot — draggable, double-click → CrossoverDialog

### Visibility Matrix
```
         │ Band 1 │ Band 2 │ Band 3 │  Σ  │
─────────┼────────┼────────┼────────┼─────┤
Targets  │   ◼    │   ◼    │   ◼    │  ◼  │  ← click = toggle row
Meas     │        │   ◼    │        │  ◼  │
Corrected│        │   ◼    │        │  ◼  │
         │  click = toggle column          │
```

### Key Differences from Band Mode

| Aspect | Band Mode | SUM Mode |
|--------|-----------|----------|
| Measurement | 1 curve (active band) | Per-band (hidden) + Σ meas (visible) |
| Target | 1 curve (active band) | Per-band (hidden) + Σ target (visible) |
| Corrected | PEQ Corrected¹ + Full Corrected | Per-band corr+XO (hidden) + Σ corrected (visible) |
| Phase | meas°, target°, corrected° | Per-band meas°, Σ meas°, Σ corr°, Σ target° |
| Snapshots | Yes | No |
| 1/1 oct | Yes (Align tab) | No |
| Hybrid split | Yes (PEQ + Corrected+XO) | No |

¹ PEQ Corrected — only in Hybrid mode

---

## 10. Snapshot System

### Frequency Snapshots (`FreqSnapshot`)
- **Storage:** `freqSnapshots` signal, keyed by `bandId`
- **Trigger:** "Take Snap" button in FrequencyPlot toolbar
- **Data:** `{ label, freq[], mag[], phase[], color }`
- **Colors:** Cycle through `FREQ_SNAP_COLORS` (#808080, #A855F7, #EC4899, #14B8A6)
- **Max:** 4 per band (cycling)
- **Scope:** Per-band, survive tab switches
- **Not available in SUM mode**

### Export Snapshots (`ExportSnapshot`)
- **Storage:** `exportSnapshots` signal, keyed by `bandId`
- **Trigger:** "Take Snapshot" button in Export tab
- **Data:** Same structure as FreqSnapshot
- **Colors:** Same cycling palette
- **Scope:** Per-band

### Cleanup
- `removeBand()` clears snapshots for that band
- `clearAllFreqSnapshots()` / `clearAllExportSnapshots()` — global clear

---

## 11. State Management

### AppState (`stores/bands.ts`)

```typescript
interface AppState {
  bands: BandState[]
  activeBandId: string        // "__sum__" or band UUID
  showPhase: boolean          // true (always-on since b82.06)
  showMag: boolean            // true (always-on)
  showTarget: boolean         // true (always-on)
  nextBandNum: number
}
```

### BandState

```typescript
interface BandState {
  id: string                  // UUID
  name: string
  measurement: Measurement | null
  measurementFile: string | null   // relative path in project
  settings: PerMeasurementSettings | null
  target: TargetCurve
  targetEnabled: boolean
  inverted: boolean           // polarity flip (+180°)
  linkedToNext: boolean       // LP ↔ HP linking
  peqBands: PeqBand[]
  firResult: FirResult | null
  crossNormDb: number
}
```

### PerMeasurementSettings

```typescript
interface PerMeasurementSettings {
  smoothing: "off" | "1/3" | "1/6" | "1/12" | "1/24" | "var"
  delay_seconds: number | null
  distance_meters: number | null
  delay_removed: boolean
  originalPhase: number[] | null
  floorBounce: FloorBounceConfig | null
  mergeSource: MergeSource | null
}
```

### Global Signals

| Signal | Type | Default |
|--------|------|---------|
| `activeTab` | `"measurements" \| "target" \| "export"` | `"measurements"` |
| `selectedPeqIdx` | `number \| null` | `null` |
| `plotShowOnly` | `string[] \| null` | `null` |
| `sharedXScale` | `{ min: number, max: number }` | — |
| `exportSampleRate` | `number` | `48000` |
| `exportTaps` | `number` | `65536` |
| `exportWindow` | `WindowType` | `"Blackman"` |
| `exportHybridPhase` | `boolean` | `false` |
| `exportYScale` | `{ min, max } \| null` | `null` |
| `isDirty` | `boolean` | `false` |
| `currentProjectPath` | `string \| null` | `null` |
| `projectDir` | `string \| null` | `null` |
| `projectName` | `string \| null` | `null` |

### PEQ Optimize Signals (`stores/peq-optimize.ts`)

| Signal | Type | Default |
|--------|------|---------|
| `tolerance` | `number` | `1.0` |
| `maxBands` | `number` | `20` |
| `computing` | `boolean` | `false` |
| `peqError` | `string \| null` | `null` |
| `maxErr` | `number \| null` | `null` |
| `iters` | `number \| null` | `null` |

### Key Store Functions

| Function | Description |
|----------|-------------|
| `addBand()` | Create band, set active |
| `removeBand(id)` | Delete band, cleanup snapshots |
| `setActiveBand(id)` | Switch active band |
| `setActiveBandSum()` | Switch to SUM view |
| `moveBand(from, to)` | Reorder + crossover rebuild |
| `setBandMeasurement(id, m)` | Set measurement, reset settings |
| `replaceBandMeasurement(id, m)` | Update measurement, preserve settings |
| `setBandSmoothing(id, mode)` | Set smoothing |
| `markBandDelayRemoved(id, phase)` | Apply delay removal, save original |
| `restoreBandDelay(id)` | Restore original phase |
| `toggleBandTarget(id)` | Enable/disable target |
| `toggleBandInverted(id)` | Toggle polarity |
| `toggleBandLinked(id)` | Toggle LP ↔ HP linking |
| `setBandHighPass/LowPass(id, cfg)` | Set filter, propagate linked |
| `setBandPeqBands(id, bands)` | Set optimized PEQ |
| `updatePeqBand(id, idx, patch)` | Edit single PEQ band |
| `commitPeqBand(id, idx)` | Sort by frequency |
| `removePeqBand(id, idx)` | Delete PEQ band |
| `handleOptimizePeq()` | Auto-fit single band |
| `handleOptimizeAll()` | Batch optimize all bands |

---

## 12. Keyboard Shortcuts

| Shortcut | Action | Implementation |
|----------|--------|----------------|
| Cmd+N | New Project | `App.tsx` keydown handler |
| Cmd+O | Open Project | `App.tsx` keydown handler |
| Cmd+S | Save Project | `App.tsx` keydown handler |
| Shift+Cmd+S | Save As | `App.tsx` keydown handler |

All shortcuts `preventDefault()` on match.

---

## 13. Known Bugs

### BUG-001: Filter Reset on OFF/ON

**Severity:** Medium
**Repro:**
1. Set up HP/LP filters on Target tab
2. Toggle target OFF
3. Toggle target ON
4. Filters are reset to defaults

**Root cause:** `toggleBandTarget()` may reinitialize target object instead of preserving filter config.

### BUG-002: Visibility Reset

**Severity:** Low
**Repro:**
1. In SUM mode, toggle some per-band curves visible
2. Switch to a band tab
3. Switch back to SUM
4. Visibility state is reset

**Root cause:** SUM visibility matrix state not persisted across tab switches.

---

## 14. Agreed Improvements

| Improvement | Status | Description |
|-------------|--------|-------------|
| Drag-and-Drop Band Reorder | Implemented | PointerEvent-based, with crossover rebuild |
| Export All | Planned | Export all bands in one click |
| Autosave | Planned | Periodic save to project file |
| Tooltips | Partial | Some controls have tooltips, not comprehensive |

---

## 15. Industrial Context

### Comparison Table

| Feature | REW | RePhase | PhaseForge |
|---------|-----|---------|------------|
| Platform | Java (cross) | Windows | macOS (Tauri) |
| Measurement Import | Native | FRD import | REW .txt, .frd |
| Target Curve | Basic | Full designer | Full + presets |
| PEQ Auto-Fit | Yes (basic) | No | Greedy + L-BFGS |
| FIR Generation | No | Yes (manual) | Auto from PEQ + target |
| Phase Strategies | N/A | Linear/Min/Mixed | Linear/Min/Mixed/Hybrid |
| Multi-band (crossover) | No | No | Yes (linked bands) |
| SUM View | No | No | Yes (coherent sum) |
| NF+FF Merge | Yes | No | Yes |
| Project Format | .mdat | .rpp | .pfproj (JSON) |
| Real-time Preview | Yes | No | No |

### PhaseForge Differentiators
1. **Multi-band architecture** — first-class crossover support with linked HP/LP
2. **SUM view** — coherent summation of all bands for system-level verification
3. **Hybrid Phase strategy** — combines linear-phase crossover with minimum-phase PEQ
4. **Auto PEQ + FIR pipeline** — single-click from measurement to correction WAV
5. **Modern stack** — native macOS, fast DSP via Rust, reactive UI via SolidJS

---

## 16. Component Tree

```
App.tsx
├── WelcomeDialog
├── ProjectNameDialog
├── CrossoverDialog
├── MergeDialog
│   └── BaffleStepDialog
├── .top-bar
│   ├── Logo + Version
│   ├── FileMenu (dropdown)
│   ├── Project Name display
│   ├── Strategy Toggle [Standard | Hybrid]
│   ├── Optimize All button
│   └── Info text
├── BandTabs
│   ├── SUM tab
│   ├── Band tabs (×N, draggable)
│   └── Add (+) button
├── .main-content-row
│   ├── .main-content-col
│   │   ├── FrequencyPlot (always)
│   │   │   ├── readout bar
│   │   │   ├── axis controls
│   │   │   └── SUM legend matrix
│   │   ├── resize-handle-plots
│   │   ├── ImpulseResponsePlot (Measurement tab)
│   │   │   — or PeqResponsePlot (Target tab)
│   │   │   — or ExportImpulsePlot (Export tab)
│   │   ├── resize-handle
│   │   └── ControlPanel
│   │       ├── Tab: Measurement
│   │       │   └── NumberInput, dropdowns, buttons
│   │       ├── Tab: Target
│   │       │   ├── General block
│   │       │   ├── HP block
│   │       │   ├── LP block
│   │       │   ├── Low Shelf block
│   │       │   └── High Shelf block
│   │       └── Tab: Export
│   │           └── SR, Taps, Window, Export buttons
│   └── PeqSidebar (Target tab only)
│       ├── Auto-Fit controls
│       ├── PEQ Table (NumberInput per cell)
│       └── Mini impulse plot
└── .status-bar
```

### Component Files (17 total)

| Component | File | Purpose |
|-----------|------|---------|
| App | `App.tsx` | Main layout, keyboard shortcuts, resize handlers |
| FileMenu | `FileMenu.tsx` | File operations dropdown |
| BandTabs | `BandTabs.tsx` | Band tab bar with drag-to-reorder |
| ControlPanel | `ControlPanel.tsx` | Measurements / Target / Export tabs |
| FrequencyPlot | `FrequencyPlot.tsx` | Frequency response visualization |
| ImpulseResponsePlot | `ImpulseResponsePlot.tsx` | Impulse/step response |
| PeqResponsePlot | `PeqResponsePlot.tsx` | PEQ filter response |
| PeqSidebar | `PeqSidebar.tsx` | PEQ band table + controls |
| ExportPlot | `ExportPlot.tsx` | FIR model visualization |
| ExportImpulsePlot | `ExportImpulsePlot.tsx` | FIR impulse response |
| NumberInput | `NumberInput.tsx` | Custom numeric input (−/value/+) |
| WelcomeDialog | `WelcomeDialog.tsx` | Startup screen |
| ProjectNameDialog | `ProjectNameDialog.tsx` | Name prompt for new/save |
| CrossoverDialog | `CrossoverDialog.tsx` | Crossover filter editor |
| MergeDialog | `MergeDialog.tsx` | NF+FF merge dialog |
| BaffleStepDialog | `BaffleStepDialog.tsx` | Baffle config & preview |

---

## 17. Color Palette

### Theme Variables

| Variable | Hex | Usage |
|----------|-----|-------|
| `--bg-primary` | `#1e1e24` | Main background |
| `--bg-secondary` | `#26262e` | Panel backgrounds |
| `--bg-surface` | `#2e2e38` | Cards, elevated surfaces |
| `--bg-input` | `#1a1a20` | Input field backgrounds |
| `--bg-hover` | `#34343e` | Hover state background |
| `--text-primary` | `#d4d4d8` | Main text |
| `--text-secondary` | `#8b8b96` | Labels, secondary text |
| `--text-muted` | `#5a5a66` | Disabled, tertiary |
| `--accent` | `#4a90d9` | Buttons, highlights |
| `--accent-hover` | `#5da3e8` | Accent hover |
| `--accent-dim` | `#3a6fa0` | Selected background |
| `--border` | `#3a3a48` | Borders |
| `--border-focus` | `#4a90d9` | Focus border |
| `--danger` | `#d94a4a` | Error red |
| `--success` | `#4ad97a` | Success green |
| `--warning` | `#d9a44a` | Warning yellow |
| `--target-color` | `#FFD700` | Target curve gold |

### Plot Colors

| Purpose | Color | Hex |
|---------|-------|-----|
| Measurement (default) | Blue | `#4A9EFF` |
| Target curve | Gold | `#FFD700` |
| Corrected | Green | `#22C55E` |
| Corrected + XO (Hybrid) | Amber | `#F59E0B` |
| Smoothed 1/1 oct | Red | `#FF6B6B` |
| Σ measurement (SUM) | White | `#FFFFFF` |
| PEQ magnitude | Light blue | `#38BDF8` |
| PEQ phase | Amber | `#F59E0B` |
| Model magnitude (Export) | Orange | `#FF9F43` |
| Model phase (Export) | Light orange | `#FFCB80` |
| FIR magnitude (Export) | Light blue | `#38BDF8` |
| FIR phase (Export) | Lighter blue | `#7DD3FC` |

### Per-Band Color Arrays (8 entries each, cycling)

**Measurement:** `#4A9EFF, #FF8C42, #22C55E, #A855F7, #EF4444, #06B6D4, #F59E0B, #EC4899`

**Target (SUM):** `#7CB3FF, #FFB870, #6EE89A, #C98DF7, #F77070, #5CD6E8, #F7C35B, #F77DBF`

**Corrected (SUM):** `#34D399, #4ADE80, #2DD4BF, #A3E635, #86EFAC, #5EEAD4, #BEF264, #6EE7B7`

**Snapshots:** `#808080, #A855F7, #EC4899, #14B8A6` (4 entries, cycling)

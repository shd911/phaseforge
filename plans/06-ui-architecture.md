# 06 — UI Architecture

## Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Menu Bar (native macOS)                                     │
│  File │ Edit │ View │ Tools │ Help                           │
├─────────────────────────────────────────────────────────────┤
│  Toolbar                                                     │
│  [Import] [Auto EQ] [Generate FIR] [Export] │ A/B │ Undo   │
├──────────────────────────────────┬──────────────────────────┤
│                                  │                          │
│  Main Plot Area (70%)            │  Side Panel (30%)        │
│                                  │                          │
│  ┌────────────────────────────┐  │  ┌────────────────────┐ │
│  │  Magnitude (dB) vs Freq   │  │  │  Target Curve      │ │
│  │  - Raw (blue)             │  │  │  HP: 20Hz LR4      │ │
│  │  - Smoothed (cyan)        │  │  │  LP: off            │ │
│  │  - Target (orange dashed) │  │  │  Tilt: -0.5 dB/oct │ │
│  │  - Corrected (green)      │  │  │  Preset: [Harman▼] │ │
│  │  - PEQ bands (draggable)  │  │  └────────────────────┘ │
│  └────────────────────────────┘  │                          │
│  ┌────────────────────────────┐  │  ┌────────────────────┐ │
│  │  Phase (°) / GD (ms)      │  │  │  PEQ Bands         │ │
│  │  - Current phase          │  │  │  #1: 63Hz -4.2 Q2  │ │
│  │  - Group delay            │  │  │  #2: 125Hz -6.8 Q3 │ │
│  │  - Excess GD              │  │  │  #3: 250Hz +2.1 Q4 │ │
│  └────────────────────────────┘  │  │  [+ Add] [Auto]    │ │
│                                  │  └────────────────────┘ │
│                                  │                          │
│                                  │  ┌────────────────────┐ │
│                                  │  │  FIR Config        │ │
│                                  │  │  Taps: [65536▼]    │ │
│                                  │  │  Phase: [Mixed▼]   │ │
│                                  │  │  Window: [Blackman▼]│ │
│                                  │  │  [Generate]        │ │
│                                  │  └────────────────────┘ │
├──────────────────────────────────┴──────────────────────────┤
│  Status Bar: "12 PEQ bands │ Max error: 1.2 dB │ Ready"    │
└─────────────────────────────────────────────────────────────┘
```

---

## Technology Choices

### SolidJS

Преимущества над React для этого приложения:
- Fine-grained reactivity: изменение одного слайдера не перерисовывает всё дерево
- Меньший bundle size (~7KB vs ~45KB React)
- Нет virtual DOM overhead — критично для 60fps слайдеров
- Stores: реактивное состояние без Redux boilerplate

### Plotting: Canvas / uPlot

**uPlot** для основных графиков:
- 10K+ точек без проседания
- Built-in zoom/pan
- Dual Y-axis (dB + degrees)
- Lightweight (~35KB)

**Canvas overlay** для интерактивных элементов:
- Draggable PEQ band markers (circles on magnitude plot)
- Hover tooltip с freq/dB
- Rubber-band selection

### State Management

```typescript
// SolidJS store
const [state, setState] = createStore({
  measurements: [] as Measurement[],
  activeMeasurement: 0,
  target: defaultTargetCurve(),
  peqBands: [] as PeqBand[],
  firConfig: defaultFirConfig(),
  ui: {
    showPhase: true,
    showGroupDelay: false,
    showSmoothed: true,
    abMode: 'after' as 'before' | 'after' | 'overlay',
    selectedBand: null as number | null,
  }
});
```

---

## Interactive PEQ Editing

### Drag on Plot

1. PEQ band отображается как кружок на графике magnitude
2. Горизонтальный drag → изменение freq
3. Вертикальный drag → изменение gain
4. Scroll wheel на кружке → изменение Q
5. Double-click → открыть precision editor (числовые поля)

```typescript
function handleBandDrag(bandIdx: number, event: PointerEvent) {
  const freq = xPixelToFreq(event.offsetX);  // log scale
  const gain = yPixelToDb(event.offsetY);
  
  setState('peqBands', bandIdx, { freq_hz: freq, gain_db: gain });
  
  // Пересчёт corrected curve через IPC
  invoke('compute_corrected', { 
    measurement: state.measurements[state.activeMeasurement],
    bands: state.peqBands 
  }).then(corrected => updatePlot(corrected));
}
```

### Real-time Update

При drag'е → debounced IPC call (16ms = 60fps). Rust пересчитывает Σ PEQ responses и возвращает corrected curve. Frontend обновляет plot.

---

## Tauri IPC Protocol

```typescript
// Frontend → Rust
type Commands = {
  import_measurement: (path: string) => Measurement;
  compute_smoothed: (m: Measurement, config: SmoothingConfig) => number[];
  compute_target: (target: TargetCurve, freq: number[]) => number[];
  auto_eq: (m: Measurement, target: number[], config: AutoEqConfig) => PeqBand[];
  compute_corrected: (m: Measurement, bands: PeqBand[]) => number[];
  generate_fir: (m: Measurement, bands: PeqBand[], target: TargetCurve, config: FirConfig) => FirResult;
  export_wav: (fir: FirResult, path: string, config: ExportConfig) => void;
  export_peq: (bands: PeqBand[], format: string, path: string) => void;
  save_project: (project: Project, path: string) => void;
  load_project: (path: string) => Project;
};

// Rust → Frontend (events)
type Events = {
  progress: { stage: string, percent: number };
  fir_preview: { impulse_snippet: number[] };  // first 1000 samples for preview
};
```

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Cmd+O` | Import measurement |
| `Cmd+S` | Save project |
| `Cmd+Shift+S` | Save project as... |
| `Cmd+E` | Export FIR WAV |
| `Cmd+Z` | Undo |
| `Cmd+Shift+Z` | Redo |
| `Space` | Toggle A/B |
| `A` | Run Auto EQ |
| `Delete` | Remove selected PEQ band |
| `+` / `-` | Zoom in/out on plot |
| `Cmd+0` | Fit plot to data |
| `P` | Toggle phase plot |
| `G` | Toggle group delay |

---

## Color Scheme

| Element | Color | Hex |
|---------|-------|-----|
| Raw measurement | Blue | `#4A9EFF` |
| Smoothed | Cyan | `#00D4AA` |
| Target curve | Orange dashed | `#FF8C42` |
| Corrected | Green | `#22C55E` |
| PEQ response | Purple | `#A855F7` |
| Error fill (+) | Red translucent | `#EF444430` |
| Error fill (-) | Green translucent | `#22C55E30` |
| Grid | Gray | `#333333` / `#CCCCCC` |

Dark theme primary. Light theme optional.

---

## Responsive Panels

Side panel collapsible (toggle button). В collapsed mode — full-width plot. Полезно для детального анализа.

Нижний plot (phase/GD) — resizable по высоте через drag separator.

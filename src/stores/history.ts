// ---------------------------------------------------------------------------
// Session undo/redo (b132). Light-state snapshots only — measurement arrays
// and originalPhase are NOT included; band identity is preserved across undo
// so existing measurement data is reused from current state on apply.
// ---------------------------------------------------------------------------

import { createSignal } from "solid-js";
import type { PeqBand, TargetCurve, WindowType, ExclusionZone } from "../lib/types";
import type { PeqRangeMode } from "./peq-optimize";

export interface LightBand {
  id: string;
  name: string;
  peqBands: PeqBand[];
  target: TargetCurve;
  targetEnabled: boolean;
  inverted: boolean;
  linkedToNext: boolean;
  alignmentDelay: number;
  color: string;
  exclusionZones: ExclusionZone[];
}

export interface PeqParamsSnap {
  tolerance: number;
  maxBands: number;
  gainRegularization: number;
  peqFloor: number;
  peqRangeMode: PeqRangeMode;
  peqDirectLow: number;
  peqDirectHigh: number;
}

export interface FirParamsSnap {
  iterations: number;
  freqWeighting: boolean;
  narrowbandLimit: boolean;
  nbSmoothingOct: number;
  nbMaxExcess: number;
  maxBoost: number;
  noiseFloor: number;
}

export interface ExportParamsSnap {
  sampleRate: number;
  taps: number;
  window: WindowType;
  hybridPhase: boolean;
}

export interface HistoryEntry {
  bands: LightBand[];
  activeBandId: string;
  nextBandNum: number;
  peqParams: PeqParamsSnap;
  firParams: FirParamsSnap;
  exportParams: ExportParamsSnap;
  label: string;
  ts: number;
}

const MAX_HISTORY = 5;
const COALESCE_MS = 250;

const undoStack: HistoryEntry[] = [];
const redoStack: HistoryEntry[] = [];

let _activeBuffer: HistoryEntry | null = null;
let _coalesceLabel = "";
let _coalesceTs = 0;
let _suppressed = false;

let _captureFn: ((label: string) => HistoryEntry) | null = null;
let _applyFn: ((entry: HistoryEntry) => void) | null = null;

const [version, setVersion] = createSignal(0);

export function registerHistoryHooks(
  capture: (label: string) => HistoryEntry,
  apply: (entry: HistoryEntry) => void,
): void {
  _captureFn = capture;
  _applyFn = apply;
}

function bump() { setVersion((v) => v + 1); }

/** Push current state to undo stack. Coalesces consecutive same-label calls
 *  within COALESCE_MS to absorb rapid hammering (NumberInput repeat, wheel). */
export function pushHistory(label: string): void {
  if (_suppressed || _activeBuffer || !_captureFn) return;
  const now = performance.now();
  if (label === _coalesceLabel && now - _coalesceTs < COALESCE_MS) {
    _coalesceTs = now;
    return;
  }
  _coalesceLabel = label;
  _coalesceTs = now;
  undoStack.push(_captureFn(label));
  if (undoStack.length > MAX_HISTORY) undoStack.shift();
  redoStack.length = 0;
  bump();
}

/** Capture pre-state into a buffer. Subsequent calls are no-ops until
 *  commitInteraction or discardInteraction. Use for drag operations. */
export function beginInteraction(label: string): void {
  if (_activeBuffer || !_captureFn) return;
  _activeBuffer = _captureFn(label);
}

/** Flush buffered pre-state to undo stack. */
export function commitInteraction(): void {
  if (!_activeBuffer) return;
  undoStack.push(_activeBuffer);
  if (undoStack.length > MAX_HISTORY) undoStack.shift();
  _activeBuffer = null;
  _coalesceLabel = "";
  _coalesceTs = 0;
  redoStack.length = 0;
  bump();
}

export function discardInteraction(): void {
  _activeBuffer = null;
}

export function undo(): void {
  if (!_captureFn || !_applyFn || undoStack.length === 0) return;
  // Drop any in-flight drag buffer — its pre-state is now stale relative to
  // the state we're about to restore.
  _activeBuffer = null;
  const current = _captureFn(undoStack[undoStack.length - 1].label);
  redoStack.push(current);
  if (redoStack.length > MAX_HISTORY) redoStack.shift();
  const prev = undoStack.pop()!;
  _suppressed = true;
  try { _applyFn(prev); } finally { _suppressed = false; }
  _coalesceLabel = ""; _coalesceTs = 0;
  bump();
}

export function redo(): void {
  if (!_captureFn || !_applyFn || redoStack.length === 0) return;
  _activeBuffer = null;
  const current = _captureFn(redoStack[redoStack.length - 1].label);
  undoStack.push(current);
  if (undoStack.length > MAX_HISTORY) undoStack.shift();
  const next = redoStack.pop()!;
  _suppressed = true;
  try { _applyFn(next); } finally { _suppressed = false; }
  _coalesceLabel = ""; _coalesceTs = 0;
  bump();
}

export function clearHistory(): void {
  undoStack.length = 0;
  redoStack.length = 0;
  _activeBuffer = null;
  _coalesceLabel = "";
  _coalesceTs = 0;
  bump();
}

export function canUndo(): boolean { void version(); return undoStack.length > 0; }
export function canRedo(): boolean { void version(); return redoStack.length > 0; }
export function lastUndoLabel(): string | null {
  void version();
  return undoStack.length > 0 ? undoStack[undoStack.length - 1].label : null;
}
export function lastRedoLabel(): string | null {
  void version();
  return redoStack.length > 0 ? redoStack[redoStack.length - 1].label : null;
}

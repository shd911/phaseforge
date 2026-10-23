// ---------------------------------------------------------------------------
// Session undo/redo (b132). Light-state snapshots only — measurement arrays
// and originalPhase are NOT included; band identity is preserved across undo
// so existing measurement data is reused from current state on apply.
//
// b141.103 (audit 2026-10-23 3.1): history holds BAND CONTENT only — target,
// PEQ, crossovers, order, names, delays, smoothing. Global export / FIR /
// optimizer settings (SR, taps, window, Standard/Hybrid, «Порог», «Подъём до»,
// «Без отражений», …) are not part of an entry: undoing a slope change used
// to put back the SR/taps chosen before it as well.
// ---------------------------------------------------------------------------

import { createSignal } from "solid-js";
import type { PeqBand, TargetCurve, ExclusionZone, PeqOptimizedTarget } from "../lib/types";

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
  peqOptimizedTarget: PeqOptimizedTarget | null;
  /** b141.103 (audit 3.9): measurement smoothing is band content — the
   *  «Сглаживание» recommendation pushed an entry that restored nothing.
   *  null = the band had no measurement settings. */
  smoothing?: string | null;
}

export interface HistoryEntry {
  bands: LightBand[];
  activeBandId: string;
  nextBandNum: number;
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
/** b141.103 (audit 3.10): bumped whenever the state jumps outside the normal
 *  edit flow (undo, redo, history cleared by open/new). An async operation
 *  that pushed its entry before an await compares it after: a result landing
 *  after an undo used to sit on top of the restored state with no entry of
 *  its own, and the next undo rolled back two steps. */
let _epoch = 0;
export function historyEpoch(): number { return _epoch; }

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

/** One undo step for a compound action: the pre-state is pushed once and
 *  the pushHistory calls of the setters inside are absorbed. Inside a
 *  begin/commitInteraction buffer it just runs `fn` (the buffer is the step). */
export function historyGroup(label: string, fn: () => void): void {
  if (_activeBuffer) { fn(); return; }
  pushHistory(label);
  const prev = _suppressed;
  _suppressed = true;
  try { fn(); } finally { _suppressed = prev; }
}

/** Band ids referenced by any undo/redo entry (or the open drag buffer). */
export function historyBandIds(): Set<string> {
  const ids = new Set<string>();
  for (const e of [...undoStack, ...redoStack, ...(_activeBuffer ? [_activeBuffer] : [])]) {
    for (const b of e.bands) ids.add(b.id);
  }
  return ids;
}

/** Capture pre-state into a buffer. Subsequent calls are no-ops until
 *  commitInteraction. Use for drag operations. */
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

export function undo(): void {
  if (!_captureFn || !_applyFn || undoStack.length === 0) return;
  // Drop any in-flight drag buffer — its pre-state is now stale relative to
  // the state we're about to restore.
  _activeBuffer = null;
  const current = _captureFn(undoStack[undoStack.length - 1].label);
  redoStack.push(current);
  if (redoStack.length > MAX_HISTORY) redoStack.shift();
  const prev = undoStack.pop()!;
  _epoch++;
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
  _epoch++;
  _suppressed = true;
  try { _applyFn(next); } finally { _suppressed = false; }
  _coalesceLabel = ""; _coalesceTs = 0;
  bump();
}

export function clearHistory(): void {
  _epoch++;
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

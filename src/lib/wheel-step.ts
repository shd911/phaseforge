// b141.43: one wheel model for every field that adjusts by scroll wheel —
// NumberInput, the PEQ table, the Σ alignment delay and PEQ Q on the chart.
//
// The four sites each took one step per wheel EVENT. A mouse sends one event
// per notch, but a trackpad or Magic Mouse sends dozens of small pixel deltas
// per gesture, so values ran away. The PEQ table and the delay field also
// called preventDefault before they were activated, which froze panel
// scrolling whenever the cursor crossed them, and every step started a full
// re-evaluation (FIR on the Export tab included).

/** Pixel travel that makes one step on continuous devices (trackpad). */
export const WHEEL_PX_PER_STEP = 40;
/** Pause that ends a gesture: accumulator resets, onEnd fires. */
export const WHEEL_GESTURE_IDLE_MS = 350;
/** Legacy wheelDelta of one mouse notch (Windows / Chromium). */
const NOTCH = 120;
/** One notch of a mouse wheel in macOS WebKit: deltaY is exactly this many
 *  pixels (wheelDeltaY ±12) on a slow click, larger values only with wheel
 *  acceleration. Measured on the user's mouse 2026-09-17: every slow notch
 *  read 4.000244140625; fast spins 17–310 px. Accumulating the 4 px against
 *  WHEEL_PX_PER_STEP needed ten slow clicks per step. */
const MAC_NOTCH_PX = 4.000244140625;

export interface WheelAccumulator {
  acc: number;
  lastTs: number;
}

export function newWheelAccumulator(): WheelAccumulator {
  return { acc: 0, lastTs: -Infinity };
}

export interface WheelLike {
  deltaY: number;
  deltaMode: number;
  /** Legacy WebKit property: ±120 per notch on a notched mouse wheel. */
  wheelDeltaY?: number;
}

/** Whole steps this event is worth; positive = scroll up = increase.
 *  Notched wheels and line-mode deltas give one step per notch/line batch;
 *  pixel deltas accumulate until they cover WHEEL_PX_PER_STEP. */
export function wheelSteps(e: WheelLike, st: WheelAccumulator, now: number): number {
  if (now - st.lastTs > WHEEL_GESTURE_IDLE_MS) st.acc = 0;
  st.lastTs = now;
  if (e.deltaY === 0) return 0;
  const up = e.deltaY < 0 ? 1 : -1;

  const legacy = e.wheelDeltaY ?? 0;
  if (Math.abs(Math.abs(e.deltaY) - MAC_NOTCH_PX) < 1e-3) {
    st.acc = 0;
    return up;
  }
  if (e.deltaMode === 1 || e.deltaMode === 2 || (legacy !== 0 && legacy % NOTCH === 0)) {
    st.acc = 0;
    const notches = legacy !== 0 && legacy % NOTCH === 0 ? Math.abs(legacy) / NOTCH : 1;
    return up * notches;
  }

  // Direction change drops what was left over from the other way.
  if (st.acc !== 0 && Math.sign(st.acc) !== up) st.acc = 0;
  st.acc += -e.deltaY;
  const steps = Math.trunc(st.acc / WHEEL_PX_PER_STEP) || 0; // no -0
  st.acc -= steps * WHEEL_PX_PER_STEP;
  return steps;
}

export interface FieldWheelHandlers {
  /** Apply `steps` (signed); `coarse` = Shift held. */
  onSteps: (steps: number, coarse: boolean) => void;
  /** Gesture ended (idle) — commit / rebuild here. */
  onEnd?: () => void;
  /** Defaults to "the element (or something inside it) has focus": the wheel
   *  adjusts a field only after it was clicked, otherwise the panel scrolls. */
  isActive?: () => boolean;
}

/** Attach the shared wheel behaviour to an element. Non-passive, so an
 *  active field can stop the panel from scrolling; an inactive one leaves
 *  the event alone. */
export function attachFieldWheel(el: HTMLElement, h: FieldWheelHandlers): void {
  const st = newWheelAccumulator();
  let endTimer: ReturnType<typeof setTimeout> | undefined;
  const active = h.isActive ?? (() => el.contains(document.activeElement));
  el.addEventListener("wheel", (e: WheelEvent) => {
    if (!active()) return;
    e.preventDefault();
    e.stopPropagation();
    const legacy = (e as WheelEvent & { wheelDeltaY?: number }).wheelDeltaY;
    const steps = wheelSteps({ deltaY: e.deltaY, deltaMode: e.deltaMode, wheelDeltaY: legacy }, st, performance.now());
    if (steps !== 0) h.onSteps(steps, e.shiftKey);
    if (h.onEnd) {
      if (endTimer) clearTimeout(endTimer);
      endTimer = setTimeout(() => { endTimer = undefined; h.onEnd!(); }, WHEEL_GESTURE_IDLE_MS);
    }
  }, { passive: false });
}

/** Call `fn` at most every `ms` while values keep coming, and once more with
 *  the last value when they stop. */
export function throttleTrailing<T>(fn: (v: T) => void, ms: number): (v: T) => void {
  let last = -Infinity;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let pending: { v: T } | null = null;
  return (v: T) => {
    const now = performance.now();
    if (now - last >= ms) {
      last = now;
      pending = null;
      fn(v);
      return;
    }
    pending = { v };
    if (!timer) {
      timer = setTimeout(() => {
        timer = undefined;
        if (pending) { last = performance.now(); const p = pending; pending = null; fn(p.v); }
      }, ms - (now - last));
    }
  };
}

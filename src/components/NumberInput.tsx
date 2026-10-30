import { createEffect, onCleanup } from "solid-js";
import { attachFieldWheel, throttleTrailing } from "../lib/wheel-step";

interface NumberInputProps {
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
  unit?: string;
  precision?: number;
  freqMode?: boolean; // adaptive step for frequency values
  /** Accessible name of the text field (b141.105: labels were not tied to fields). */
  label?: string;
  id?: string;
}

/**
 * Imperative number input with ±buttons, wheel, and click-to-edit.
 *
 * Uses ONLY direct DOM manipulation (no SolidJS signals for display).
 * This avoids SolidJS reactive context issues with native event handlers.
 * Same proven pattern as PeqSidebar wheelNumber.
 */
export default function NumberInput(props: NumberInputProps) {
  let repeatTimer: number | undefined;
  let repeatInterval: number | undefined;
  let inputEl!: HTMLInputElement;

  // Current value — plain mutable variable
  let val = props.value;

  const prec = () => {
    if (props.precision !== undefined) return props.precision;
    const s = props.step.toString();
    const d = s.indexOf(".");
    return d >= 0 ? s.length - d - 1 : 0;
  };

  const fmt = (v: number) => v.toFixed(prec());
  const clamp = (v: number) => Math.min(props.max, Math.max(props.min, v));

  const step = (dir: number) => {
    if (props.freqMode) {
      if (val < 100) return 1;
      if (val < 1000) return 10;
      return 100;
    }
    return props.step;
  };

  // Push value to DOM + notify parent. b141.105: only a REAL change reaches
  // the parent — every blur/Enter/Escape used to call onChange with the same
  // value: the project went «изменён», an empty undo step appeared, the band
  // re-evaluated, and an HP/LP edit switched a disabled target back on.
  const push = (v: number) => {
    const next = parseFloat(v.toFixed(prec()));
    inputEl.value = fmt(next);
    // b141.118: a direct edit supersedes the wheel's trailing value — it
    // used to land 120 ms later and overwrite the newer one. If the parent
    // has not seen the wheel's last value yet, an unchanged push still sends it.
    const wheelPending = notifyThrottled.cancel();
    if (next === val) { if (wheelPending) props.onChange(val); return; }
    val = next;
    props.onChange(val);
  };
  // b141.43: wheel gestures update the field every step but the parent (a
  // filter frequency, a tolerance…) at most every 120 ms plus the final
  // value — each onChange can start a full re-evaluation.
  const notifyThrottled = throttleTrailing((v: number) => props.onChange(v), 120);
  // b141.118: never call onChange after unmount — the handler may already
  // point at another band.
  onCleanup(() => notifyThrottled.cancel());

  const inc = (dir: number) => {
    const s = step(dir);
    push(clamp(Math.round((val + dir * s) / s) * s));
  };

  // Sync from parent → local + DOM
  createEffect(() => {
    val = props.value;
    // Don't overwrite while user is typing
    if (inputEl && inputEl !== document.activeElement) {
      inputEl.value = fmt(val);
    }
  });

  // ± buttons with repeat
  let repDir = 0;
  const startRepeat = (dir: number) => {
    repDir = dir;
    inc(dir);
    repeatTimer = window.setTimeout(() => {
      repeatInterval = window.setInterval(() => inc(repDir), 60);
    }, 350);
  };
  const stopRepeat = () => {
    clearTimeout(repeatTimer); repeatTimer = undefined;
    clearInterval(repeatInterval); repeatInterval = undefined;
  };
  onCleanup(stopRepeat);

  // Commit text edit. b141.38: a value outside min/max used to be clamped
  // silently — the field flashes so the substitution is seen.
  let rootEl!: HTMLDivElement;
  const flashClamped = () => {
    rootEl.classList.remove("num-input-clamped");
    void rootEl.offsetWidth; // restart the animation on repeated clamps
    rootEl.classList.add("num-input-clamped");
  };
  const commit = () => {
    // Untouched text (focus + blur, Enter on the shown value) is no edit.
    if (inputEl.value === fmt(val)) return;
    const n = parseFloat(inputEl.value);
    if (isNaN(n)) { inputEl.value = fmt(val); return; }
    const c = clamp(n);
    push(c);
    if (c !== n) flashClamped();
  };

  return (
    <div
      class="num-input"
      ref={(el: HTMLDivElement) => {
        rootEl = el;
        el.addEventListener("animationend", () => el.classList.remove("num-input-clamped"));
        // b141.43: shared wheel model (lib/wheel-step.ts) — active only while
        // the field has focus, trackpad deltas accumulate into steps.
        attachFieldWheel(el, {
          onSteps: (steps, coarse) => {
            const dir = Math.sign(steps);
            const before = val;
            for (let k = 0; k < Math.abs(steps); k++) {
              val = parseFloat(clamp(val + dir * step(dir) * (coarse ? 10 : 1)).toFixed(prec()));
            }
            inputEl.value = fmt(val);
            if (val !== before) notifyThrottled(val); // at a limit: nothing changed
          },
        });
      }}
    >
      <button
        class="num-btn num-btn-dec"
        onMouseDown={() => startRepeat(-1)}
        onMouseUp={stopRepeat}
        onMouseLeave={stopRepeat}
        tabIndex={-1}
      >−</button>
      <input
        class="num-field"
        type="text"
        ref={(el: HTMLInputElement) => { inputEl = el; el.value = fmt(val); }}
        id={props.id}
        aria-label={props.label}
        onFocus={(e) => e.currentTarget.select()}
        onBlur={() => { commit(); notifyThrottled.flush(); }}
        onKeyDown={(e) => {
          if (e.key === "Enter") { e.preventDefault(); commit(); e.currentTarget.blur(); }
          // b141.105: Escape cancels the draft and stops here — it used to
          // bubble to the dialog overlay and close the whole dialog.
          if (e.key === "Escape") {
            e.preventDefault();
            e.stopPropagation();
            inputEl.value = fmt(val);
            e.currentTarget.blur();
          }
          // b141.38: arrows step like the ± buttons (Shift ×10), which are
          // mouse-only by design (tabIndex -1).
          if (e.key === "ArrowUp" || e.key === "ArrowDown") {
            e.preventDefault();
            commit();
            const dir = e.key === "ArrowUp" ? 1 : -1;
            push(clamp(val + dir * step(dir) * (e.shiftKey ? 10 : 1)));
          }
        }}
        tabIndex={0}
      />
      <button
        class="num-btn num-btn-inc"
        onMouseDown={() => startRepeat(1)}
        onMouseUp={stopRepeat}
        onMouseLeave={stopRepeat}
        tabIndex={-1}
      >+</button>
      {props.unit && <span class="num-unit">{props.unit}</span>}
    </div>
  );
}

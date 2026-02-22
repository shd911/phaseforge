import { createEffect, onCleanup } from "solid-js";

interface NumberInputProps {
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
  unit?: string;
  precision?: number;
  freqMode?: boolean; // adaptive step for frequency values
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

  // Push value to DOM + notify parent
  const push = (v: number) => {
    val = parseFloat(v.toFixed(prec()));
    inputEl.value = fmt(val);
    props.onChange(val);
  };

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

  // Commit text edit
  const commit = () => {
    const n = parseFloat(inputEl.value);
    if (!isNaN(n)) push(clamp(n));
    else inputEl.value = fmt(val);
  };

  return (
    <div
      class="num-input"
      ref={(el: HTMLDivElement) => {
        // Non-passive wheel handler — identical pattern to PeqSidebar wheelNumber
        el.addEventListener("wheel", (e: WheelEvent) => {
          e.preventDefault();
          e.stopPropagation();
          const dir = e.deltaY < 0 ? 1 : -1;
          const mult = e.shiftKey ? 10 : 1;
          const s = step(dir) * mult;
          push(clamp(val + dir * s));
        }, { passive: false });
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
        onFocus={(e) => e.currentTarget.select()}
        onBlur={() => commit()}
        onKeyDown={(e) => {
          if (e.key === "Enter") { e.preventDefault(); commit(); e.currentTarget.blur(); }
          if (e.key === "Escape") { inputEl.value = fmt(val); e.currentTarget.blur(); }
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

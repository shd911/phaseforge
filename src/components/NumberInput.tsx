import { createSignal, createEffect, onCleanup } from "solid-js";

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

export default function NumberInput(props: NumberInputProps) {
  const [editing, setEditing] = createSignal(false);
  const [editText, setEditText] = createSignal("");
  // Local display value — tracks props.value reactively AND updates from wheel
  const [localVal, setLocalVal] = createSignal(props.value);
  let repeatTimer: number | undefined;
  let repeatInterval: number | undefined;
  let containerRef!: HTMLDivElement;

  // Sync from parent → local (reactive)
  createEffect(() => setLocalVal(props.value));

  const precision = () => {
    if (props.precision !== undefined) return props.precision;
    const s = props.step.toString();
    const dot = s.indexOf(".");
    return dot >= 0 ? s.length - dot - 1 : 0;
  };

  const displayValue = () => localVal().toFixed(precision());

  const effectiveStep = (dir: number) => {
    if (props.freqMode) {
      const v = localVal();
      if (v < 100) return 1;
      if (v < 1000) return 10;
      return 100;
    }
    return props.step;
  };

  const clamp = (v: number) => Math.min(props.max, Math.max(props.min, v));

  const applyValue = (newVal: number) => {
    const v = parseFloat(newVal.toFixed(precision()));
    setLocalVal(v);
    props.onChange(v);
  };

  const adjust = (dir: number) => {
    const step = effectiveStep(dir);
    const newVal = clamp(
      Math.round((localVal() + dir * step) / step) * step
    );
    applyValue(newVal);
  };

  const startRepeat = (dir: number) => {
    adjust(dir);
    repeatTimer = window.setTimeout(() => {
      repeatInterval = window.setInterval(() => adjust(dir), 60);
    }, 350);
  };

  const stopRepeat = () => {
    if (repeatTimer !== undefined) {
      clearTimeout(repeatTimer);
      repeatTimer = undefined;
    }
    if (repeatInterval !== undefined) {
      clearInterval(repeatInterval);
      repeatInterval = undefined;
    }
  };

  onCleanup(stopRepeat);

  const commitEdit = () => {
    const parsed = parseFloat(editText());
    if (!isNaN(parsed)) {
      applyValue(clamp(parsed));
    }
    setEditing(false);
  };

  const handleWheel = (e: WheelEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const dir = e.deltaY < 0 ? 1 : -1;
    const mult = e.shiftKey ? 10 : 1;
    const step = effectiveStep(dir) * mult;
    const newVal = clamp(localVal() + dir * step);
    applyValue(newVal);
  };

  // Attach wheel via ref callback with { passive: false }.
  // SolidJS JSX onWheel uses passive listeners → preventDefault() is ignored.
  const bindWheel = (el: HTMLDivElement) => {
    containerRef = el;
    el.addEventListener("wheel", handleWheel, { passive: false });
  };

  return (
    <div class="num-input" ref={bindWheel}>
      <button
        class="num-btn num-btn-dec"
        onMouseDown={() => startRepeat(-1)}
        onMouseUp={stopRepeat}
        onMouseLeave={stopRepeat}
        tabIndex={-1}
      >−</button>
      {editing() ? (
        <input
          class="num-field"
          type="text"
          value={editText()}
          onInput={(e) => setEditText(e.currentTarget.value)}
          onBlur={() => { commitEdit(); }}
          onKeyDown={(e) => {
            if (e.key === "Enter") { e.preventDefault(); commitEdit(); }
            if (e.key === "Escape") { setEditing(false); }
            if (e.key === "Tab") { commitEdit(); }
          }}
          ref={(el) => requestAnimationFrame(() => { el.focus(); el.select(); })}
        />
      ) : (
        <span
          class="num-field num-display"
          onClick={() => {
            setEditText(displayValue());
            setEditing(true);
          }}
        >
          {displayValue()}
        </span>
      )}
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

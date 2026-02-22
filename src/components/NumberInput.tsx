import { createSignal, onCleanup, onMount } from "solid-js";

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
  let repeatTimer: number | undefined;
  let repeatInterval: number | undefined;
  let containerRef!: HTMLDivElement;

  const precision = () => {
    if (props.precision !== undefined) return props.precision;
    const s = props.step.toString();
    const dot = s.indexOf(".");
    return dot >= 0 ? s.length - dot - 1 : 0;
  };

  const displayValue = () => props.value.toFixed(precision());

  const effectiveStep = (dir: number) => {
    if (props.freqMode) {
      const v = props.value;
      if (v < 100) return 1;
      if (v < 1000) return 10;
      return 100;
    }
    return props.step;
  };

  const clamp = (v: number) => Math.min(props.max, Math.max(props.min, v));

  const adjust = (dir: number) => {
    const step = effectiveStep(dir);
    const newVal = clamp(
      Math.round((props.value + dir * step) / step) * step
    );
    props.onChange(parseFloat(newVal.toFixed(precision())));
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
      props.onChange(clamp(parseFloat(parsed.toFixed(precision()))));
    }
    setEditing(false);
  };

  const handleWheel = (e: WheelEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const dir = e.deltaY < 0 ? 1 : -1;
    const mult = e.shiftKey ? 10 : 1;
    const step = effectiveStep(dir) * mult;
    const newVal = clamp(props.value + dir * step);
    props.onChange(parseFloat(newVal.toFixed(precision())));
  };

  // Register wheel handler with { passive: false } so preventDefault() works.
  // SolidJS onWheel in JSX uses passive listeners by default, which
  // silently ignores preventDefault() → scroll bleeds through to parent.
  onMount(() => {
    containerRef.addEventListener("wheel", handleWheel, { passive: false });
    onCleanup(() => containerRef.removeEventListener("wheel", handleWheel));
  });

  return (
    <div class="num-input" ref={containerRef}>
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

// b141.105 (audit 2026-10-23 §5.8/5.26): the numeric field of a dialog.
//
// UI convention (b141.38): parse on input, clamp on blur/Apply. What was
// missing is the third half — an empty or garbage draft left the field
// showing one thing while the dialog kept (and saved) another. On blur the
// field now always shows the model: the clamped value, or the previous one
// when the draft does not parse.
import { createEffect } from "solid-js";
import { clampNum } from "../lib/slope";

export interface NumFieldProps {
  value: number;
  /** Draft values while typing (unclamped), the clamped value on blur. */
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number | string;
  integer?: boolean;
  /** Report an unparsable draft as NaN (the dialog then disables Apply) —
   *  otherwise the previous value stays in the model. */
  invalidAsNaN?: boolean;
  class?: string;
  id?: string;
  label?: string;
  autofocus?: boolean;
}

export default function NumField(props: NumFieldProps) {
  let el!: HTMLInputElement;
  const parse = (s: string): number | null => {
    const n = props.integer ? parseInt(s, 10) : parseFloat(s);
    return Number.isFinite(n) ? n : null;
  };
  const show = (v: number) => { if (el && Number.isFinite(v)) el.value = String(v); };

  // Model → field while the user is not typing in it.
  createEffect(() => {
    const v = props.value;
    if (el && el !== document.activeElement) show(v);
  });

  const commit = () => {
    const n = parse(el.value);
    const base = n ?? (Number.isFinite(props.value) ? props.value : null);
    if (base == null) return; // nothing valid yet: the dialog shows why
    const c = clampNum(base, props.min, props.max);
    if (c !== props.value) props.onChange(c);
    show(c);
  };

  return (
    <input
      ref={(e) => {
        el = e;
        show(props.value);
        if (props.autofocus) requestAnimationFrame(() => { e.focus(); e.select(); });
      }}
      class={props.class ?? "xo-input"}
      type="number"
      id={props.id}
      aria-label={props.label}
      min={props.min}
      max={props.max}
      step={props.step ?? "any"}
      onInput={(e) => {
        const n = parse(e.currentTarget.value);
        if (n != null) props.onChange(n);
        else if (props.invalidAsNaN) props.onChange(NaN);
      }}
      onBlur={commit}
      onKeyDown={(e) => { if (e.key === "Enter") commit(); }}
    />
  );
}

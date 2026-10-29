// b141.105 (audit 2026-10-23 §5.1): NumberInput reports only real changes;
// Escape cancels the draft and does not reach the dialog behind the field.
import { describe, it, expect, vi, afterEach } from "vitest";
import { render } from "solid-js/web";
import NumberInput from "../NumberInput";
import { WHEEL_PX_PER_STEP } from "../../lib/wheel-step";

let dispose: (() => void) | undefined;
afterEach(() => { dispose?.(); dispose = undefined; document.body.innerHTML = ""; });

function mount(onChange: (v: number) => void, onOuterKey?: (e: KeyboardEvent) => void) {
  const host = document.createElement("div");
  document.body.appendChild(host);
  dispose = render(() => (
    <div onKeyDown={(e) => onOuterKey?.(e)}>
      <NumberInput value={80} onChange={onChange} min={10} max={1000} step={1} label="Частота" />
    </div>
  ), host);
  return host.querySelector("input.num-field") as HTMLInputElement;
}

const key = (el: HTMLElement, k: string) =>
  el.dispatchEvent(new KeyboardEvent("keydown", { key: k, bubbles: true, cancelable: true }));

describe("NumberInput", () => {
  it("focus + blur without editing does not call onChange", () => {
    const onChange = vi.fn();
    const input = mount(onChange);
    input.focus(); input.blur();
    key(input, "Enter");
    expect(onChange).not.toHaveBeenCalled();
  });

  it("an edit is reported once", () => {
    const onChange = vi.fn();
    const input = mount(onChange);
    input.focus();
    input.value = "120";
    key(input, "Enter");
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange).toHaveBeenCalledWith(120);
  });

  it("the same number typed differently is no change", () => {
    const onChange = vi.fn();
    const input = mount(onChange);
    input.focus();
    input.value = "80.0";
    key(input, "Enter");
    expect(onChange).not.toHaveBeenCalled();
    expect(input.value).toBe("80");
  });

  it("Escape restores the value and does not bubble", () => {
    const onChange = vi.fn();
    const outer = vi.fn();
    const input = mount(onChange, outer);
    input.focus();
    input.value = "500";
    key(input, "Escape");
    expect(input.value).toBe("80");
    expect(onChange).not.toHaveBeenCalled();
    expect(outer).not.toHaveBeenCalled();
  });

  it("arrow at the limit does not call onChange", () => {
    const onChange = vi.fn();
    const host = document.createElement("div");
    document.body.appendChild(host);
    dispose = render(() => <NumberInput value={1000} onChange={onChange} min={10} max={1000} step={1} />, host);
    const input = host.querySelector("input.num-field") as HTMLInputElement;
    input.focus();
    key(input, "ArrowUp");
    expect(onChange).not.toHaveBeenCalled();
  });

  // b141.118 (external audit 2026-10-29 §1): the wheel's trailing value
  // landed 120 ms after a newer direct edit and overwrote it.
  it("a direct edit is not overwritten by the wheel's trailing value", () => {
    vi.useFakeTimers();
    const now = vi.spyOn(performance, "now");
    let t = 1000; now.mockImplementation(() => t);
    const onChange = vi.fn();
    const input = mount(onChange);
    input.focus();
    const field = input.closest(".num-input")!;
    const wheel = () => field.dispatchEvent(new WheelEvent("wheel", { deltaY: -WHEEL_PX_PER_STEP, bubbles: true, cancelable: true }));
    wheel(); t += 10; wheel(); t += 10;          // 81 now, 82 pending
    key(input, "ArrowUp");                        // 83 directly
    t += 200; vi.advanceTimersByTime(200);
    expect(onChange.mock.calls.map((c) => c[0])).toEqual([81, 83]);
    now.mockRestore(); vi.useRealTimers();
  });

  it("no wheel value reaches the parent after unmount", () => {
    vi.useFakeTimers();
    const now = vi.spyOn(performance, "now");
    let t = 1000; now.mockImplementation(() => t);
    const onChange = vi.fn();
    const input = mount(onChange);
    input.focus();
    const field = input.closest(".num-input")!;
    const wheel = () => field.dispatchEvent(new WheelEvent("wheel", { deltaY: -WHEEL_PX_PER_STEP, bubbles: true, cancelable: true }));
    wheel(); t += 10; wheel();
    dispose?.(); dispose = undefined;
    t += 200; vi.advanceTimersByTime(200);
    expect(onChange.mock.calls.map((c) => c[0])).toEqual([81]);
    now.mockRestore(); vi.useRealTimers();
  });

  it("the field carries its accessible name", () => {
    const input = mount(vi.fn());
    expect(input.getAttribute("aria-label")).toBe("Частота");
  });
});

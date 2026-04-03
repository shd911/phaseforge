/**
 * UI-level test: render two FilterBlock components (HP + LP) and verify
 * that clicking Lin-φ on HP does NOT affect LP's state or vice versa.
 *
 * This test renders actual SolidJS components in jsdom to catch
 * reactivity/event issues that store-level tests miss.
 */
import { describe, it, expect, vi } from "vitest";
import { createRoot, createSignal } from "solid-js";
import { createStore, reconcile } from "solid-js/store";
import type { FilterConfig } from "../lib/types";

// Minimal FilterBlock that reproduces the real component's pattern
function TestFilterBlock(props: {
  title: string;
  config: FilterConfig | null;
  onChange: (c: FilterConfig) => void;
}) {
  const c = () => props.config;

  return (
    <div data-testid={`filter-${props.title}`}>
      {c() && (
        <>
          <span data-testid={`${props.title}-type`}>{c()!.filter_type}</span>
          <span data-testid={`${props.title}-linphase`}>{String(c()!.linear_phase)}</span>
          <button
            data-testid={`${props.title}-toggle`}
            onClick={() => props.onChange({ ...c()!, linear_phase: !c()!.linear_phase })}
          >
            Toggle Lin-φ
          </button>
          <select
            data-testid={`${props.title}-select`}
            value={c()!.filter_type}
            onChange={(e) => {
              const ft = e.currentTarget.value;
              props.onChange({ ...c()!, filter_type: ft as any });
            }}
          >
            <option value="Gaussian">Gaussian</option>
            <option value="Butterworth">Butterworth</option>
          </select>
        </>
      )}
    </div>
  );
}

function makeFilter(type: string, freq: number, linPhase: boolean): FilterConfig {
  return {
    filter_type: type as any,
    order: 4,
    freq_hz: freq,
    shape: type === "Gaussian" ? 1.0 : null,
    linear_phase: linPhase,
    q: null,
  };
}

// --- Test 1: Direct spread of SolidJS store proxy ---
describe("FilterBlock HP/LP isolation with SolidJS store", () => {
  it("spread of store proxy: HP change does NOT affect LP", () => {
    createRoot((dispose) => {
      const [store, setStore] = createStore({
        target: {
          high_pass: makeFilter("Gaussian", 100, true) as FilterConfig | null,
          low_pass: makeFilter("Gaussian", 800, true) as FilterConfig | null,
        },
      });

      // Simulate what FilterBlock does: spread proxy + override
      const hpProxy = store.target.high_pass!;
      const patched = { ...hpProxy, linear_phase: false };
      setStore("target", "high_pass", patched);

      expect(store.target.high_pass?.linear_phase).toBe(false);
      expect(store.target.low_pass?.linear_phase).toBe(true);
      dispose();
    });
  });

  it("spread of store proxy: LP change does NOT affect HP", () => {
    createRoot((dispose) => {
      const [store, setStore] = createStore({
        target: {
          high_pass: makeFilter("Gaussian", 100, true) as FilterConfig | null,
          low_pass: makeFilter("Gaussian", 800, true) as FilterConfig | null,
        },
      });

      const lpProxy = store.target.low_pass!;
      const patched = { ...lpProxy, linear_phase: false };
      setStore("target", "low_pass", patched);

      expect(store.target.low_pass?.linear_phase).toBe(false);
      expect(store.target.high_pass?.linear_phase).toBe(true);
      dispose();
    });
  });

  it("SHARED object: SolidJS store does NOT deep-copy (this is the bug)", () => {
    // This test documents the SolidJS behavior that causes the bug:
    // when the same object reference is used for both HP and LP,
    // changing one changes the other.
    createRoot((dispose) => {
      const shared = makeFilter("Gaussian", 500, true);
      const [store, setStore] = createStore({
        target: {
          high_pass: shared as FilterConfig | null,
          low_pass: shared as FilterConfig | null, // SAME reference!
        },
      });

      // Change HP
      setStore("target", "high_pass", { ...store.target.high_pass!, linear_phase: false });

      expect(store.target.high_pass?.linear_phase).toBe(false);
      // SolidJS store DOES cross-contaminate with shared refs!
      // This proves the bug. The fix is to always deep-clone before setState.
      expect(store.target.low_pass?.linear_phase).toBe(false); // BUG: both change
      dispose();
    });
  });

  it("CLONED objects: HP/LP isolation works when configs are separate objects", () => {
    createRoot((dispose) => {
      const hp = makeFilter("Gaussian", 500, true);
      const lp = makeFilter("Gaussian", 500, true); // separate object, same values
      const [store, setStore] = createStore({
        target: {
          high_pass: hp as FilterConfig | null,
          low_pass: lp as FilterConfig | null,
        },
      });

      // Change HP
      setStore("target", "high_pass", { ...store.target.high_pass!, linear_phase: false });

      expect(store.target.high_pass?.linear_phase).toBe(false);
      expect(store.target.low_pass?.linear_phase).toBe(true); // ISOLATED
      dispose();
    });
  });
});

// --- Test 2: Rendered FilterBlock components with onClick ---
describe("Rendered FilterBlock interaction", () => {
  it("clicking HP toggle does not change LP state", () => {
    createRoot((dispose) => {
      const [store, setStore] = createStore({
        target: {
          high_pass: makeFilter("Gaussian", 100, true) as FilterConfig | null,
          low_pass: makeFilter("Gaussian", 800, true) as FilterConfig | null,
        },
      });

      const hpChanges: FilterConfig[] = [];
      const lpChanges: FilterConfig[] = [];

      const el = (
        <div>
          <TestFilterBlock
            title="HP"
            config={store.target.high_pass}
            onChange={(c) => {
              hpChanges.push(c);
              setStore("target", "high_pass", c);
            }}
          />
          <TestFilterBlock
            title="LP"
            config={store.target.low_pass}
            onChange={(c) => {
              lpChanges.push(c);
              setStore("target", "low_pass", c);
            }}
          />
        </div>
      ) as HTMLDivElement;

      // Find HP toggle button and click it
      const hpBtn = el.querySelector('[data-testid="HP-toggle"]') as HTMLButtonElement;
      expect(hpBtn).toBeTruthy();
      hpBtn.click();

      // HP should have changed
      expect(hpChanges.length).toBe(1);
      expect(hpChanges[0].linear_phase).toBe(false);
      expect(store.target.high_pass?.linear_phase).toBe(false);

      // LP should NOT have changed
      expect(lpChanges.length).toBe(0);
      expect(store.target.low_pass?.linear_phase).toBe(true);

      // Verify DOM reflects correct state
      const hpText = el.querySelector('[data-testid="HP-linphase"]')?.textContent;
      const lpText = el.querySelector('[data-testid="LP-linphase"]')?.textContent;
      expect(hpText).toBe("false");
      expect(lpText).toBe("true");

      dispose();
    });
  });

  it("clicking LP toggle does not change HP state", () => {
    createRoot((dispose) => {
      const [store, setStore] = createStore({
        target: {
          high_pass: makeFilter("Gaussian", 100, true) as FilterConfig | null,
          low_pass: makeFilter("Gaussian", 800, true) as FilterConfig | null,
        },
      });

      const hpChanges: FilterConfig[] = [];
      const lpChanges: FilterConfig[] = [];

      const el = (
        <div>
          <TestFilterBlock
            title="HP"
            config={store.target.high_pass}
            onChange={(c) => {
              hpChanges.push(c);
              setStore("target", "high_pass", c);
            }}
          />
          <TestFilterBlock
            title="LP"
            config={store.target.low_pass}
            onChange={(c) => {
              lpChanges.push(c);
              setStore("target", "low_pass", c);
            }}
          />
        </div>
      ) as HTMLDivElement;

      const lpBtn = el.querySelector('[data-testid="LP-toggle"]') as HTMLButtonElement;
      lpBtn.click();

      expect(lpChanges.length).toBe(1);
      expect(lpChanges[0].linear_phase).toBe(false);
      expect(store.target.low_pass?.linear_phase).toBe(false);

      expect(hpChanges.length).toBe(0);
      expect(store.target.high_pass?.linear_phase).toBe(true);

      dispose();
    });
  });

  it("select onChange on HP does NOT trigger LP onChange", () => {
    createRoot((dispose) => {
      const [store, setStore] = createStore({
        target: {
          high_pass: makeFilter("Gaussian", 100, true) as FilterConfig | null,
          low_pass: makeFilter("Gaussian", 800, true) as FilterConfig | null,
        },
      });

      const hpChanges: FilterConfig[] = [];
      const lpChanges: FilterConfig[] = [];

      const el = (
        <div>
          <TestFilterBlock
            title="HP"
            config={store.target.high_pass}
            onChange={(c) => {
              hpChanges.push(c);
              setStore("target", "high_pass", c);
            }}
          />
          <TestFilterBlock
            title="LP"
            config={store.target.low_pass}
            onChange={(c) => {
              lpChanges.push(c);
              setStore("target", "low_pass", c);
            }}
          />
        </div>
      ) as HTMLDivElement;

      // Simulate changing HP select to Butterworth
      const hpSelect = el.querySelector('[data-testid="HP-select"]') as HTMLSelectElement;
      hpSelect.value = "Butterworth";
      hpSelect.dispatchEvent(new Event("change", { bubbles: true }));

      // HP should change
      expect(hpChanges.length).toBeGreaterThanOrEqual(1);
      expect(store.target.high_pass?.filter_type).toBe("Butterworth");

      // LP must NOT change
      expect(lpChanges.length).toBe(0);
      expect(store.target.low_pass?.filter_type).toBe("Gaussian");

      dispose();
    });
  });
});

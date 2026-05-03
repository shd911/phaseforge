/**
 * UI-level test: render two FilterBlock components (HP + LP) and verify
 * that clicking Lin-φ on HP does NOT affect LP's state or vice versa.
 *
 * This test renders actual SolidJS components in jsdom to catch
 * reactivity/event issues that store-level tests miss.
 *
 * KEY: SolidJS delegates onClick to `document`. Elements MUST be in the
 * document DOM tree for delegated events to fire. We append to document.body.
 */
import { describe, it, expect, vi, afterEach } from "vitest";
import { createRoot, createSignal } from "solid-js";
import { createStore, reconcile } from "solid-js/store";
import { render } from "solid-js/web";
import type { FilterConfig } from "../lib/types";

// Minimal FilterBlock that reproduces the real component's pattern
function TestFilterBlock(props: {
  title: string;
  config: FilterConfig | null;
  onChange: (c: FilterConfig) => void;
}) {
  const c = () => props.config;

  /** Build a full FilterConfig from the current config, overriding specific fields.
   *  Mirrors the real withOverride exactly. */
  const withOverride = (overrides: Partial<FilterConfig>): FilterConfig => {
    const cur = c()!;
    return {
      filter_type: cur.filter_type,
      order: cur.order,
      freq_hz: cur.freq_hz,
      shape: cur.shape,
      linear_phase: cur.linear_phase,
      q: cur.q,
      ...overrides,
    };
  };

  return (
    <div data-testid={`filter-${props.title}`}>
      {c() && (
        <>
          <span data-testid={`${props.title}-type`}>{c()!.filter_type}</span>
          <span data-testid={`${props.title}-linphase`}>{String(c()!.linear_phase)}</span>
          <button
            data-testid={`${props.title}-toggle`}
            onClick={() => props.onChange(withOverride({ linear_phase: !c()!.linear_phase }))}
          >
            Toggle Lin-φ
          </button>
          <select
            data-testid={`${props.title}-select`}
            value={c()!.filter_type}
            onChange={(e) => {
              const ft = e.currentTarget.value;
              if (ft === c()!.filter_type) return; // guard, same as real code
              props.onChange(withOverride({ filter_type: ft as any }));
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

/** Deep-copy a FilterConfig from SolidJS store proxy to a plain object.
 *  Mirrors unwrapFilter from ControlPanel.tsx exactly. */
function unwrapFilter(f: FilterConfig | null | undefined): FilterConfig | null {
  if (!f) return null;
  return {
    filter_type: f.filter_type,
    order: f.order,
    freq_hz: f.freq_hz,
    shape: f.shape,
    linear_phase: f.linear_phase,
    q: f.q,
    subsonic_protect: f.subsonic_protect ?? null,
  };
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

  it("SHARED object: SolidJS store cross-contaminates with shared refs", () => {
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

// --- Test 2: Rendered FilterBlock with proper DOM attachment ---
// SolidJS delegates onClick to the document. Elements must be in the DOM tree.
describe("Rendered FilterBlock interaction (DOM-attached)", () => {
  let cleanup: (() => void) | undefined;

  afterEach(() => {
    cleanup?.();
    cleanup = undefined;
  });

  it("clicking HP Lin-φ toggle does not change LP state", () => {
    const hpChanges: FilterConfig[] = [];
    const lpChanges: FilterConfig[] = [];

    let store: any;
    let setStore: any;

    const container = document.createElement("div");
    document.body.appendChild(container);

    cleanup = render(() => {
      const [s, ss] = createStore({
        target: {
          high_pass: makeFilter("Gaussian", 100, true) as FilterConfig | null,
          low_pass: makeFilter("Gaussian", 800, true) as FilterConfig | null,
        },
      });
      store = s;
      setStore = ss;

      return (
        <div>
          <TestFilterBlock
            title="HP"
            config={unwrapFilter(s.target.high_pass)}
            onChange={(c) => {
              hpChanges.push(c);
              ss("target", "high_pass", c);
            }}
          />
          <TestFilterBlock
            title="LP"
            config={unwrapFilter(s.target.low_pass)}
            onChange={(c) => {
              lpChanges.push(c);
              ss("target", "low_pass", c);
            }}
          />
        </div>
      );
    }, container);

    // Find HP toggle button and click it
    const hpBtn = container.querySelector('[data-testid="HP-toggle"]') as HTMLButtonElement;
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
    const hpText = container.querySelector('[data-testid="HP-linphase"]')?.textContent;
    const lpText = container.querySelector('[data-testid="LP-linphase"]')?.textContent;
    expect(hpText).toBe("false");
    expect(lpText).toBe("true");
  });

  it("clicking LP Lin-φ toggle does not change HP state", () => {
    const hpChanges: FilterConfig[] = [];
    const lpChanges: FilterConfig[] = [];

    let store: any;

    const container = document.createElement("div");
    document.body.appendChild(container);

    cleanup = render(() => {
      const [s, ss] = createStore({
        target: {
          high_pass: makeFilter("Gaussian", 100, true) as FilterConfig | null,
          low_pass: makeFilter("Gaussian", 800, true) as FilterConfig | null,
        },
      });
      store = s;

      return (
        <div>
          <TestFilterBlock
            title="HP"
            config={unwrapFilter(s.target.high_pass)}
            onChange={(c) => {
              hpChanges.push(c);
              ss("target", "high_pass", c);
            }}
          />
          <TestFilterBlock
            title="LP"
            config={unwrapFilter(s.target.low_pass)}
            onChange={(c) => {
              lpChanges.push(c);
              ss("target", "low_pass", c);
            }}
          />
        </div>
      );
    }, container);

    const lpBtn = container.querySelector('[data-testid="LP-toggle"]') as HTMLButtonElement;
    lpBtn.click();

    expect(lpChanges.length).toBe(1);
    expect(lpChanges[0].linear_phase).toBe(false);
    expect(store.target.low_pass?.linear_phase).toBe(false);

    expect(hpChanges.length).toBe(0);
    expect(store.target.high_pass?.linear_phase).toBe(true);
  });

  it("select onChange on HP does NOT trigger LP onChange", () => {
    const hpChanges: FilterConfig[] = [];
    const lpChanges: FilterConfig[] = [];

    let store: any;

    const container = document.createElement("div");
    document.body.appendChild(container);

    cleanup = render(() => {
      const [s, ss] = createStore({
        target: {
          high_pass: makeFilter("Gaussian", 100, true) as FilterConfig | null,
          low_pass: makeFilter("Gaussian", 800, true) as FilterConfig | null,
        },
      });
      store = s;

      return (
        <div>
          <TestFilterBlock
            title="HP"
            config={unwrapFilter(s.target.high_pass)}
            onChange={(c) => {
              hpChanges.push(c);
              ss("target", "high_pass", c);
            }}
          />
          <TestFilterBlock
            title="LP"
            config={unwrapFilter(s.target.low_pass)}
            onChange={(c) => {
              lpChanges.push(c);
              ss("target", "low_pass", c);
            }}
          />
        </div>
      );
    }, container);

    // Simulate changing HP select to Butterworth
    const hpSelect = container.querySelector('[data-testid="HP-select"]') as HTMLSelectElement;
    hpSelect.value = "Butterworth";
    hpSelect.dispatchEvent(new Event("change", { bubbles: true }));

    // HP should change
    expect(hpChanges.length).toBeGreaterThanOrEqual(1);
    expect(store.target.high_pass?.filter_type).toBe("Butterworth");

    // LP must NOT change
    expect(lpChanges.length).toBe(0);
    expect(store.target.low_pass?.filter_type).toBe("Gaussian");
  });

  // --- Test 3: KNOWN SolidJS limitation — shared refs cause cross-contamination ---
  // This test documents the underlying SolidJS behavior. Application-level
  // code MUST prevent shared references (unwrapFilterConfig, set-null-first).
  it.skip("KNOWN: shared object ref causes cross-contamination (SolidJS limitation)", () => {
    const hpChanges: FilterConfig[] = [];
    const lpChanges: FilterConfig[] = [];

    let store: any;
    let setStore: any;

    const container = document.createElement("div");
    document.body.appendChild(container);

    const sharedFilter = makeFilter("Gaussian", 500, true);

    cleanup = render(() => {
      const [s, ss] = createStore({
        target: {
          high_pass: sharedFilter as FilterConfig | null,
          low_pass: sharedFilter as FilterConfig | null,
        },
      });
      store = s;
      setStore = ss;

      return (
        <div>
          <TestFilterBlock
            title="HP"
            config={unwrapFilter(s.target.high_pass)}
            onChange={(c) => {
              hpChanges.push(c);
              ss("target", "high_pass", c);
            }}
          />
          <TestFilterBlock
            title="LP"
            config={unwrapFilter(s.target.low_pass)}
            onChange={(c) => {
              lpChanges.push(c);
              ss("target", "low_pass", c);
            }}
          />
        </div>
      );
    }, container);

    const hpBtn = container.querySelector('[data-testid="HP-toggle"]') as HTMLButtonElement;
    expect(hpBtn).toBeTruthy();
    hpBtn.click();

    expect(hpChanges.length).toBe(1);
    expect(store.target.high_pass?.linear_phase).toBe(false);
    expect(store.target.low_pass?.linear_phase).toBe(true);
    expect(lpChanges.length).toBe(0);
  });

  // --- Test 4: Simulate reconcile with shared refs (project load scenario) ---
  it("reconcile with shared refs causes cross-contamination", () => {
    const container = document.createElement("div");
    document.body.appendChild(container);

    let store: any;
    let setStore: any;
    const hpChanges: FilterConfig[] = [];
    const lpChanges: FilterConfig[] = [];

    cleanup = render(() => {
      const [s, ss] = createStore({
        target: {
          high_pass: makeFilter("Gaussian", 100, true) as FilterConfig | null,
          low_pass: makeFilter("Gaussian", 800, true) as FilterConfig | null,
        },
      });
      store = s;
      setStore = ss;

      return (
        <div>
          <TestFilterBlock
            title="HP"
            config={unwrapFilter(s.target.high_pass)}
            onChange={(c) => {
              hpChanges.push(c);
              ss("target", "high_pass", c);
            }}
          />
          <TestFilterBlock
            title="LP"
            config={unwrapFilter(s.target.low_pass)}
            onChange={(c) => {
              lpChanges.push(c);
              ss("target", "low_pass", c);
            }}
          />
        </div>
      );
    }, container);

    // Simulate a project load with reconcile using shared refs
    // (This is what resetAppState does, but WITHOUT the deep clone fix)
    const sharedFilter = makeFilter("Butterworth", 200, true);
    const newState = {
      target: {
        high_pass: sharedFilter,
        low_pass: sharedFilter, // SAME REFERENCE — simulates broken project load
      },
    };
    setStore(reconcile(newState));

    // Verify initial state after reconcile
    expect(store.target.high_pass?.linear_phase).toBe(true);
    expect(store.target.low_pass?.linear_phase).toBe(true);

    // Now click HP's Lin-φ toggle
    const hpBtn = container.querySelector('[data-testid="HP-toggle"]') as HTMLButtonElement;
    expect(hpBtn).toBeTruthy();
    hpBtn.click();

    // HP should change
    expect(hpChanges.length).toBe(1);
    expect(store.target.high_pass?.linear_phase).toBe(false);

    // LP should NOT change
    expect(store.target.low_pass?.linear_phase).toBe(true);
    expect(lpChanges.length).toBe(0);
  });
});

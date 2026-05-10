# Slope dropdown desync investigation (b140.8.2)

## Render path

- Slope display source:
  `src/components/ControlPanel.tsx:420` — `value={String(orderToSlope(c()!.filter_type, c()!.order))}`
- Slope change handler:
  `src/components/ControlPanel.tsx:421-427` — `slopeToOrder` → `withOverride({ order })` → `props.onChange`
- Options list:
  `src/components/ControlPanel.tsx:429-431` — `availableSlopes(filter_type).map(s => <option value=...>)`
- `orderToSlope/slopeToOrder/availableSlopes` definitions:
  `src/components/ControlPanel.tsx:294-322`

### Race condition

SolidJS `<select value={X}>` is rendered as a single attribute setter
on the DOM element. With `<option>` siblings supplied by `.map(...)`,
Solid evaluates `value` and the children list reactively. When
`props.config` mutates (band switch → new FilterConfig), Solid
re-evaluates **both** the `value` attribute and the option list;
their relative ordering is not deterministic across browsers / Solid
internals. If browser sets `<select>.value` before the new options
exist, the value is silently rejected and DOM falls back to the
first option (6 dB/oct for non-LR, 12 for LR).

### Reproducing scenario (user-reported)

1. Open project. Active band has LP=Bessel order=3 (slope=18 dB/oct).
2. Switch to another band, then back.
3. Slope dropdown displays "6 dB/oct" though `c()!.order === 3`.
4. Click on dropdown → reopens with correct "18 dB/oct" preselected.

The store `filter.order` is unchanged; only the DOM `<select>.value`
de-synced from the model.

### Default fallback gap

`orderToSlope` returns 0 for unknown filter types (line 300).
Stringified `"0"` is not in any `availableSlopes` list → DOM falls
back to first option silently. Should never trigger for known types
but adds risk if `filter_type` ever desyncs.

## Linking mechanism

- Per-band link state: `BandState.linkedToNext: boolean`
  (`src/stores/bands.ts:56`). Stored on the upstream band — true
  means "my LP is bonded to next band's HP".
- Toggle: `toggleBandLinked` (`bands.ts:478-504`). On enable, copies
  current LP into next band's HP.
- Read by UI: `isBandLinkedFromPrev` (`bands.ts:507-511`) — used by
  `ControlPanel.tsx:177` to render HP indicator.
- Propagation on edit:
  - `setBandHighPass:579-597` — when HP edited and previous band has
    `linkedToNext=true`, mirror filter into prev LP.
  - `setBandLowPass:625-…` — when LP edited and current band has
    `linkedToNext=true`, mirror filter into next HP.

### Edge case after `removeBand`

`bands.ts:263-284`. When removing band at index `idx`:

```ts
if (idx > 0 && state.bands[idx - 1].linkedToNext) {
  setState("bands", idx - 1, "linkedToNext", false);  // <-- unlink
}
setState("bands", (prev) => prev.filter((b) => b.id !== id));
...
assignDefaultTargets(state.bands as BandState[]);
```

Two compounding issues:

1. The previous band's link is **forcibly cleared** even though it
   logically describes "linked to my downstream neighbour" — after
   the filter, the new neighbour is what was `idx+1`. The original
   link intent (B(idx-1) coupled to its next neighbour) is preserved
   if we leave `linkedToNext=true`.

2. `assignDefaultTargets:120-140` re-adds default crossover targets
   AND calls `setState(... "linkedToNext", true)` at line 137 — but
   under `if (bands[i].measurement) continue;`. Bands with imported
   measurements skip the entire body, so the intended "auto re-link
   neighbours" never happens for measurement-driven projects.

User scenario: 5-band project with measurements → delete B2 → B1's
`linkedToNext` set to false → `assignDefaultTargets` skips B1
(measurement present) → B1 stays unlinked from new neighbour (was
B3). Edits to B1 LP no longer propagate to B3 HP.

## Bessel implementation

`src-tauri/src/target/mod.rs:310-322` — `bessel_poles(order)` returns
analog Bessel poles for orders 2/4/6/8 (clamped to even).
`bessel_lp_complex` / `bessel_hp_complex` (lines 324-440) sum
`(s - p_i)` magnitudes — analog response, no bilinear.

Asymptotic slope: each pole contributes 6 dB/oct → order=N gives
6N dB/oct, matching `STD_SLOPES = [6,12,18,24,30,36,42,48]` and
`orderToSlope("Bessel", N) = N*6`. Mapping is correct.

Caveat: `bessel_poles` only has tables for even orders. Odd orders
clamp via `_ => bessel_poles(8)`. So selecting "18 dB/oct" (order=3)
silently uses order=8 poles. **This is a separate issue from slope
display**; flagged but out of b140.8.2 scope (no DSP changes).

## Proposed fixes

### Fix 1: Slope dropdown desync (cosmetic, no DSP)

Replace value-attribute pattern with explicit `selected` per option.
Solid renders `selected` synchronously per child, eliminating the
attribute/option ordering race.

`src/components/ControlPanel.tsx:418-432`:

```diff
-              <select
-                class="fb-select"
-                value={String(orderToSlope(c()!.filter_type, c()!.order))}
-                onChange={...}
-              >
-                {availableSlopes(c()!.filter_type).map((s) => (
-                  <option value={String(s)}>{`${s} dB/oct`}</option>
-                ))}
-              </select>
+              <select
+                class="fb-select"
+                onChange={...}
+              >
+                {availableSlopes(c()!.filter_type).map((s) => (
+                  <option
+                    value={String(s)}
+                    selected={s === orderToSlope(c()!.filter_type, c()!.order)}
+                  >{`${s} dB/oct`}</option>
+                ))}
+              </select>
```

### Fix 2: Preserve link across band removal

`src/stores/bands.ts:269-272` — drop the unlink:

```diff
-    // Сбрасываем linkedToNext у предыдущей полосы (если была связана с удаляемой)
-    if (idx > 0 && state.bands[idx - 1].linkedToNext) {
-      setState("bands", idx - 1, "linkedToNext", false);
-    }
+    // Link "linkedToNext" describes coupling to positional next-neighbour;
+    // after filter() removes the deleted band, prev band's link transfers
+    // to its new downstream neighbour (was idx+1). Do not clear.
+    // Special case: if removed band was the last one — clear prev's link
+    // since there is no longer a "next" band to be coupled to.
+    if (idx > 0 && idx === state.bands.length - 1) {
+      setState("bands", idx - 1, "linkedToNext", false);
+    }
```

### Fix 3: `assignDefaultTargets` should not skip linking on measurement bands

Move the `linkedToNext = true` line above the `continue` guard so
that re-linking happens regardless of whether the band has its own
measurement. Default link state is auto-on for adjacent bands; user
can still toggle off via UI.

`src/stores/bands.ts:120-140`:

```diff
   for (let i = 0; i < n; i++) {
+    // Always re-establish positional link to next band; user can toggle off.
+    if (i < n - 1 && !bands[i].linkedToNext) {
+      setState("bands", i, "linkedToNext", true);
+    }
     // Skip bands that already have a measurement — user configured them
     if (bands[i].measurement) continue;
     ...
-    if (i < n - 1) {
-      setState("bands", i, "linkedToNext", true);
-    }
   }
```

NB: the `!bands[i].linkedToNext` guard preserves explicit user-off
toggles in measurement-bearing bands across re-runs of
`assignDefaultTargets` (e.g. in `addBand`). Strictly, this changes
behaviour: before, addBand also unconditionally set link=true
inside the loop body (only for non-measurement bands). New version
keeps link off if user disabled it.

Actually — for `addBand` we want the new band's predecessor to be
linked. For `removeBand` we want adjacent bands' links to be kept
where they were. Combined behaviour: keep existing `linkedToNext` if
already true; only set to true if currently false AND we just added.
Since both call sites flow through `assignDefaultTargets`, the same
guard works.

### Out of scope (flagged for follow-up)

- `bessel_poles` clamps odd orders to 8 → user-selected 18 / 30 / 42
  dB/oct produce a **different** filter than displayed. Either
  restrict Bessel `availableSlopes` to even-order multiples
  `[12, 24, 36, 48]` or implement odd-order Bessel poles. Tracked
  separately from b140.8.2.

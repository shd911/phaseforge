// b141.7: content-keyed LRU memo cache for evaluateBandFull / evaluateSum.
// Display-only re-renders (phase/tab/SUM toggles) repeat the exact same
// evaluation request; the DSP inputs live in the key, so a repeat is a pure
// Map hit instead of 10–50 Tauri IPC round-trips.
// Design: docs/superpowers/specs/2026-06-11-band-eval-cache-design.md.
//
// Mutation safety: renderBandMode mutates the result envelope in place
// (FrequencyPlot interpolation onto extendedFreq), so the cache stores the
// pristine object and every return — hit AND miss — is a structuredClone.
//
// Measurement is keyed by object identity (WeakMap id): production replaces
// band.measurement wholesale (project-io.ts), never mutates it in place.

import type { BandState } from "../../stores/bands";
import type { BandEvalRequest } from "./evaluate";
import type { SumEvalOptions } from "./sum";

const CACHE_CAP = 32;
const store = new Map<string, unknown>();

let measSeq = 0;
const measIds = new WeakMap<object, number>();

function objectId(o: object | null | undefined): number | null {
  if (!o) return null;
  let id = measIds.get(o);
  if (id === undefined) {
    id = ++measSeq;
    measIds.set(o, id);
  }
  return id;
}

/** Measurement key = identity of the object AND of its data arrays.
 *  bands.ts delay-remove/restore swaps `measurement.phase` in place via
 *  setState path update — the measurement object keeps its identity, so the
 *  sub-array identities must be part of the key. */
function measurementKey(m: { freq?: number[]; magnitude?: number[]; phase?: number[] | null } | null | undefined): string | null {
  if (!m) return null;
  return [objectId(m), objectId(m.freq), objectId(m.magnitude), objectId(m.phase)].join(".");
}

/** FNV-1a (32-bit) over the raw float bits of a numeric array. Endpoints +
 *  length are NOT collision-safe as a grid key (a log grid and a measurement
 *  grid can share both), so the hash covers every value. */
function hashGrid(values: number[] | undefined): string | null {
  if (!values || values.length === 0) return null;
  const f = new Float64Array(values);
  const u = new Uint32Array(f.buffer);
  let h = 0x811c9dc5;
  for (let i = 0; i < u.length; i++) {
    h ^= u[i];
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(36) + ":" + values.length;
}

/** DSP-relevant band content as read by evaluateBandFull: targetEnabled,
 *  target curve, ENABLED PEQ bands (disabled ones never reach the pipeline),
 *  smoothing mode, measurement identity. */
function bandContentKey(band: BandState): string {
  const enabledPeq = (band.peqBands ?? []).filter((p) => p.enabled);
  return JSON.stringify({
    te: band.targetEnabled,
    t: band.target,
    p: enabledPeq,
    s: band.settings?.smoothing ?? null,
    m: measurementKey(band.measurement),
  });
}

export function bandRequestKey(req: BandEvalRequest): string {
  return "band|" + bandContentKey(req.band) + "|" + JSON.stringify({
    f: hashGrid(req.freq),
    fir: req.fir ?? null,
    ir: req.includeIr ?? false,
    ref: req.refLevelOverride ?? null,
    sr: req.sampleRate ?? null,
  });
}

/** evaluateSum additionally reads inverted + alignmentDelay per band. */
export function sumRequestKey(bands: BandState[], options?: SumEvalOptions): string {
  const perBand = bands
    .map((b) => bandContentKey(b) + `|inv:${b.inverted}|ad:${b.alignmentDelay}`)
    .join("§");
  return "sum|" + perBand + "|" + JSON.stringify({
    f: hashGrid(options?.freq),
    ir: options?.includeIr ?? false,
    sr: options?.sampleRate ?? null,
  });
}

export async function memoEval<T>(key: string, compute: () => Promise<T>): Promise<T> {
  const hit = store.get(key);
  if (hit !== undefined) {
    // LRU bump: delete + re-set moves the key to the back of the Map.
    store.delete(key);
    store.set(key, hit);
    return structuredClone(hit) as T;
  }
  const result = await compute();
  store.set(key, result);
  if (store.size > CACHE_CAP) {
    store.delete(store.keys().next().value as string);
  }
  return structuredClone(result);
}

export function clearBandEvalCache(): void {
  store.clear();
}

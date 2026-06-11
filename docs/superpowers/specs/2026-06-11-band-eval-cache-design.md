# BandEvalResult cache — design (b141.7)

## Problem
Display-only toggles (phase/tab/SUM switch) re-run `evaluateBandFull` /
`evaluateSum` from scratch: 10–50 Tauri IPC round-trips per render, full DSP
recompute although no DSP input changed. Single render effect in
FrequencyPlot.tsx (line ~1643) tracks both display signals and data signals.

## Approach (chosen of 3)
Transparent memo cache at the `evaluateBandFull` / `evaluateSum` boundary in
`src/lib/band-evaluator/cache.ts`. Callers unchanged; golden tests stay valid.
(Rejected: bandsVersion-keyed cache — global invalidation + store dependency in
lib; render-effect split in FrequencyPlot — high regression risk on 4.5k-line
file.)

## Cache key (string, composite)
- Band content: `targetEnabled`, JSON of `target`, JSON of **enabled** PEQ
  bands, `settings.smoothing`.
- Measurement: object identity via `WeakMap<object, number>` id. Assumption
  (verified): production replaces `band.measurement` wholesale
  (project-io.ts:445,455,482), never mutates in place.
- Request options: FNV-1a hash over `req.freq` values (endpoints+length are
  not collision-safe: log grid vs measurement grid can share both), JSON of
  `fir`, `includeIr`, `refLevelOverride`, `sampleRate`.

## Storage & eviction
Module-level `Map<string, BandEvalResult>` as LRU (delete+re-set on hit),
cap 32 entries. `evaluateSum` cached too: key = joined per-band keys + options.
Invalidation is automatic — any content change yields a new key; stale entries
age out by LRU.

## Mutation safety
`renderBandMode` mutates the result envelope in place
(FrequencyPlot.tsx:2918-2921). Cache therefore stores the pristine object and
returns `structuredClone` on every hit AND miss. Clone cost (≤ a few ms even
with IR/FIR payloads) ≪ IPC recompute (100s of ms).

## Testing (TDD)
- Existing golden-sum / grid-alignment / band-evaluator suites = behaviour
  guard (cache is transparent).
- New `band-eval-cache.test.ts`: repeat identical call → 0 extra `invoke`
  calls; change PEQ / smoothing / grid / sampleRate / measurement object →
  recompute; LRU eviction at cap; returned object is not the cached instance
  (mutation of a result does not poison subsequent hits).

## Version
b141.7 (tauri.conf.json title, lib.rs startup log, version.ts).

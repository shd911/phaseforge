// Canonical BandEvaluator (b139.1) — single source of truth for band
// evaluation.
//
// b140.14: file is a thin coordinator. The actual implementation lives
// in the submodules under ./band-evaluator/:
//   - grid.ts      (b140.14)   — freq-grid helpers
//   - route.ts     (b140.14.1) — Rust pipeline dispatch
//   - extension.ts (b140.14.2) — appendNoiseFloorTail, autoRefLevel,
//                                computeExtension
//   - sum.ts       (b140.14.3) — evaluateSum + sum-only helpers
//   - evaluate.ts  (b140.14.4) — evaluateBandFull + types +
//                                reconstructTargetPhase
//
// All names re-exported below so existing callers
// (`import { evaluateBandFull, evaluateSum, reconstructTargetPhase }
// from "../lib/band-evaluator"`) keep working unchanged.

// --- Public re-exports -----------------------------------------------------

export {
  evaluateBandFull,
  reconstructTargetPhase,
} from "./band-evaluator/evaluate";
export type {
  BandEvalRequest,
  BandEvalResult,
  FirRequestConfig,
} from "./band-evaluator/evaluate";

export { evaluateSum } from "./band-evaluator/sum";
export type { SumEvalResult, SumEvalOptions } from "./band-evaluator/sum";

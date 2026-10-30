import { addExclusionZone, setBandSmoothing, type SmoothingMode } from "../stores/bands";
import { setPeqDirectLow, setPeqDirectHigh, setPeqRangeMode } from "../stores/peq-optimize";
import { historyGroup } from "../stores/history";
import type { AnalysisRecommendation } from "./types";

const SMOOTHING_VALUES: SmoothingMode[] = ["off", "1/3", "1/6", "1/12", "1/24", "var"];

export function applyRecommendation(bandId: string, rec: AnalysisRecommendation, findingId: string): void {
  // The optimizer range is a project setting and is not undone (b141.103,
  // audit 3.1): an undo step for it would restore nothing (b141.128).
  if (rec.action.type === "SetOptLowerBound" || rec.action.type === "SetOptUpperBound") {
    applyAction(bandId, rec);
    return;
  }
  // One undo step per recommendation: the setters' own history pushes
  // (smoothing, exclusion zone) are absorbed.
  historyGroup(`Apply recommendation: ${findingId}`, () => applyAction(bandId, rec));
}

function applyAction(bandId: string, rec: AnalysisRecommendation): void {
  const a = rec.action;
  switch (a.type) {
    case "SetOptLowerBound":
      setPeqDirectLow(a.value);
      setPeqRangeMode("direct");
      break;
    case "SetOptUpperBound":
      setPeqDirectHigh(a.value);
      setPeqRangeMode("direct");
      break;
    case "AddExclusionZone":
      addExclusionZone(bandId, { startHz: a.value.low_hz, endHz: a.value.high_hz });
      break;
    case "ApplySmoothing": {
      const v = SMOOTHING_VALUES.includes(a.value as SmoothingMode)
        ? (a.value as SmoothingMode)
        : "1/6";
      setBandSmoothing(bandId, v);
      break;
    }
  }
}

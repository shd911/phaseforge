import { addExclusionZone, setBandSmoothing, type SmoothingMode } from "../stores/bands";
import { setPeqDirectLow, setPeqDirectHigh, setPeqRangeMode } from "../stores/peq-optimize";
import { pushHistory } from "../stores/history";
import type { AnalysisRecommendation } from "./types";

const SMOOTHING_VALUES: SmoothingMode[] = ["off", "1/3", "1/6", "1/12", "1/24", "var"];

export function applyRecommendation(bandId: string, rec: AnalysisRecommendation, findingId: string): void {
  pushHistory(`Apply recommendation: ${findingId}`);
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

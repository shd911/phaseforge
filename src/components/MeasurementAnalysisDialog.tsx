import { createSignal, Show, For, onCleanup } from "solid-js";
import type { AnalysisFinding, AnalysisRecommendation, AnalysisResult } from "../lib/types";
import { applyRecommendation } from "../lib/analysis-actions";
import { setBandAnalysisDismissed } from "../stores/bands";
import { beginInteraction, commitInteraction } from "../stores/history";

interface OpenRequest {
  bandId: string;
  bandName: string;
  fileName: string;
  result: AnalysisResult;
}

const [_open, setOpen] = createSignal<OpenRequest | null>(null);
const [_appliedIds, _setAppliedIds] = createSignal<Set<string>>(new Set());

export function openMeasurementAnalysis(req: OpenRequest): void {
  // Reset applied state per open so a previous band's marks don't leak in.
  _setAppliedIds(new Set<string>());
  setOpen(req);
}

export const isAnalysisDialogOpen = () => _open() !== null;

function severityIcon(s: AnalysisFinding["severity"]): string {
  if (s === "Error") return "✕";
  if (s === "Warning") return "⚠";
  return "ℹ";
}

export default function MeasurementAnalysisDialog() {
  const appliedFindingIds = _appliedIds;

  function close() {
    const req = _open();
    if (req) setBandAnalysisDismissed(req.bandId, true);
    setOpen(null);
  }

  function markApplied(findingId: string) {
    const a = new Set(_appliedIds());
    a.add(findingId);
    _setAppliedIds(a);
  }

  function applyFinding(req: OpenRequest, finding: AnalysisFinding, rec: AnalysisRecommendation) {
    applyRecommendation(req.bandId, rec, finding.id);
    markApplied(finding.id);
  }

  function applyAll() {
    const req = _open();
    if (!req) return;
    // Single history entry for the whole batch (b132 begin/commit suppresses
    // intermediate pushHistory calls, so undo restores pre-batch state in
    // one Cmd+Z regardless of finding count).
    beginInteraction("Apply all recommendations");
    try {
      for (const f of req.result.findings) {
        if (_appliedIds().has(f.id)) continue;
        const rec = f.recommendations[0];
        if (rec) applyFinding(req, f, rec);
      }
    } finally {
      commitInteraction();
    }
  }

  const onKey = (e: KeyboardEvent) => {
    if (!_open()) return;
    if (e.key === "Escape") {
      e.preventDefault();
      close();
    }
  };
  window.addEventListener("keydown", onKey);
  onCleanup(() => window.removeEventListener("keydown", onKey));

  return (
    <Show when={_open()}>
      {(req) => {
        const findings = () => req().result.findings;
        const counts = () => {
          const c = { warn: 0, info: 0, err: 0 };
          for (const f of findings()) {
            if (f.severity === "Warning") c.warn++;
            else if (f.severity === "Error") c.err++;
            else c.info++;
          }
          return c;
        };
        const allApplied = () =>
          findings().length > 0 && appliedFindingIds().size >= findings().length;
        const remainingCount = () => findings().length - appliedFindingIds().size;
        return (
          <div
            class="pn-overlay"
            onMouseDown={(e) => { if (e.target === e.currentTarget) close(); }}
          >
            <div class="pn-dialog" style={{ "min-width": "520px", "max-width": "640px" }}>
              <div class="pn-title">Анализ замера: {req().fileName}</div>

              <Show
                when={findings().length > 0}
                fallback={
                  <div style={{ padding: "32px 0", "text-align": "center" }}>
                    <div style={{ "font-size": "32px", color: "#7c7", "margin-bottom": "12px" }}>✓</div>
                    <div style={{ "font-size": "16px", "margin-bottom": "6px" }}>Замер выглядит чисто</div>
                    <div style={{ color: "#aaa", "font-size": "13px" }}>Анализ не выявил подозрительных участков.</div>
                  </div>
                }
              >
                <div style={{ "margin-bottom": "8px", "font-size": "13px" }}>
                  Выберите рекомендации, которые применить. Можно по одной или сразу все.
                </div>
                <div style={{ "margin-bottom": "12px", color: "#aaa", "font-size": "12px" }}>
                  Найдено: {counts().warn} предупреждений, {counts().info} заметок
                  {counts().err > 0 ? `, ${counts().err} ошибок` : ""}
                </div>

                <div style={{ "max-height": "420px", "overflow-y": "auto", "border-top": "1px solid #333" }}>
                  <For each={findings()}>
                    {(f) => {
                      const isApplied = () => appliedFindingIds().has(f.id);
                      return (
                        <div style={{ padding: "10px 4px", "border-bottom": "1px solid #333" }}>
                          <div style={{ "font-weight": 600 }}>
                            {severityIcon(f.severity)} {f.title}
                            <Show when={isApplied()}>
                              <span style={{ "margin-left": "8px", color: "#7c7", "font-weight": 400 }}>
                                ✓ применено
                              </span>
                            </Show>
                          </div>
                          <div style={{ color: "#aaa", "font-size": "12px", "margin": "2px 0 6px" }}>
                            {f.description}
                          </div>
                          <For each={f.recommendations}>
                            {(rec) => (
                              <button
                                class="dlg-btn"
                                style={{ "margin-right": "6px" }}
                                disabled={isApplied()}
                                onClick={() => applyFinding(req(), f, rec)}
                              >{isApplied() ? `✓ Установлено: ${rec.label}` : rec.label}</button>
                            )}
                          </For>
                        </div>
                      );
                    }}
                  </For>
                </div>
              </Show>

              <div class="pn-buttons" style={{ "margin-top": "16px" }}>
                <button class="dlg-btn" onClick={close}>Закрыть</button>
                <Show when={findings().length > 0}>
                  <button
                    class="dlg-btn dlg-btn-primary"
                    onClick={applyAll}
                    disabled={allApplied()}
                  >{allApplied()
                    ? "Все применены"
                    : remainingCount() === findings().length
                      ? "Применить все"
                      : `Применить оставшиеся (${remainingCount()})`}</button>
                </Show>
              </div>
            </div>
          </div>
        );
      }}
    </Show>
  );
}

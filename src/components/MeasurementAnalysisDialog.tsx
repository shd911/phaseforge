import { createSignal, Show, For, onCleanup } from "solid-js";
import type { AnalysisFinding, AnalysisRecommendation, AnalysisResult } from "../lib/types";
import { applyRecommendation } from "../lib/analysis-actions";
import { setBandAnalysisDismissed } from "../stores/bands";

interface OpenRequest {
  bandId: string;
  bandName: string;
  fileName: string;
  result: AnalysisResult;
}

const [_open, setOpen] = createSignal<OpenRequest | null>(null);

export function openMeasurementAnalysis(req: OpenRequest): void {
  setOpen(req);
}

export const isAnalysisDialogOpen = () => _open() !== null;

function severityIcon(s: AnalysisFinding["severity"]): string {
  if (s === "Error") return "✕";
  if (s === "Warning") return "⚠";
  return "ℹ";
}

export default function MeasurementAnalysisDialog() {
  const [selected, setSelected] = createSignal<Set<string>>(new Set());
  const [appliedFindingIds, setAppliedFindingIds] = createSignal<Set<string>>(new Set());

  function close(dismissed: boolean) {
    const req = _open();
    if (req && dismissed) {
      setBandAnalysisDismissed(req.bandId, true);
    }
    setOpen(null);
    setSelected(new Set<string>());
    setAppliedFindingIds(new Set<string>());
  }

  function toggleSelect(findingId: string) {
    const s = new Set(selected());
    if (s.has(findingId)) s.delete(findingId);
    else s.add(findingId);
    setSelected(s);
  }

  function applyOne(finding: AnalysisFinding, rec: AnalysisRecommendation) {
    const req = _open();
    if (!req) return;
    applyRecommendation(req.bandId, rec, finding.id);
    const a = new Set(appliedFindingIds());
    a.add(finding.id);
    setAppliedFindingIds(a);
  }

  function applySelected() {
    const req = _open();
    if (!req) return;
    const sel = selected();
    for (const f of req.result.findings) {
      if (!sel.has(f.id)) continue;
      if (appliedFindingIds().has(f.id)) continue;
      const rec = f.recommendations[0];
      if (rec) applyRecommendation(req.bandId, rec, f.id);
    }
    close(true);
  }

  const onKey = (e: KeyboardEvent) => {
    if (!_open()) return;
    if (e.key === "Escape") {
      e.preventDefault();
      close(true);
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
        return (
          <div
            class="pn-overlay"
            onMouseDown={(e) => { if (e.target === e.currentTarget) close(true); }}
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
                <div style={{ "margin-bottom": "12px", color: "#aaa", "font-size": "12px" }}>
                  Найдено: {counts().warn} предупреждений, {counts().info} заметок
                  {counts().err > 0 ? `, ${counts().err} ошибок` : ""}
                </div>

                <div style={{ "max-height": "420px", "overflow-y": "auto", "border-top": "1px solid #333" }}>
                  <For each={findings()}>
                    {(f) => (
                      <div style={{
                        padding: "10px 4px",
                        "border-bottom": "1px solid #333",
                        opacity: appliedFindingIds().has(f.id) ? 0.5 : 1,
                      }}>
                        <label style={{ display: "flex", "align-items": "flex-start", gap: "8px", cursor: "pointer" }}>
                          <input
                            type="checkbox"
                            checked={selected().has(f.id)}
                            disabled={appliedFindingIds().has(f.id)}
                            onChange={() => toggleSelect(f.id)}
                            style={{ "margin-top": "3px" }}
                          />
                          <div style={{ flex: 1 }}>
                            <div style={{ "font-weight": 600 }}>
                              {severityIcon(f.severity)} {f.title}
                              <Show when={appliedFindingIds().has(f.id)}>
                                <span style={{ "margin-left": "8px", color: "#7c7" }}>✓ применено</span>
                              </Show>
                            </div>
                            <div style={{ color: "#aaa", "font-size": "12px", "margin-top": "2px" }}>
                              {f.description}
                            </div>
                            <For each={f.recommendations}>
                              {(rec) => (
                                <button
                                  class="dlg-btn"
                                  style={{ "margin-top": "6px" }}
                                  disabled={appliedFindingIds().has(f.id)}
                                  onClick={() => applyOne(f, rec)}
                                >{rec.label}</button>
                              )}
                            </For>
                          </div>
                        </label>
                      </div>
                    )}
                  </For>
                </div>
              </Show>

              <div class="pn-buttons" style={{ "margin-top": "16px" }}>
                <button class="dlg-btn" onClick={() => close(true)}>
                  {findings().length > 0 ? "Игнорировать всё" : "Закрыть"}
                </button>
                <Show when={findings().length > 0}>
                  <button
                    class="dlg-btn dlg-btn-primary"
                    onClick={applySelected}
                    disabled={selected().size === 0}
                  >Применить отмеченное</button>
                </Show>
              </div>
            </div>
          </div>
        );
      }}
    </Show>
  );
}

import { createSignal, Show } from "solid-js";
import { open } from "@tauri-apps/plugin-dialog";
import { invoke } from "@tauri-apps/api/core";
import type { Measurement, MergeConfig, MergeResult, BaffleConfig } from "../lib/types";
import type { MergeSource } from "../stores/bands";
import NumberInput from "./NumberInput";
import BaffleStepDialog from "./BaffleStepDialog";

interface MergeDialogProps {
  onClose: () => void;
  onMerge: (measurement: Measurement, source: MergeSource) => void;
}

const BLEND_OPTIONS = [0.5, 1.0, 2.0, 3.0];

export default function MergeDialog(props: MergeDialogProps) {
  const [nfPath, setNfPath] = createSignal<string | null>(null);
  const [ffPath, setFfPath] = createSignal<string | null>(null);
  const [nfName, setNfName] = createSignal("");
  const [ffName, setFfName] = createSignal("");
  const [spliceFreq, setSpliceFreq] = createSignal(300);
  const [blendOctaves, setBlendOctaves] = createSignal(1.0);
  const [autoOffset, setAutoOffset] = createSignal<number | null>(null);
  const [manualOffset, setManualOffset] = createSignal<number | null>(null);
  const [useManualOffset, setUseManualOffset] = createSignal(false);
  const [merging, setMerging] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);

  // Baffle step
  const [baffleEnabled, setBaffleEnabled] = createSignal(false);
  const [baffleConfig, setBaffleConfig] = createSignal<BaffleConfig>({
    baffle_width_m: 0.25,
    baffle_height_m: 0.35,
    driver_offset_x_m: 0.25 / 2,
    driver_offset_y_m: 0.12,
  });
  const [showBaffleDialog, setShowBaffleDialog] = createSignal(false);

  async function selectFile(which: "nf" | "ff") {
    try {
      const selected = await open({
        multiple: false,
        filters: [{ name: "Measurement Files", extensions: ["txt", "frd"] }],
      });
      if (!selected) return;
      const filePath = Array.isArray(selected) ? selected[0] : selected;
      const name =
        filePath
          .split(/[/\\]/)
          .pop()
          ?.replace(/\.(txt|frd)$/i, "") ?? filePath;

      if (which === "nf") {
        setNfPath(filePath);
        setNfName(name);
      } else {
        setFfPath(filePath);
        setFfName(name);
      }

      // Auto-compute offset when both files selected
      const nf = which === "nf" ? filePath : nfPath();
      const ff = which === "ff" ? filePath : ffPath();
      if (nf && ff) {
        await tryAutoCompute(nf, ff);
      }
    } catch (e) {
      setError(String(e));
    }
  }

  async function tryAutoCompute(nf: string, ff: string) {
    try {
      const config: MergeConfig = {
        splice_freq: spliceFreq(),
        blend_octaves: blendOctaves(),
        level_offset_db: null,
        match_range: null,
        baffle: null,
      };
      const result = await invoke<MergeResult>("merge_measurements", {
        nfPath: nf,
        ffPath: ff,
        config,
      });
      setAutoOffset(result.auto_level_offset_db);
      setError(null);
    } catch (e) {
      setAutoOffset(null);
      setError(String(e));
    }
  }

  async function handleMerge() {
    const nf = nfPath();
    const ff = ffPath();
    if (!nf || !ff) return;

    setMerging(true);
    setError(null);
    try {
      const config: MergeConfig = {
        splice_freq: spliceFreq(),
        blend_octaves: blendOctaves(),
        level_offset_db: useManualOffset() ? manualOffset() : null,
        match_range: null,
        baffle: baffleEnabled() ? baffleConfig() : null,
      };
      const result = await invoke<MergeResult>("merge_measurements", {
        nfPath: nf,
        ffPath: ff,
        config,
      });
      props.onMerge(result.measurement, { nfPath: nf, ffPath: ff, config });
      props.onClose();
    } catch (e) {
      setError(String(e));
    } finally {
      setMerging(false);
    }
  }

  return (
    <div class="merge-overlay" onClick={props.onClose}>
      <div class="merge-dialog" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div class="merge-header">
          <span class="merge-title">Merge NF + FF</span>
          <button class="merge-close" onClick={props.onClose}>
            ×
          </button>
        </div>

        {/* Body */}
        <div class="merge-body">
          {/* File selectors */}
          <div class="merge-file-row">
            <label class="merge-label">Near-Field</label>
            <button class="merge-file-btn" onClick={() => selectFile("nf")}>
              {nfName() || "Select NF file…"}
            </button>
          </div>
          <div class="merge-file-row">
            <label class="merge-label">Far-Field</label>
            <button class="merge-file-btn" onClick={() => selectFile("ff")}>
              {ffName() || "Select FF file…"}
            </button>
          </div>

          {/* Parameters */}
          <Show when={nfPath() && ffPath()}>
            <div class="merge-params">
              <div class="merge-param-row">
                <label class="merge-label">Splice Freq</label>
                <NumberInput
                  value={spliceFreq()}
                  onChange={(v) => setSpliceFreq(v)}
                  min={50}
                  max={1000}
                  step={10}
                  unit="Hz"
                  freqMode
                />
              </div>

              <div class="merge-param-row">
                <label class="merge-label">Blend Width</label>
                <select
                  class="merge-select"
                  value={blendOctaves()}
                  onChange={(e) =>
                    setBlendOctaves(parseFloat(e.currentTarget.value))
                  }
                >
                  {BLEND_OPTIONS.map((v) => (
                    <option value={v}>{v} oct</option>
                  ))}
                </select>
              </div>

              <div class="merge-param-row">
                <label class="merge-label">Level Offset</label>
                <span class="merge-auto-val">
                  Auto:{" "}
                  {autoOffset() !== null
                    ? autoOffset()!.toFixed(1) + " dB"
                    : "…"}
                </span>
              </div>

              <div class="merge-param-row">
                <label class="merge-check-label">
                  <input
                    type="checkbox"
                    checked={useManualOffset()}
                    onChange={() => setUseManualOffset(!useManualOffset())}
                  />
                  Manual override
                </label>
                <Show when={useManualOffset()}>
                  <NumberInput
                    value={manualOffset() ?? autoOffset() ?? 0}
                    onChange={(v) => setManualOffset(v)}
                    min={-30}
                    max={30}
                    step={0.5}
                    unit="dB"
                    precision={1}
                  />
                </Show>
              </div>

              {/* Baffle Step */}
              <div class="merge-param-row">
                <label class="merge-check-label">
                  <input
                    type="checkbox"
                    checked={baffleEnabled()}
                    onChange={() => setBaffleEnabled(!baffleEnabled())}
                  />
                  Baffle step
                </label>
                <Show when={baffleEnabled()}>
                  <button
                    class="meas-action-btn"
                    onClick={() => setShowBaffleDialog(true)}
                  >
                    Configure...
                  </button>
                </Show>
              </div>
            </div>
          </Show>

          {/* Error */}
          <Show when={error()}>
            <div class="merge-error">{error()}</div>
          </Show>
        </div>

        {/* Footer */}
        <div class="merge-footer">
          <button class="tb-btn" onClick={props.onClose}>
            Cancel
          </button>
          <button
            class="tb-btn primary"
            disabled={!nfPath() || !ffPath() || merging()}
            onClick={handleMerge}
          >
            {merging() ? "Merging…" : "Merge"}
          </button>
        </div>
      </div>

      {/* Baffle step configuration dialog */}
      <Show when={showBaffleDialog()}>
        <BaffleStepDialog
          config={baffleConfig()}
          onSave={(c) => {
            setBaffleConfig(c);
            setShowBaffleDialog(false);
          }}
          onClose={() => setShowBaffleDialog(false)}
        />
      </Show>
    </div>
  );
}

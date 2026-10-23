import { createSignal, createEffect, on, Show } from "solid-js";
import { open } from "@tauri-apps/plugin-dialog";
import { invoke } from "@tauri-apps/api/core";
import type { Measurement, MergeConfig, MergeResult, BaffleConfig } from "../lib/types";
import type { MergeSource } from "../stores/bands";
import { showToast } from "../lib/toast";
import NumberInput from "./NumberInput";
import BaffleStepDialog from "./BaffleStepDialog";
import Modal from "./Modal";

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
        filters: [{ name: "Файлы измерений", extensions: ["txt", "frd"] }],
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

  /** The merge config «Слить» will send; `level_offset_db` decides auto vs manual. */
  const mergeConfig = (levelOffset: number | null): MergeConfig => ({
    splice_freq: spliceFreq(),
    blend_octaves: blendOctaves(),
    level_offset_db: levelOffset,
    match_range: null,
    baffle: baffleEnabled() ? baffleConfig() : null,
  });

  // b141.105 (audit 2026-10-23 §5.23): «Авто» was computed without the baffle
  // correction and only when a file was picked, but the merge applies the
  // baffle (and the current splice/blend) before matching levels — the shown
  // number, and the manual field seeded from it, were not what the merge
  // used. Recompute from the same config whenever an input changes.
  let autoGen = 0;
  async function tryAutoCompute(nf: string, ff: string) {
    const gen = ++autoGen;
    try {
      const config = mergeConfig(null);
      const result = await invoke<MergeResult>("merge_measurements", {
        nfPath: nf,
        ffPath: ff,
        config,
      });
      if (gen !== autoGen) return; // a newer request superseded this one
      setAutoOffset(result.auto_level_offset_db);
      setError(null);
    } catch (e) {
      if (gen !== autoGen) return;
      setAutoOffset(null);
      setError(String(e));
    }
  }

  createEffect(on(
    () => [spliceFreq(), blendOctaves(), baffleEnabled(), baffleConfig()] as const,
    () => { const nf = nfPath(); const ff = ffPath(); if (nf && ff) void tryAutoCompute(nf, ff); },
    { defer: true },
  ));

  async function handleMerge() {
    const nf = nfPath();
    const ff = ffPath();
    if (!nf || !ff) return;

    setMerging(true);
    setError(null);
    try {
      const config = mergeConfig(useManualOffset() ? (manualOffset() ?? autoOffset()) : null);
      const result = await invoke<MergeResult>("merge_measurements", {
        nfPath: nf,
        ffPath: ff,
        config,
      });
      props.onMerge(result.measurement, { nfPath: nf, ffPath: ff, config });
      for (const w of result.warnings ?? []) showToast(w, "warn", 12000);
      props.onClose();
    } catch (e) {
      setError(String(e));
    } finally {
      setMerging(false);
    }
  }

  return (
    <Modal overlayClass="merge-overlay" dialogClass="merge-dialog" titleId="merge-dlg-title"
      onEscape={props.onClose}>
        {/* Header */}
        <div class="merge-header">
          <span class="merge-title" id="merge-dlg-title">Слияние NF + FF</span>
          <button class="merge-close" onClick={props.onClose} aria-label="Закрыть" title="Закрыть">
            ×
          </button>
        </div>

        {/* Body */}
        <div class="merge-body">
          {/* File selectors */}
          <div class="merge-file-row">
            <label class="merge-label">Ближнее поле (NF)</label>
            <button class="merge-file-btn" onClick={() => selectFile("nf")}>
              {nfName() || "Выбрать файл NF…"}
            </button>
          </div>
          <div class="merge-file-row">
            <label class="merge-label">Дальнее поле (FF)</label>
            <button class="merge-file-btn" onClick={() => selectFile("ff")}>
              {ffName() || "Выбрать файл FF…"}
            </button>
          </div>

          {/* Parameters */}
          <Show when={nfPath() && ffPath()}>
            <div class="merge-params">
              <div class="merge-param-row">
                <label class="merge-label">Частота сшивки</label>
                <NumberInput
                  label="Частота сшивки"
                  value={spliceFreq()}
                  onChange={(v) => setSpliceFreq(v)}
                  min={50}
                  max={1000}
                  step={10}
                  unit="Гц"
                  freqMode
                />
              </div>

              <div class="merge-param-row">
                <label class="merge-label" for="merge-blend">Ширина перехода</label>
                <select
                  id="merge-blend"
                  class="merge-select"
                  value={blendOctaves()}
                  onChange={(e) =>
                    setBlendOctaves(parseFloat(e.currentTarget.value))
                  }
                >
                  {BLEND_OPTIONS.map((v) => (
                    <option value={v}>{v} окт</option>
                  ))}
                </select>
              </div>

              <div class="merge-param-row">
                <label class="merge-label">Сдвиг уровня</label>
                <span class="merge-auto-val">
                  Авто:{" "}
                  {autoOffset() !== null
                    ? autoOffset()!.toFixed(1) + " дБ"
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
                  Задать вручную
                </label>
                <Show when={useManualOffset()}>
                  <NumberInput
                    label="Сдвиг уровня вручную"
                    value={manualOffset() ?? autoOffset() ?? 0}
                    onChange={(v) => setManualOffset(v)}
                    min={-30}
                    max={30}
                    step={0.5}
                    unit="дБ"
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
                  Коррекция baffle step
                </label>
                <Show when={baffleEnabled()}>
                  <button
                    class="meas-action-btn"
                    onClick={() => setShowBaffleDialog(true)}
                  >
                    Настроить…
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
          <button class="dlg-btn" onClick={props.onClose}>
            Отмена
          </button>
          <button
            class="dlg-btn dlg-btn-primary"
            disabled={!nfPath() || !ffPath() || merging()}
            onClick={handleMerge}
          >
            {merging() ? "Слияние…" : "Слить"}
          </button>
        </div>

      {/* Baffle step configuration dialog — its own modal on top */}
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
    </Modal>
  );
}

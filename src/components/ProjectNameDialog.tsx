import { createSignal, Show } from "solid-js";
import { promptVisible, promptMode, resolvePrompt } from "../lib/project-io";
import type { ProjectPromptResult } from "../lib/project-io";
import Modal from "./Modal";
import { L } from "../lib/i18n";

export default function ProjectNameDialog() {
  const [name, setName] = createSignal("");
  const [bandCount, setBandCount] = createSignal(2);

  function handleCreate() {
    const v = name().trim();
    if (v) resolvePrompt({ name: v, bandCount: bandCount() });
  }

  function handleCancel() {
    resolvePrompt(null);
  }

  function incBands() {
    setBandCount((c) => Math.min(8, c + 1));
  }
  function decBands() {
    setBandCount((c) => Math.max(1, c - 1));
  }

  const isSaveAs = () => promptMode() === "saveAs";

  return (
    <Show when={promptVisible()}>
      <Modal overlayClass="pn-overlay" dialogClass="pn-dialog" titleId="pn-dlg-title"
        onEnter={handleCreate} onEscape={handleCancel}>
          <div class="pn-title" id="pn-dlg-title">{isSaveAs() ? L("Сохранить как", "Save As") : L("Новый проект", "New project")}</div>

          <label class="pn-label" for="pn-name">{L("Название проекта", "Project name")}</label>
          <input
            ref={(el) => {
              requestAnimationFrame(() => el.focus());
            }}
            id="pn-name"
            class="pn-input"
            type="text"
            placeholder={L("Мой проект", "My project")}
            value={name()}
            onInput={(e) => setName(e.currentTarget.value)}
          />

          <Show when={!isSaveAs()}>
            <label class="pn-label" id="pn-band-count">{L("Количество полос", "Number of bands")}</label>
            <div class="pn-band-count" role="group" aria-labelledby="pn-band-count">
              <button class="pn-bc-btn" onClick={decBands} disabled={bandCount() <= 1} aria-label={L("Меньше полос", "Fewer bands")}>−</button>
              <span class="pn-bc-value">{bandCount()}</span>
              <button class="pn-bc-btn" onClick={incBands} disabled={bandCount() >= 8} aria-label={L("Больше полос", "More bands")}>+</button>
            </div>
          </Show>

          <div class="pn-buttons">
            <button class="dlg-btn" onClick={handleCancel}>{L("Отмена", "Cancel")}</button>
            <button
              class="dlg-btn dlg-btn-primary"
              onClick={handleCreate}
              disabled={!name().trim()}
            >{isSaveAs() ? L("Сохранить", "Save") : L("Создать", "Create")}</button>
          </div>
      </Modal>
    </Show>
  );
}

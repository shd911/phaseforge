import { createSignal, Show } from "solid-js";
import { promptVisible, promptMode, resolvePrompt } from "../lib/project-io";
import type { ProjectPromptResult } from "../lib/project-io";
import Modal from "./Modal";

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
          <div class="pn-title" id="pn-dlg-title">{isSaveAs() ? "Сохранить как" : "Новый проект"}</div>

          <label class="pn-label">Название проекта</label>
          <input
            ref={(el) => {
              requestAnimationFrame(() => el.focus());
            }}
            class="pn-input"
            type="text"
            placeholder="Мой проект"
            value={name()}
            onInput={(e) => setName(e.currentTarget.value)}
          />

          <Show when={!isSaveAs()}>
            <label class="pn-label">Количество бэндов</label>
            <div class="pn-band-count">
              <button class="pn-bc-btn" onClick={decBands} disabled={bandCount() <= 1}>−</button>
              <span class="pn-bc-value">{bandCount()}</span>
              <button class="pn-bc-btn" onClick={incBands} disabled={bandCount() >= 8}>+</button>
            </div>
          </Show>

          <div class="pn-buttons">
            <button class="dlg-btn" onClick={handleCancel}>Отмена</button>
            <button
              class="dlg-btn dlg-btn-primary"
              onClick={handleCreate}
              disabled={!name().trim()}
            >{isSaveAs() ? "Сохранить" : "Создать"}</button>
          </div>
      </Modal>
    </Show>
  );
}

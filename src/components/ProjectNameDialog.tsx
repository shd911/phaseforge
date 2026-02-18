import { createSignal, Show } from "solid-js";
import { promptVisible, resolvePrompt } from "../lib/project-io";
import type { ProjectPromptResult } from "../lib/project-io";

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

  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === "Enter") {
      e.preventDefault();
      handleCreate();
    } else if (e.key === "Escape") {
      e.preventDefault();
      handleCancel();
    }
  }

  function incBands() {
    setBandCount((c) => Math.min(8, c + 1));
  }
  function decBands() {
    setBandCount((c) => Math.max(1, c - 1));
  }

  return (
    <Show when={promptVisible()}>
      <div class="pn-overlay" onMouseDown={(e) => { if (e.target === e.currentTarget) handleCancel(); }}>
        <div class="pn-dialog">
          <div class="pn-title">New Project</div>

          <label class="pn-label">Project name</label>
          <input
            ref={(el) => {
              requestAnimationFrame(() => el.focus());
            }}
            class="pn-input"
            type="text"
            placeholder="My Project"
            value={name()}
            onInput={(e) => setName(e.currentTarget.value)}
            onKeyDown={handleKeyDown}
          />

          <label class="pn-label">Number of bands</label>
          <div class="pn-band-count">
            <button class="pn-bc-btn" onClick={decBands} disabled={bandCount() <= 1}>−</button>
            <span class="pn-bc-value">{bandCount()}</span>
            <button class="pn-bc-btn" onClick={incBands} disabled={bandCount() >= 8}>+</button>
          </div>

          <div class="pn-buttons">
            <button class="pn-btn pn-btn-cancel" onClick={handleCancel}>Cancel</button>
            <button
              class="pn-btn pn-btn-create"
              onClick={handleCreate}
              disabled={!name().trim()}
            >Create</button>
          </div>
        </div>
      </div>
    </Show>
  );
}

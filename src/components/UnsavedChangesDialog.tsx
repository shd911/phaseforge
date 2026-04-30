import { Show } from "solid-js";
import { unsavedDialogVisible, resolveUnsavedDialog } from "../lib/project-io";

export default function UnsavedChangesDialog() {
  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === "Escape") {
      e.preventDefault();
      resolveUnsavedDialog("cancel");
    } else if (e.key === "Enter") {
      e.preventDefault();
      resolveUnsavedDialog("save");
    }
  }

  return (
    <Show when={unsavedDialogVisible()}>
      <div
        class="pn-overlay"
        onMouseDown={(e) => { if (e.target === e.currentTarget) resolveUnsavedDialog("cancel"); }}
        tabIndex={-1}
        ref={(el) => requestAnimationFrame(() => el.focus())}
        onKeyDown={handleKeyDown}
      >
        <div class="pn-dialog">
          <div class="pn-title">Unsaved Changes</div>
          <div class="pn-label" style={{ "margin-bottom": "16px" }}>
            You have unsaved changes. Do you want to save them?
          </div>
          <div class="pn-buttons">
            <button class="dlg-btn" onClick={() => resolveUnsavedDialog("cancel")}>Cancel</button>
            <button class="dlg-btn" onClick={() => resolveUnsavedDialog("discard")}>Don't Save</button>
            <button
              class="dlg-btn dlg-btn-primary"
              onClick={() => resolveUnsavedDialog("save")}
            >Save</button>
          </div>
        </div>
      </div>
    </Show>
  );
}

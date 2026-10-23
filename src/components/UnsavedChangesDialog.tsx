import { Show } from "solid-js";
import { unsavedDialogVisible, resolveUnsavedDialog } from "../lib/project-io";
import Modal from "./Modal";

export default function UnsavedChangesDialog() {
  return (
    <Show when={unsavedDialogVisible()}>
      <Modal overlayClass="pn-overlay" dialogClass="pn-dialog" titleId="unsaved-dlg-title"
        onEnter={() => resolveUnsavedDialog("save")} onEscape={() => resolveUnsavedDialog("cancel")}>
          <div class="pn-title" id="unsaved-dlg-title">Несохранённые изменения</div>
          <div class="pn-label" style={{ "margin-bottom": "16px" }}>
            Есть несохранённые изменения. Сохранить их?
          </div>
          <div class="pn-buttons">
            <button class="dlg-btn" onClick={() => resolveUnsavedDialog("cancel")}>Отмена</button>
            <button class="dlg-btn" onClick={() => resolveUnsavedDialog("discard")}>Не сохранять</button>
            <button
              class="dlg-btn dlg-btn-primary"
              onClick={() => resolveUnsavedDialog("save")}
            >Сохранить</button>
          </div>
      </Modal>
    </Show>
  );
}

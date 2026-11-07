import { Show } from "solid-js";
import { unsavedDialogVisible, resolveUnsavedDialog } from "../lib/project-io";
import Modal from "./Modal";
import { L } from "../lib/i18n";

export default function UnsavedChangesDialog() {
  return (
    <Show when={unsavedDialogVisible()}>
      <Modal overlayClass="pn-overlay" dialogClass="pn-dialog" titleId="unsaved-dlg-title"
        onEnter={() => resolveUnsavedDialog("save")} onEscape={() => resolveUnsavedDialog("cancel")}>
          <div class="pn-title" id="unsaved-dlg-title">{L("Несохранённые изменения", "Unsaved changes")}</div>
          <div class="pn-label" style={{ "margin-bottom": "16px" }}>
            {L("Есть несохранённые изменения. Сохранить их?", "There are unsaved changes. Save them?")}
          </div>
          <div class="pn-buttons">
            <button class="dlg-btn" onClick={() => resolveUnsavedDialog("cancel")}>{L("Отмена", "Cancel")}</button>
            <button class="dlg-btn" onClick={() => resolveUnsavedDialog("discard")}>{L("Не сохранять", "Don’t save")}</button>
            <button
              class="dlg-btn dlg-btn-primary"
              onClick={() => resolveUnsavedDialog("save")}
            >{L("Сохранить", "Save")}</button>
          </div>
      </Modal>
    </Show>
  );
}

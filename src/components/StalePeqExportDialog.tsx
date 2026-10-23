import { createSignal, Show, For } from "solid-js";
import Modal from "./Modal";

interface PendingState {
  bandNames: string[];
  resolve: (proceed: boolean) => void;
}

const [_pending, setPending] = createSignal<PendingState | null>(null);

/** Open the modal; resolves true if user clicks "Экспортировать всё равно". */
export function showStaleConfirmDialog(bandNames: string[]): Promise<boolean> {
  if (_pending()) {
    // Resolve any in-flight prompt with cancel before replacing.
    const prev = _pending();
    setPending(null);
    prev?.resolve(false);
  }
  return new Promise((resolve) => {
    setPending({ bandNames, resolve });
  });
}


function resolveAndClose(value: boolean) {
  const p = _pending();
  setPending(null);
  p?.resolve(value);
}

export default function StalePeqExportDialog() {
  return (
    <Show when={_pending()}>
      {(p) => (
        <Modal overlayClass="pn-overlay" dialogClass="pn-dialog"
          dialogStyle={{ "min-width": "440px", "max-width": "560px" }} titleId="stale-dlg-title"
          onEnter={() => resolveAndClose(true)} onEscape={() => resolveAndClose(false)}>
            <div class="pn-title" id="stale-dlg-title">Экспорт устаревшего PEQ</div>
            <div class="pn-label" style={{ "margin-bottom": "8px" }}>
              На следующих полосах PEQ устарел:
            </div>
            <ul style={{ margin: "0 0 12px 0", padding: "0 0 0 18px" }}>
              <For each={p().bandNames}>{(n) => <li>{n}</li>}</For>
            </ul>
            <div class="pn-label" style={{ "margin-bottom": "16px", color: "#aaa" }}>
              Target изменён после последней оптимизации. Экспорт даст коррекцию,
              не соответствующую текущему target.
            </div>
            <div class="pn-buttons">
              <button class="dlg-btn" onClick={() => resolveAndClose(false)}>Отмена</button>
              <button class="dlg-btn dlg-btn-primary" onClick={() => resolveAndClose(true)}>
                Экспортировать всё равно
              </button>
            </div>
        </Modal>
      )}
    </Show>
  );
}

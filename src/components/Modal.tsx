// b141.105 (audit 2026-10-23 §5.3/5.4/5.14): the one overlay every modal
// dialog renders through.
//
// - Escape, Enter and a backdrop press act on the TOPMOST dialog only, and
//   Escape stops there: a nested window (Baffle inside «Слияние NF+FF», the
//   restore question inside «Версии») no longer closes its parent too.
// - The backdrop closes on mousedown on the backdrop itself, so a drag that
//   starts in a field and ends outside does not close the dialog.
// - While any modal is open the app shortcuts are suspended (`anyModalOpen`).
// - role="dialog", aria-modal, aria-labelledby; Tab stays inside the dialog;
//   focus returns to where it was when the dialog closes.
import { onCleanup, onMount, type JSX } from "solid-js";
import { closeModal, handleDialogKeys, isTopModal, openModal, trapTab } from "../lib/dialog-keys";

export interface ModalProps {
  overlayClass: string;
  dialogClass: string;
  dialogStyle?: JSX.CSSProperties;
  /** id of the element that names the dialog (aria-labelledby). */
  titleId: string;
  onEscape?: () => void;
  onEnter?: () => void;
  /** Backdrop press; defaults to onEscape. `false` = the backdrop does nothing. */
  onBackdrop?: (() => void) | false;
  children: JSX.Element;
}

export default function Modal(props: ModalProps) {
  const token = openModal();
  const prevFocus = (typeof document !== "undefined" ? document.activeElement : null) as HTMLElement | null;
  let dialogEl!: HTMLDivElement;

  onMount(() => {
    // Dialogs that focus their own field do so in a rAF from a ref, which
    // runs before this one; otherwise the dialog itself takes focus so Enter
    // and Escape reach it.
    requestAnimationFrame(() => {
      if (dialogEl?.isConnected && !dialogEl.contains(document.activeElement)) dialogEl.focus();
    });
  });

  onCleanup(() => {
    closeModal(token);
    if (prevFocus && prevFocus !== document.body && prevFocus.isConnected) {
      try { prevFocus.focus(); } catch { /* element gone */ }
    }
  });

  const onKeyDown = (e: KeyboardEvent) => {
    if (!isTopModal(token)) return;
    if (trapTab(e, dialogEl)) return;
    if (e.key === "Escape" || e.key === "Enter") {
      handleDialogKeys(e, { onEnter: props.onEnter, onEscape: props.onEscape });
      if (e.key === "Escape") e.stopPropagation();
    }
  };

  const onMouseDown = (e: MouseEvent) => {
    if (e.target !== e.currentTarget || !isTopModal(token)) return;
    const act = props.onBackdrop === undefined ? props.onEscape : props.onBackdrop;
    if (act) act();
  };

  return (
    <div class={props.overlayClass} onMouseDown={onMouseDown} onKeyDown={onKeyDown}>
      <div
        ref={dialogEl}
        class={props.dialogClass}
        style={props.dialogStyle}
        role="dialog"
        aria-modal="true"
        aria-labelledby={props.titleId}
        tabIndex={-1}
      >
        {props.children}
      </div>
    </div>
  );
}

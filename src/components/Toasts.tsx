import { For } from "solid-js";
import { toasts, dismissToast } from "../lib/toast";

export default function Toasts() {
  return (
    <div
      style={{
        position: "fixed",
        right: "16px",
        bottom: "16px",
        display: "flex",
        "flex-direction": "column",
        gap: "8px",
        "z-index": 9999,
        "max-width": "420px",
        "pointer-events": "none",
      }}
    >
      <For each={toasts()}>
        {(t) => (
          <div
            style={{
              "pointer-events": "auto",
              padding: "10px 14px",
              background: t.kind === "warn" ? "#5a2a2a" : "#2a3a5a",
              color: "#eee",
              "border-radius": "4px",
              "box-shadow": "0 2px 8px rgba(0,0,0,0.5)",
              "font-size": "13px",
              cursor: "pointer",
            }}
            onClick={() => dismissToast(t.id)}
            title="Скрыть"
          >
            {t.text}
          </div>
        )}
      </For>
    </div>
  );
}

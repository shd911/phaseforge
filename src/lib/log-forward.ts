/**
 * b141.63: forward frontend diagnostics to the log file (Rust `log_frontend`,
 * src-tauri/src/applog.rs). Sent: every console.error / console.warn, every
 * console.log whose first argument starts with a "[TAG]" (the project's
 * diagnostic convention), uncaught errors and unhandled rejections. The
 * console keeps working as before. Rate-limited so a runaway loop cannot
 * flood the IPC: at most 100 lines per second, the excess is counted.
 */
import { invoke } from "@tauri-apps/api/core";

type Level = "error" | "warn" | "info";

function fmt(args: unknown[]): string {
  return args.map((a) => {
    if (typeof a === "string") return a;
    if (a instanceof Error) return `${a.name}: ${a.message}${a.stack ? "\n" + a.stack : ""}`;
    try { return JSON.stringify(a); } catch { return String(a); }
  }).join(" ");
}

export function installLogForwarding(): void {
  let windowStart = 0, sent = 0, dropped = 0;
  const send = (level: Level, args: unknown[]) => {
    const now = Date.now();
    if (now - windowStart > 1000) {
      if (dropped > 0) {
        invoke("log_frontend", { level: "warn", message: `[log] ${dropped} lines dropped (rate limit)` }).catch(() => {});
      }
      windowStart = now; sent = 0; dropped = 0;
    }
    if (++sent > 100) { dropped++; return; }
    invoke("log_frontend", { level, message: fmt(args) }).catch(() => {});
  };
  const orig = { error: console.error, warn: console.warn, log: console.log };
  console.error = (...a: unknown[]) => { orig.error(...a); send("error", a); };
  console.warn = (...a: unknown[]) => { orig.warn(...a); send("warn", a); };
  console.log = (...a: unknown[]) => {
    orig.log(...a);
    if (typeof a[0] === "string" && /^\[[^\]]+\]/.test(a[0])) send("info", a);
  };
  window.addEventListener("error", (e) => send("error", [`[uncaught] ${e.message} @ ${e.filename}:${e.lineno}`, e.error ?? ""]));
  window.addEventListener("unhandledrejection", (e) => send("error", ["[unhandled rejection]", e.reason]));
  console.log("[log] frontend → log file forwarding on");
}

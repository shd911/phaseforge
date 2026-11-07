//! b141.143: «Quit» from the Dock menu and ⌘Q must ask about unsaved changes.
//!
//! Both send `-[NSApplication terminate:]`. tao's app delegate does not answer
//! `applicationShouldTerminate:`, so AppKit terminates at once: no window
//! CloseRequested, no confirmation, unsaved edits lost. We add that method to
//! the delegate's class: while the close is not confirmed it cancels the
//! termination and routes it through the window-close flow — the frontend asks
//! (save / don't save / cancel) and then calls `close_window_now`, which closes
//! the last window and the app exits; that exit, if it comes back through
//! `terminate:`, now finds `allow_close` set and goes through.
use std::sync::atomic::Ordering;
use std::sync::OnceLock;

use tauri::{AppHandle, Emitter, Manager};

use crate::AppCloseState;

static APP: OnceLock<AppHandle> = OnceLock::new();

/// NSApplicationTerminateReply.
const TERMINATE_CANCEL: usize = 0;
const TERMINATE_NOW: usize = 1;

/// Decides one `terminate:` request. Pure in the close state — the AppKit hook
/// below only adapts it.
fn should_terminate(app: &AppHandle) -> bool {
    if app.state::<AppCloseState>().allow_close.load(Ordering::SeqCst) {
        return true;
    }
    match app.webview_windows().into_values().next() {
        // Same event as the window's close button.
        Some(window) => {
            // Dock Quit works on a minimised or hidden window too: bring it
            // up, or the question would be asked where nobody sees it.
            let _ = window.unminimize();
            let _ = window.show();
            let _ = window.set_focus();
            let _ = window.emit("request-close-confirm", ());
            false
        }
        // No window left — nothing to lose.
        None => true,
    }
}

#[cfg(target_os = "macos")]
unsafe extern "C-unwind" fn application_should_terminate(
    _this: *mut objc2::runtime::AnyObject,
    _cmd: objc2::runtime::Sel,
    _sender: *mut objc2::runtime::AnyObject,
) -> usize {
    let terminate = APP.get().map_or(true, should_terminate);
    tracing::info!("[QUIT] terminate: requested, {}", if terminate { "allowed" } else { "routed to the close confirmation" });
    if terminate { TERMINATE_NOW } else { TERMINATE_CANCEL }
}

/// Call once from `setup`, on the main thread, after the event loop exists.
pub fn install(app: &AppHandle) {
    let _ = APP.set(app.clone());
    #[cfg(target_os = "macos")]
    unsafe {
        use objc2::runtime::{AnyClass, AnyObject, Imp};
        use objc2::{class, msg_send, sel};
        let ns_app: *mut AnyObject = msg_send![class!(NSApplication), sharedApplication];
        let delegate: *mut AnyObject = msg_send![ns_app, delegate];
        if delegate.is_null() {
            tracing::warn!("[QUIT] no NSApplication delegate — Dock Quit is not guarded");
            return;
        }
        let cls = (*delegate).class() as *const AnyClass as *mut AnyClass;
        let imp: Imp = std::mem::transmute(
            application_should_terminate
                as unsafe extern "C-unwind" fn(*mut AnyObject, objc2::runtime::Sel, *mut AnyObject) -> usize,
        );
        let added = objc2::ffi::class_addMethod(cls, sel!(applicationShouldTerminate:), imp, c"Q@:@".as_ptr());
        if added.as_bool() {
            tracing::info!("[QUIT] Dock Quit / ⌘Q guarded by the unsaved-changes confirmation");
        } else {
            tracing::warn!("[QUIT] delegate already answers applicationShouldTerminate: — not replaced");
        }
    }
}

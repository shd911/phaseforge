//! b141.142: the UI language for messages worded in the backend (measurement
//! analysis, export and merge warnings). Set by the frontend on start and on
//! every switch; Russian until then, so tests read the Russian wording.
use std::sync::atomic::{AtomicBool, Ordering};

static EN: AtomicBool = AtomicBool::new(false);

pub fn en() -> bool { EN.load(Ordering::Relaxed) }

/// The string of the current language.
pub fn l(ru: &str, en: &str) -> String { if self::en() { en.into() } else { ru.into() } }

#[tauri::command]
pub fn set_ui_language(lang: String) { EN.store(lang == "en", Ordering::Relaxed); }

/// `tr!("… {:.0} Гц", "… {:.0} Hz", f)` — format! in the current language.
#[macro_export]
macro_rules! tr {
    ($ru:literal, $en:literal $(, $a:expr)* $(,)?) => {
        if $crate::i18n::en() { format!($en $(, $a)*) } else { format!($ru $(, $a)*) }
    };
}

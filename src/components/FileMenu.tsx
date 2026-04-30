import { createSignal, onMount, onCleanup, Show, For } from "solid-js";
import { invoke } from "@tauri-apps/api/core";
import {
  saveProject,
  saveProjectAs,
  loadProject,
  loadProjectFromPath,
  newProject,
  projectDir,
} from "../lib/project-io";
import { isDirty } from "../stores/bands";
import { openVersionsDialog } from "./VersionsDialog";

export default function FileMenu() {
  const [isOpen, setIsOpen] = createSignal(false);
  const [recentPaths, setRecentPaths] = createSignal<string[]>([]);
  const [showRecent, setShowRecent] = createSignal(false);

  let menuRef: HTMLDivElement | undefined;

  // Load recent projects list when menu opens
  async function loadRecent() {
    try {
      const paths = await invoke<string[]>("load_recent_projects");
      setRecentPaths(paths);
    } catch (e) {
      console.warn("Failed to load recent projects:", e);
      setRecentPaths([]);
    }
  }

  function toggleMenu() {
    const opening = !isOpen();
    setIsOpen(opening);
    setShowRecent(false);
    if (opening) loadRecent();
  }

  // Close on click outside
  function handleClickOutside(e: MouseEvent) {
    if (menuRef && !menuRef.contains(e.target as Node)) {
      setIsOpen(false);
      setShowRecent(false);
    }
  }
  onMount(() => document.addEventListener("mousedown", handleClickOutside));
  onCleanup(() => document.removeEventListener("mousedown", handleClickOutside));

  function closeMenu() {
    setIsOpen(false);
    setShowRecent(false);
  }

  async function handleNew() {
    closeMenu();
    await newProject();
  }

  async function handleOpen() {
    closeMenu();
    await loadProject();
  }

  async function handleSave() {
    closeMenu();
    await saveProject();
  }

  async function handleSaveAs() {
    closeMenu();
    await saveProjectAs();
  }

  async function handleOpenRecent(path: string) {
    closeMenu();
    try {
      await loadProjectFromPath(path);
    } catch (e) {
      console.error("Failed to open recent project:", e);
    }
  }

  async function handleClearRecent() {
    try {
      await invoke("clear_recent_projects");
      setRecentPaths([]);
    } catch (e) {
      console.warn("Failed to clear recent:", e);
    }
    setShowRecent(false);
  }

  const shortPath = (path: string) => {
    const parts = path.split("/");
    if (parts.length <= 3) return path;
    return ".../" + parts.slice(-2).join("/");
  };

  // Debounce for submenu hover
  let recentTimer: number | undefined;

  function onRecentEnter() {
    clearTimeout(recentTimer);
    setShowRecent(true);
  }

  function onRecentLeave() {
    recentTimer = window.setTimeout(() => setShowRecent(false), 150);
  }

  return (
    <div class="file-menu-dropdown" ref={menuRef}>
      <button
        class={`tb-btn file-menu-trigger ${isOpen() ? "active" : ""}`}
        onClick={toggleMenu}
      >
        File{isDirty() ? " \u2022" : ""}
      </button>

      <Show when={isOpen()}>
        <div class="file-menu">
          <button class="file-menu-item" onClick={handleNew}>
            <span class="file-menu-label">New Project</span>
            <span class="file-menu-shortcut">{"\u2318"}N</span>
          </button>

          <button class="file-menu-item" onClick={handleOpen}>
            <span class="file-menu-label">Open...</span>
            <span class="file-menu-shortcut">{"\u2318"}O</span>
          </button>

          {/* Recent Projects submenu */}
          <div
            class="file-menu-item file-menu-submenu-trigger"
            onMouseEnter={onRecentEnter}
            onMouseLeave={onRecentLeave}
          >
            <span class="file-menu-label">Recent Projects</span>
            <span class="file-menu-arrow">{"\u25B6"}</span>

            <Show when={showRecent()}>
              <div
                class="file-submenu"
                onMouseEnter={onRecentEnter}
                onMouseLeave={onRecentLeave}
              >
                <Show when={recentPaths().length > 0} fallback={
                  <span class="file-menu-item file-menu-disabled">No recent projects</span>
                }>
                  <For each={recentPaths()}>
                    {(path) => (
                      <button
                        class="file-menu-item"
                        onClick={() => handleOpenRecent(path)}
                        title={path}
                      >
                        {shortPath(path)}
                      </button>
                    )}
                  </For>
                  <div class="file-menu-sep" />
                  <button class="file-menu-item" onClick={handleClearRecent}>
                    Clear Recent
                  </button>
                </Show>
              </div>
            </Show>
          </div>

          <div class="file-menu-sep" />

          <button
            class="file-menu-item"
            onClick={handleSave}
            title="\u0421\u043E\u0445\u0440\u0430\u043D\u0438\u0442\u044C \u043F\u0440\u043E\u0435\u043A\u0442 (Cmd+S)"
          >
            <span class="file-menu-label">Save</span>
            <span class="file-menu-shortcut">{"\u2318"}S</span>
          </button>

          <button
            class="file-menu-item"
            onClick={handleSaveAs}
            title="\u0421\u043E\u0445\u0440\u0430\u043D\u0438\u0442\u044C \u043F\u0440\u043E\u0435\u043A\u0442 \u043F\u043E\u0434 \u043D\u043E\u0432\u044B\u043C \u0438\u043C\u0435\u043D\u0435\u043C (Cmd+Shift+S)"
          >
            <span class="file-menu-label">Save As...</span>
            <span class="file-menu-shortcut">{"\u21E7\u2318"}S</span>
          </button>

          <div class="file-menu-sep" />

          <button
            class="file-menu-item"
            onClick={() => { closeMenu(); openVersionsDialog(); }}
            disabled={projectDir() === null}
            title={projectDir() === null
              ? "\u0421\u043E\u0445\u0440\u0430\u043D\u0438\u0442\u0435 \u043F\u0440\u043E\u0435\u043A\u0442 \u0447\u0435\u0440\u0435\u0437 Save As, \u0447\u0442\u043E\u0431\u044B \u0441\u043E\u0437\u0434\u0430\u0432\u0430\u0442\u044C \u0432\u0435\u0440\u0441\u0438\u0438"
              : "\u0423\u043F\u0440\u0430\u0432\u043B\u0435\u043D\u0438\u0435 \u0438\u043C\u0435\u043D\u043E\u0432\u0430\u043D\u043D\u044B\u043C\u0438 \u0432\u0435\u0440\u0441\u0438\u044F\u043C\u0438 \u043F\u0440\u043E\u0435\u043A\u0442\u0430 (Cmd+Shift+V)"}
          >
            <span class="file-menu-label">{"\u0412\u0435\u0440\u0441\u0438\u0438\u2026"}</span>
            <span class="file-menu-shortcut">{"\u21E7\u2318"}V</span>
          </button>
        </div>
      </Show>
    </div>
  );
}

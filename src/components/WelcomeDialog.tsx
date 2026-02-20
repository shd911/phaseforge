import { createSignal, Show, For, onMount } from "solid-js";
import { invoke } from "@tauri-apps/api/core";
import { currentProjectPath, newProject, loadProject, loadProjectFromPath } from "../lib/project-io";

interface RecentProject {
  path: string;
  name: string;
}

export default function WelcomeDialog() {
  const [recentProjects, setRecentProjects] = createSignal<RecentProject[]>([]);

  onMount(async () => {
    try {
      const paths = await invoke<string[]>("load_recent_projects");
      const projects: RecentProject[] = paths.map((p) => {
        const parts = p.split("/");
        const filename = parts.pop() ?? "project.pfproj";
        return { path: p, name: filename.replace(/\.pfproj$/, "") };
      });
      setRecentProjects(projects);
    } catch (e) {
      console.warn("Failed to load recent projects:", e);
    }
  });

  return (
    <Show when={currentProjectPath() === null}>
      <div class="welcome-overlay">
        <div class="welcome-dialog">
          <img src="/logo.png" class="welcome-logo-img" alt="ClearWave Systems" />
          <div class="welcome-logo">PhaseForge</div>
          <div class="welcome-subtitle">DSP Room & Speaker Correction</div>

          <div class="welcome-actions">
            <button class="welcome-btn welcome-btn-primary" onClick={() => newProject()}>
              New Project
            </button>
            <button class="welcome-btn" onClick={() => loadProject()}>
              Open Project
            </button>
          </div>

          <Show when={recentProjects().length > 0}>
            <div class="welcome-recent">
              <div class="welcome-recent-title">Recent Projects</div>
              <For each={recentProjects()}>
                {(proj) => (
                  <button
                    class="welcome-recent-item"
                    onClick={() => loadProjectFromPath(proj.path)}
                    title={proj.path}
                  >
                    <span class="welcome-recent-name">{proj.name}</span>
                    <span class="welcome-recent-path">{proj.path}</span>
                  </button>
                )}
              </For>
            </div>
          </Show>
        </div>
      </div>
    </Show>
  );
}

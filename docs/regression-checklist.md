# Regression Checklist для b139.x этапов

После каждого этапа запускать соответствующий
`PhaseForge_<version>.dmg` (не dev-сервер) и проверить:

1. **New Project с двумя полосами** — Cmd+S сохраняет, Cmd+O открывает
   тот же проект, обе полосы и их target восстанавливаются.
2. **Импорт измерения** (любой `.txt`) — после успешного импорта
   автоматически появляется диалог анализа замера (b135).
3. **Gaussian HP=632, linear_phase=true, subsonic ON** — на SPL phase
   крутится в диапазоне 5–40 Гц, в 200–2000 Гц = 0.
4. **Gaussian HP=632, linear_phase=false, subsonic ON** — phase
   крутится в обеих зонах (Gaussian min-phase + Butterworth subsonic).
5. **Optimize PEQ** — все полученные полосы имеют Q ≤ `q_cap_at(freq)`
   по envelope b137 (12 на басу, 4 на высоких).
6. **Изменение HP freq** после Optimize → оранжевый банер
   «PEQ устарел: target изменён…» в PEQ-вкладке (b136).
7. **Cmd+Z после Optimize** возвращает PEQ-полосы и `peqOptimizedTarget`
   к pre-Optimize состоянию (b132).
8. **File → Versions → Save Version → Restore** — состояние снапшота
   полностью восстанавливается, индикатор `(modified)` появляется
   (b133).
9. **Cmd+Q при unsaved changes** — диалог Save / Don't Save / Cancel,
   каждая кнопка ведёт себя корректно (b131).
10. **FIR Export → файл .wav сохранён**, импульс непустой
    (визуально: при открытии файла в audio-tool виден ненулевой
    response, не тишина).

Все 10 пунктов должны проходить на каждом этапе b139.x. Любое
отклонение — diagnostic patch + точечный фикс, **не слепой**.

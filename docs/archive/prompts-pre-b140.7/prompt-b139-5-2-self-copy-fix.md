# Промт для Code: b139.5.2 — фикс самокопирования при импорте

**Тип:** критичный фикс потери данных. Bump до 0.1.0-b139.5.2.

## Контекст

`copy_file_to_project` в `src-tauri/src/project.rs:249` делает
`std::fs::copy(source, dest)` без проверки что source и dest — один
и тот же файл. Когда пользователь руками кладёт файл в
`<project>/inbox/` и потом импортирует через UI, выбирая тот же
файл — Rust сначала truncates destination, потом читает source
(который теперь пустой), пишет 0 байт. **Данные пользователя
теряются**.

Воспроизведено: пользователь положил `6.5M-FF.txt` (~26 КБ) в
inbox, импортировал → файл стал 0 байт.

## Что нужно сделать

### 1. Проверка source == dest в `copy_file_to_project`

В `src-tauri/src/project.rs`:

```rust
#[tauri::command]
pub fn copy_file_to_project(source_path: String, dest_path: String) -> Result<(), String> {
    let dest = std::path::Path::new(&dest_path);
    for component in dest.components() {
        if matches!(component, std::path::Component::ParentDir) {
            return Err("Invalid destination path: contains '..'".into());
        }
    }

    // Защита от self-copy: если source и dest указывают на один файл,
    // std::fs::copy сначала truncates destination, потом читает source
    // (теперь пустой) → данные теряются. Просто пропускаем копию.
    let src_canon = std::fs::canonicalize(&source_path)
        .map_err(|e| format!("Source not accessible: {e}"))?;
    let dst_path_buf = std::path::PathBuf::from(&dest_path);
    let dst_canon_opt = if dst_path_buf.exists() {
        std::fs::canonicalize(&dst_path_buf).ok()
    } else {
        // Destination ещё нет — собираем canonical через canonical parent.
        dst_path_buf.parent().and_then(|p| std::fs::canonicalize(p).ok())
            .and_then(|cp| dst_path_buf.file_name().map(|f| cp.join(f)))
    };

    if let Some(dst_canon) = dst_canon_opt {
        if src_canon == dst_canon {
            info!("copy_file_to_project: source == dest ({}), skipping", src_canon.display());
            return Ok(());
        }
    }

    info!("copy_file_to_project: {} -> {}", source_path, dest_path);
    std::fs::copy(&source_path, &dest_path)
        .map_err(|e| format!("Copy error: {e}"))?;
    Ok(())
}
```

### 2. Cargo тесты

```rust
#[test]
fn copy_file_to_project_handles_self_copy() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.txt");
    let mut f = std::fs::File::create(&path).unwrap();
    writeln!(f, "important data").unwrap();
    drop(f);
    let original_size = std::fs::metadata(&path).unwrap().len();
    assert!(original_size > 0);

    // Source == dest должно вернуть Ok без обнуления.
    let same = path.to_string_lossy().to_string();
    let result = copy_file_to_project(same.clone(), same.clone());
    assert!(result.is_ok(), "self-copy must not fail: {:?}", result.err());

    let after_size = std::fs::metadata(&path).unwrap().len();
    assert_eq!(after_size, original_size,
        "self-copy must not truncate: original={}, after={}", original_size, after_size);
}

#[test]
fn copy_file_to_project_normal_copy_works() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src.txt");
    let dst = dir.path().join("dst.txt");
    let mut f = std::fs::File::create(&src).unwrap();
    writeln!(f, "data").unwrap();
    drop(f);

    let result = copy_file_to_project(
        src.to_string_lossy().to_string(),
        dst.to_string_lossy().to_string(),
    );
    assert!(result.is_ok());
    assert!(dst.exists());
}
```

Если `tempfile` crate не подключён — добавить как dev-dependency, или
использовать `std::env::temp_dir()` + ручная очистка.

### 3. Audit других похожих мест

Запустить:
```
grep -rn "fs::copy" src-tauri/src/
```

Проверить **все** точки где `std::fs::copy` используется. Если
где-то ещё может случиться self-copy (например `copy_dir_contents` в
том же файле — копирование top-level файлов из source_dir в dest_dir
если они один dir), добавить аналогичную проверку.

### 4. Bump

- `src-tauri/tauri.conf.json` → `0.1.139` (numeric для MSI).
- `src-tauri/src/lib.rs` startup-лог: b139.5.2.
- `src/lib/version.ts` → `0.1.0-b139.5.2`.
- skill `build-version`.

## Acceptance

1. Cargo тест `copy_file_to_project_handles_self_copy` PASS.
2. Cargo тест `copy_file_to_project_normal_copy_works` PASS.
3. Все existing 165+ cargo тестов PASS.
4. Manual на `.dmg b139.5.2`: положить файл в `<project>/inbox/`
   руками, импортировать его через UI выбирая тот же путь → файл
   **не обнуляется**, импорт проходит.

## Что НЕ делать

- Не менять основную логику copy для разных файлов.
- Не блокировать import если destination уже существует с другими
  данными — это другой сценарий (обычное overwrite).

## Правила

- Один коммит: `fix: prevent data loss on self-copy in copy_file_to_project (b139.5.2)`
  + Co-Authored-By.
- 7-vector review.
- Cargo тесты обязательны.
- Без нарратива.

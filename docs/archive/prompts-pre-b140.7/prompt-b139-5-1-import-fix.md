# Промт для Code: b139.5.1 — фикс импорта файлов с нестандартными переводами строк

**Тип:** регрессионный фикс. Bump до 0.1.0-b139.5.1.

## Контекст

Пользователь импортирует REW measurement `.txt` который раньше
(b138.x и раньше) импортировался корректно, в b139.5 — отказывает.

Анализ файла: данные хранятся **на одной строке** без переводов
строк (CR-only line endings — классический Mac формат, либо файл
вообще без переводов).

## Pre-flight audit

Прочитать `src-tauri/src/io/parser.rs` целиком. Найти точку где
текстовый файл разбивается на строки. Скорее всего:

```rust
for line in content.lines() { ... }
// или
content.split('\n').for_each(...)
// или
BufReader::lines()
```

`str::lines()` в Rust **обрабатывает** LF и CRLF, но **НЕ**
обрабатывает CR-only (классический Mac). Если использовался `lines()`
— это и есть регрессия (либо она была всегда, но раньше попадались
только LF/CRLF файлы).

Запустить `git log -p src-tauri/src/io/parser.rs` за последние 30
коммитов — найти момент когда логика парсинга строк менялась
(возможно между b138 и b139.x).

## Что нужно сделать

### 1. Универсальный split по любым line endings

Перед основным циклом парсинга — нормализовать переводы строк:

```rust
let normalized = content.replace("\r\n", "\n").replace('\r', "\n");
for line in normalized.lines() { ... }
```

Это обрабатывает все три случая (LF, CRLF, CR) за один проход.
Memory overhead — одна копия строки, для measurement файла (~150 KB)
несущественно.

### 2. Cargo тест с CR-only fixture

Добавить в `src-tauri/src/io/parser.rs` (mod tests):

```rust
#[test]
fn parser_handles_cr_only_line_endings() {
    // Synthetic REW measurement file with CR-only line endings (Mac classic).
    let content = "* Measurement data\r\
                   * Freq(Hz) SPL(dB) Phase(degrees)\r\
                   20.0 47.5 78.2\r\
                   100.0 50.0 -10.5\r\
                   1000.0 48.3 0.0\r";

    let result = parse_measurement_text(content);  // или эквивалент
    assert!(result.is_ok(), "CR-only parser failed: {:?}", result.err());
    let m = result.unwrap();
    assert_eq!(m.freq.len(), 3);
    assert!((m.freq[0] - 20.0).abs() < 1e-6);
    assert!((m.magnitude[0] - 47.5).abs() < 1e-6);
}

#[test]
fn parser_handles_lf_line_endings() {
    let content = "* Measurement data\n\
                   * Freq(Hz) SPL(dB) Phase(degrees)\n\
                   20.0 47.5 78.2\n";
    let result = parse_measurement_text(content);
    assert!(result.is_ok());
}

#[test]
fn parser_handles_crlf_line_endings() {
    let content = "* Measurement data\r\n\
                   * Freq(Hz) SPL(dB) Phase(degrees)\r\n\
                   20.0 47.5 78.2\r\n";
    let result = parse_measurement_text(content);
    assert!(result.is_ok());
}
```

Если функция парсинга имеет другое имя или сигнатуру (принимает
путь, не строку) — адаптировать тесты соответственно. Либо
протестировать через временный файл, либо рефакторить — выделить
функцию принимающую `&str`.

### 3. TODO: sr=None — отдельной задачей

Из memory: при импорте `.txt` `sample_rate` приходит как `None`
(b135 диагностика). Это другой баг в том же файле, **не фиксить в
этом промте**. Записать в `docs/TODO.md` если ещё нет.

### 4. Bump

- `src-tauri/tauri.conf.json` → `0.1.139` (без pre-release suffix
  для совместимости с MSI, как в b139.5).
- `src-tauri/src/lib.rs` startup-лог: b139.5.1.
- `src/lib/version.ts` → `0.1.0-b139.5.1`.
- skill `build-version`.

## Acceptance

1. Cargo тесты `parser_handles_cr_only_line_endings`,
   `parser_handles_lf_line_endings`,
   `parser_handles_crlf_line_endings` — все PASS.
2. Все existing 162+ cargo тестов остаются PASS.
3. Vitest без изменений (фикс только в Rust).
4. Manual на `.dmg b139.5.1`: импорт реального файла
   `6.5M-FF.txt` (присланного пользователем) — работает.
5. Импорт обычных REW файлов (LF / CRLF) — без регрессий.

## Что НЕ делать

- Не менять структуру measurement (поля, типы).
- Не править sr=None — отдельная задача.
- Не трогать другие парсеры (FRD, ZMA если есть) — фикс только для
  REW `.txt`.

## Тестировать на `.dmg`

После сборки `PhaseForge_0.1.139_aarch64.dmg` — импортировать оба
файла:
- `6,5 FF.txt` (LF/CRLF, должен работать как раньше).
- `6.5M-FF.txt` (CR-only, раньше не работал, теперь должен).

Оба должны импортироваться без ошибки, кривая измерения нарисоваться
на SPL plot.

## Правила

- Один коммит: `fix: parser handles CR-only line endings (b139.5.1)`
  + Co-Authored-By.
- 7-vector review.
- Cargo тесты обязательны.
- Без нарратива.

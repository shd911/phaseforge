# Промт для Code: validate test-logs file workflow

**Тип:** workflow validation. Без bump, без коммита, без правок DSP/UI.

## Что нужно сделать

```
cd /Users/olegryzhikov/phaseforge
mkdir -p .test-logs
echo ".test-logs/" >> .gitignore.local 2>/dev/null || true
grep -q "^.test-logs/$" .gitignore || echo ".test-logs/" >> .gitignore

# Cargo lib tests
cd src-tauri
cargo test --lib > /Users/olegryzhikov/phaseforge/.test-logs/cargo-lib.log 2>&1
cargo test --test rephase_compare > /Users/olegryzhikov/phaseforge/.test-logs/rephase.log 2>&1

# Vitest
cd /Users/olegryzhikov/phaseforge
npx vitest run > /Users/olegryzhikov/phaseforge/.test-logs/vitest.log 2>&1

# Summary в один файл
cd /Users/olegryzhikov/phaseforge
{
  echo "=== Cargo lib ==="
  grep "test result" .test-logs/cargo-lib.log | tail -5
  echo ""
  echo "=== REPhase compare ==="
  grep "test result" .test-logs/rephase.log | tail -5
  echo ""
  echo "=== Vitest ==="
  grep -E "Test Files|Tests" .test-logs/vitest.log | head -5
  echo ""
  echo "=== Build version ==="
  grep "version" src-tauri/tauri.conf.json | head -1
} > /Users/olegryzhikov/phaseforge/.test-logs/summary.log

echo "Logs written:"
ls -la /Users/olegryzhikov/phaseforge/.test-logs/
```

После выполнения — сообщить "done" в чат. Cowork сам прочитает
`.test-logs/summary.log` и full files если нужно.

## Что НЕ делать

- Не запускать dev.
- Не commit-ить .gitignore (если изменился — проверим вручную).
- Не править ничего DSP / UI.

## Acceptance

- 4 файла в `.test-logs/`: cargo-lib.log, rephase.log, vitest.log, summary.log.
- Cowork прочтёт и validate workflow.

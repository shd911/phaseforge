# Промт для Code: re-run rephase_match tests с обновлёнными fixtures

User обновил test-fixtures/rephase/ — должны быть 4 уникальных файла
с правильными sr в RIFF header (44100, 48000, 88200, 176400).

## Что нужно сделать

### 1. Verify fixtures

```
cd /Users/olegryzhikov/phaseforge
ls -la test-fixtures/rephase/
md5 test-fixtures/rephase/*.wav
afinfo test-fixtures/rephase/44100.wav | grep rate
afinfo test-fixtures/rephase/48000.wav | grep rate
afinfo test-fixtures/rephase/88200.wav | grep rate
afinfo test-fixtures/rephase/176400.wav | grep rate
```

Должны быть 4 разных md5 и 4 разных rate (соответственно).

Если что-то не сходится — STOP, report что не так.

### 2. Re-run rephase_match tests

```
cd src-tauri && cargo test rephase_match -- --nocapture 2>&1 | tail -60
```

Прислать **полный** eprintln вывод (все freq на всех sr).

### 3. Если 4/4 PASS

Commit:
```
git add -A
git commit -m "$(cat <<'EOF'
test: REPhase reference comparison for HP min-phase IIR (b140.7.12)

Adds 4 tests comparing PhaseForge IIR path output against REPhase
reference WAVs at sr={44.1, 48, 88.2, 176.4k}. Reference WAVs in
test-fixtures/rephase/ (gitignored) generated with REPhase Min-Phase
HP=2000 Linkwitz-Riley 48 dB/oct, taps=65536, centered, hann.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

### 4. Если что-то FAIL

STOP, не commit. Прислать конкретные числа по failing sr (mag/phase
diff на каждой freq). Без гипотез — только данные.

### 5. End-of-prompt

```
osascript -e 'tell application "PhaseForge" to quit' 2>/dev/null || true
pkill -9 -f -i "phaseforge" 2>/dev/null || true
pkill -9 -f "tauri dev" 2>/dev/null || true
pkill -9 -f "tauri-driver" 2>/dev/null || true
sleep 1
lsof -ti:1420 | xargs kill -9 2>/dev/null || true
lsof -ti:5173 | xargs kill -9 2>/dev/null || true
sleep 1
cd /Users/olegryzhikov/phaseforge && nohup cargo tauri dev > /tmp/phaseforge-dev.log 2>&1 &
```

## Правила

- Без нарратива.
- Если PASS — commit.
- Если FAIL — числа.

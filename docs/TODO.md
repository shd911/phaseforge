# TODO

## sr=None в Measurement после import .txt

`measurement.sample_rate` приходит `None` для .txt-замеров — REW txt
parser не извлекает sample_rate, fallback нет. Влияет на:

- `analyze_measurement` (b135) — Nyquist оценивается по
  `sample_rate.unwrap_or(48000.0)`, но если файл реально 96 kHz, HF
  cliff детектор не дотянется до правильного диапазона.
- FIR export — sample rate берётся из export-настроек, но связь с
  measurement потенциально полезна для warning при mismatch.
- IR computation — `compute_impulse` принимает sample_rate явно, но
  при отсутствии в Measurement приходится класть дефолт.

Fix: посмотреть `src-tauri/src/io/parser.rs` (REW txt header), вынуть
`Sample Rate:` поле если есть. Если нет — fallback в parser на 48000
явно (а не молча оставлять None), чтобы downstream не сюрпризил.

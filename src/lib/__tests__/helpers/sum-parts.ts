/** b141.59: test-side stand-in for the Rust `compute_sum_impulse` — the
 *  coherent sum of the parts on the shared grid (delay ramp −360·f·τ, sign),
 *  returned as the (magnitude dB, phase °) pair the old single-spectrum
 *  `compute_impulse` call used to receive. */
export function sumParts(
  freq: number[],
  parts: { magnitude: number[]; phase: number[]; delay: number; sign: number }[],
): { magnitude: number[]; phase: number[] } {
  const magnitude: number[] = [], phase: number[] = [];
  for (let j = 0; j < freq.length; j++) {
    let re = 0, im = 0;
    for (const p of parts) {
      const a = p.sign * Math.pow(10, (p.magnitude[j] ?? -200) / 20);
      const ph = ((p.phase[j] ?? 0) - 360 * freq[j] * p.delay) * Math.PI / 180;
      re += a * Math.cos(ph);
      im += a * Math.sin(ph);
    }
    const amp = Math.hypot(re, im);
    magnitude.push(amp > 0 ? 20 * Math.log10(amp) : -200);
    phase.push(Math.atan2(im, re) * 180 / Math.PI);
  }
  return { magnitude, phase };
}

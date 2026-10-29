/** b141.111/115: reading of the ФВЧ/ФНЧ section ring (Rust
 *  `fir::section_ring::SectionRing`) for the FilterBlock line. */
export interface SectionRing { post60Ms: number; post100Ms: number; pre60Ms: number; pre100Ms: number; lowerBound: boolean }

export const fmtRingMs = (ms: number) => (ms < 10 ? ms.toFixed(1) : String(Math.round(ms)));

/** A linear Gaussian HP with subsonic protection is not symmetric (the
 *  protection is min-phase): then before and after are shown apart. */
export function ringSymmetric(r: SectionRing): boolean {
  const near = (a: number, b: number) => Math.abs(a - b) <= Math.max(0.05, 0.02 * Math.max(a, b));
  return near(r.pre60Ms, r.post60Ms) && near(r.pre100Ms, r.post100Ms);
}

/** A linear section rings on both sides of the file centre, each side has
 *  half the file; a min-phase one fills the half after the centre from its
 *  onset — rise to the peak plus the tail. */
export function ringTooLong(r: SectionRing, linear: boolean, halfFileMs: number): boolean {
  return linear
    ? Math.max(r.pre100Ms, r.post100Ms) > halfFileMs
    : r.pre100Ms + r.post100Ms > halfFileMs;
}

export function ringLabel(r: SectionRing, linear: boolean): string {
  const lb = r.lowerBound ? "≥ " : "";
  const pair = (a: number, b: number) => `${fmtRingMs(a)} / ${fmtRingMs(b)}`;
  if (linear && !ringSymmetric(r)) {
    return `Звон −60/−100 дБ: ${lb}до пика ${pair(r.pre60Ms, r.pre100Ms)}, после ${pair(r.post60Ms, r.post100Ms)} мс`;
  }
  return `Звон −60/−100 дБ: ${linear ? "до и после пика " : ""}${lb}${pair(r.post60Ms, r.post100Ms)} мс`;
}

export function ringTitle(r: SectionRing, linear: boolean, sampleRate: number, halfFileMs: number): string {
  const how = !linear
    ? "минимальная фаза звенит только после пика"
    : ringSymmetric(r)
      ? "линейная фаза звенит симметрично — столько же до пика, сколько после"
      : "линейная фаза звенит в обе стороны, минимально-фазовая защита от инфразвука добавляет хвост после пика";
  return `Звон фильтра на частоте экспорта ${sampleRate / 1000} кГц: ${how}. Половина файла экспорта: ${fmtRingMs(halfFileMs)} мс.`;
}

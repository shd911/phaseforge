/** b141.111/115: reading of the ФВЧ/ФНЧ section ring (Rust
 *  `fir::section_ring::SectionRing`) for the FilterBlock line. */
import { L } from "./i18n";

export interface SectionRing {
  post60Ms: number; post100Ms: number; pre60Ms: number; pre100Ms: number; lowerBound: boolean;
  /** b141.152: linear phase — first side lobe before the peak, dB re the peak;
   *  null on minimum phase or without lobes (Gaussian). */
  preLobeDb?: number | null;
}

/** b141.152: a first pre-ring lobe louder than this turns red. */
export const PRE_LOBE_LIMIT_DB = -60;

export const preLobeLoud = (r: SectionRing) => r.preLobeDb != null && r.preLobeDb > PRE_LOBE_LIMIT_DB;

export function preLobeLabel(r: SectionRing): string {
  return r.preLobeDb == null
    ? L("Предзвон: боковых лепестков нет", "Pre-ring: no side lobes")
    : L(`Предзвон: 1-й лепесток ${r.preLobeDb.toFixed(1).replace("-", "−")} дБ`,
      `Pre-ring: 1st lobe ${r.preLobeDb.toFixed(1).replace("-", "−")} dB`);
}

export const preLobeTitle = () =>
  L(`Уровень первого бокового лепестка перед пиком импульса, относительно пика (у ФВЧ — относительно отклонения от ровной АЧХ, как и звон). Он самый громкий в предзвоне; выше ${PRE_LOBE_LIMIT_DB} дБ предзвон может быть слышен — красный.`,
    `Level of the first side lobe before the impulse peak, re the peak (for an HP — re the deviation from flat, as the ringing). It is the loudest of the pre-ring; above ${PRE_LOBE_LIMIT_DB} dB the pre-ring may be audible — red.`);

export const fmtRingMs = (ms: number) => (ms < 10 ? ms.toFixed(1) : String(Math.round(ms)));

/** b141.116: a −60/−100 pair shares one precision — «9.8 / 12» mixed two
 *  and the line changed width as the corner moved across 10 ms. */
const fmtPair = (a: number, b: number) =>
  Math.min(a, b) < 10 ? `${a.toFixed(1)} / ${b.toFixed(1)}` : `${Math.round(a)} / ${Math.round(b)}`;

export const ringWarning = (halfFileMs: number) =>
  L(`⚠ Хвост длиннее ½ файла (${fmtRingMs(halfFileMs)} мс): звон обрежется или сдвинется задержка — увеличьте число отсчётов`,
    `⚠ Tail longer than ½ file (${fmtRingMs(halfFileMs)} ms): ringing gets cut or the delay shifts — raise the taps`);

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
  const pair = fmtPair;
  if (linear && !ringSymmetric(r)) {
    return L(`Звон −60/−100 дБ: ${lb}до пика ${pair(r.pre60Ms, r.pre100Ms)}, после ${pair(r.post60Ms, r.post100Ms)} мс`,
      `Ringing −60/−100 dB: ${lb}pre ${pair(r.pre60Ms, r.pre100Ms)}, post ${pair(r.post60Ms, r.post100Ms)} ms`);
  }
  return L(`Звон −60/−100 дБ: ${linear ? "до и после пика " : ""}${lb}${pair(r.post60Ms, r.post100Ms)} мс`,
    `Ringing −60/−100 dB: ${linear ? "each side " : ""}${lb}${pair(r.post60Ms, r.post100Ms)} ms`);
}

export function ringTitle(r: SectionRing, linear: boolean, sampleRate: number, halfFileMs: number): string {
  const how = !linear
    ? L("минимальная фаза звенит только после пика", "min phase rings only after the peak")
    : ringSymmetric(r)
      ? L("линейная фаза звенит симметрично — столько же до пика, сколько после", "lin phase rings symmetrically — as long before the peak as after")
      : L("линейная фаза звенит в обе стороны, минимально-фазовая защита от инфразвука добавляет хвост после пика",
        "lin phase rings both ways; the min-phase subsonic protection adds a tail after the peak");
  return L(`Звон фильтра на частоте экспорта ${sampleRate / 1000} кГц: ${how}. Половина файла экспорта: ${fmtRingMs(halfFileMs)} мс.`,
    `Filter ringing at the ${sampleRate / 1000} kHz export rate: ${how}. Half the export file: ${fmtRingMs(halfFileMs)} ms.`);
}

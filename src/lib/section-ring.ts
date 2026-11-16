/** b141.111/115: reading of the ФВЧ/ФНЧ section ring (Rust
 *  `fir::section_ring::SectionRing`) for the FilterBlock line. */
import { L } from "./i18n";

export interface SectionRing {
  post60Ms: number; post100Ms: number; pre60Ms: number; pre100Ms: number; lowerBound: boolean;
  /** b141.152: linear phase — first side lobe before the peak, dB re the peak;
   *  null on minimum phase or without lobes (Gaussian). */
  preLobeDb?: number | null;
  /** b141.153: linear phase — pre-ring over the hearing threshold, Rust
   *  `fir::prering_audibility` (relDb at the 1/2/5 ms masking windows, re a
   *  0 dB SPL peak; channelHz decides the 2 ms value; b141.155: the windows
   *  start at the main lobe's edge, edgeMs before the peak — null: the impulse
   *  has no pre-lobe, no pre-ring). */
  audibility?: { relDb: [number, number, number]; channelHz: number; edgeMs: number | null } | null;
}

/** b141.153 («5 полос»): green when the pre-ring stays 6 dB under threshold
 *  even with a 1 ms masking window, red when it is over threshold earlier
 *  than any documented backward masking of a click (5 ms); amber between —
 *  the verdict hangs on the unvalidated window. */
export const PRERING_GREEN_MARGIN_DB = -6;
/** Below this the masking window has no data (backward masking of clicks was
 *  measured from 250 Hz): red there is a risk on an isolated transient. */
export const PRERING_VALIDATED_HZ = 300;
export type PreRingVerdict = "ok" | "maybe" | "loud" | "none";

export interface PreRingMargin { m1: number; m2: number; m5: number; verdict: PreRingVerdict; channelHz: number; unvalidated: boolean; edgeMs: number | null }

export function preRingMargin(r: SectionRing, peakSplDb: number): PreRingMargin | null {
  const a = r.audibility;
  if (!a) return null;
  const [m1, m2, m5] = a.relDb.map((x) => x + peakSplDb);
  const edgeMs = a.edgeMs ?? null;
  if (edgeMs == null) return { m1, m2, m5, verdict: "none", channelHz: a.channelHz, unvalidated: false, edgeMs };
  const verdict: PreRingVerdict = m1 <= PRERING_GREEN_MARGIN_DB ? "ok" : m5 > 0 ? "loud" : "maybe";
  return { m1, m2, m5, verdict, channelHz: a.channelHz, unvalidated: verdict !== "ok" && a.channelHz < PRERING_VALIDATED_HZ, edgeMs };
}

const fmtDb = (v: number) => (v < -60 ? "< −60" : `${v > 0 ? "+" : ""}${v.toFixed(1)}`.replace("-", "−"));
const fmtHz = (f: number) => (f >= 1000 ? `${(f / 1000).toFixed(1)} ${L("кГц", "kHz")}` : `${Math.round(f)} ${L("Гц", "Hz")}`);

export function preRingLabel(m: PreRingMargin): string {
  if (m.edgeMs == null) return L("Предзвона нет: атака → групповая задержка", "No pre-ring: the attack → group delay");
  const from = (m.edgeMs + 2).toFixed(1);
  const what = m.verdict === "ok" ? L("не слышен", "inaudible") : m.verdict === "loud" ? L("слышен", "audible") : L("на грани", "borderline");
  return L(`Предзвон ${what}: ${fmtDb(m.m2)} дБ к порогу (2 мс после края главного лепестка, ${from} мс от пика)${m.unvalidated ? " *" : ""}`,
    `Pre-ring ${what}: ${fmtDb(m.m2)} dB re threshold (2 ms after the main lobe's edge, ${from} ms before the peak)${m.unvalidated ? " *" : ""}`);
}

export function preRingTitle(m: PreRingMargin, r: SectionRing, peakSplDb: number): string {
  if (m.edgeMs == null) {
    return L("До главного лепестка импульса нет ни одного бокового: сигнал не меняет знак до атаки, предзвона нет. Фронт главного лепестка — размытая атака; её оценивает групповая задержка (сдвиг главного лепестка между полосами), а не предзвон.",
      "The impulse has no side lobe before its main lobe: the signal never changes sign ahead of the attack — no pre-ring. The main lobe's front is a smeared attack; group delay (the main lobe's shift between bands) judges it, not pre-ring.");
  }
  const lobe = r.preLobeDb == null ? L("боковых лепестков нет", "no side lobes")
    : L(`первый лепесток ${fmtDb(r.preLobeDb)} дБ от пика`, `first lobe ${fmtDb(r.preLobeDb)} dB re the peak`);
  const note = m.unvalidated
    ? L(` * Ниже ${PRERING_VALIDATED_HZ} Гц маскировка щелчка не измерена: красный здесь — риск на одиночном транзиенте после тишины, решает слепой тест.`,
      ` * Below ${PRERING_VALIDATED_HZ} Hz backward masking of a click was never measured: red here is a risk on an isolated transient after silence; a blind test decides.`)
    : "";
  return L(`Насколько предзвон выше порога слышимости (ISO 226) в слуховом фильтре на ${fmtHz(m.channelHz)}, если пик у слушателя ${peakSplDb} дБ SPL (Настройки оптимизации). Учитывается энергия, пришедшая раньше окна обратной маскировки, отсчитанного от края главного лепестка (${m.edgeMs.toFixed(1)} мс до пика; фронт самого лепестка — атака, это групповая задержка): 1 мс ${fmtDb(m.m1)}, 2 мс ${fmtDb(m.m2)}, 5 мс ${fmtDb(m.m5)} дБ. Зелёный — даже при 1 мс на 6 дБ ниже порога; красный — выше порога ещё за 5 мс до края; жёлтый — зависит от окна маскировки. Худший случай: щелчок после тишины. Справочно: ${lobe}.${note}`,
    `How far the pre-ring is over the hearing threshold (ISO 226) in the ear's filter at ${fmtHz(m.channelHz)}, with a ${peakSplDb} dB SPL peak at the listener (Optimization settings). Energy arriving before the backward-masking window counts, the window measured from the main lobe's edge (${m.edgeMs.toFixed(1)} ms before the peak; the lobe's own front is the attack — group delay): 1 ms ${fmtDb(m.m1)}, 2 ms ${fmtDb(m.m2)}, 5 ms ${fmtDb(m.m5)} dB. Green — 6 dB under threshold even at 1 ms; red — over threshold 5 ms ahead of the edge; amber — depends on the masking window. Worst case: a click after silence. For reference: ${lobe}.${note}`);
}

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

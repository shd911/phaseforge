import { For, Show, batch, createEffect, createSignal, onCleanup, onMount } from "solid-js";
import {
  appState,
  exportSampleRate,
  setActiveBand,
  setAlignmentDelay,
  setBandReferenceLevel,
  toggleBandInverted,
} from "../stores/bands";
import type { BandState } from "../stores/bands";
import { peqStale } from "../stores/peq-optimize";
import { computeAutoAlign } from "../lib/auto-align";
import { attachFieldWheel } from "../lib/wheel-step";
import { clampNum } from "../lib/slope";
import { L, plural } from "../lib/i18n";
import { BandFilterBlock } from "./ControlPanel";

/** b141.146: the Σ view's editing surface — under the plot, left to right in
 *  frequency order: a column per band (curve visibility, level, polarity,
 *  delay, PEQ state) and between two bands the card of their crossover. The
 *  number of cards follows the bands. Everything writes through the same
 *  store setters as the band tabs; the crossover cards ARE the band panel's
 *  filter block (`BandFilterBlock`). */

export type SumCat = "target" | "measurement" | "corrected" | "corrfir";
export const SUM_CATS: readonly SumCat[] = ["target", "measurement", "corrected", "corrfir"];
const SIGMA = "Σ";

/** What the strip needs to know about a curve of the plot's legend. */
export interface SumLegendCell {
  visible: boolean;
  color: string;
  dash: boolean;
}

export interface SumStripProps {
  /** Curve visibility is offered on the SPL and IR/Step tabs only. */
  visEnabled: () => boolean;
  /** IR/Step tab: a cell holds two curves (impulse, step). */
  pairMode: () => boolean;
  cell: (col: string, cat: SumCat) => SumLegendCell | undefined;
  cellPair: (col: string, cat: SumCat) => { ir: SumLegendCell | undefined; step: SumLegendCell | undefined };
  toggleCell: (entry: SumLegendCell) => void;
  toggleColumn: (col: string) => void;
  toggleCategory: (cat: SumCat) => void;
  /** Seconds each band's WAV starts early (tail did not fit half the file). */
  convDelays: () => Record<string, number>;
  /** Client X of each linked crossover's point on the plot, by the id of the
   *  band whose low-pass it is; absent when the point is off the plot. */
  xoX: () => Record<string, number>;
  /** The crossover hovered or dragged on the plot (band id), and the way back. */
  hotXo: () => string | null;
  onHoverXo: (bandId: string | null) => void;
}

const catLabel = (cat: SumCat): string => ({
  target: L("Цель", "Target"),
  measurement: L("Замер", "Measured"),
  corrected: L("Коррекция", "Corrected"),
  corrfir: L("Через FIR", "Through FIR"),
}[cat]);

const catTitle = (cat: SumCat): string => ({
  target: L("Целевая кривая полосы с её фильтрами раздела. Клик — показать/скрыть у всех полос.",
    "The band's target curve with its crossover filters. Click — show/hide for every band."),
  measurement: L("Замер как есть, без коррекции. Клик — показать/скрыть у всех полос.",
    "The measurement as is, uncorrected. Click — show/hide for every band."),
  corrected: L("Замер + PEQ + фильтры раздела — расчётный результат коррекции. Клик — показать/скрыть у всех полос.",
    "Measurement + PEQ + crossover filters — the computed result of the correction. Click — show/hide for every band."),
  corrfir: L("Замер, пропущенный через экспортируемый FIR-файл: то, что реально сыграет конвольвер. Клик — показать/скрыть у всех полос.",
    "The measurement through the exported FIR file: what the convolver will actually play. Click — show/hide for every band."),
}[cat]);

/** A number field of the strip: typed or wheeled, written to the store 250 ms
 *  after the last change (one undo entry per pause, as the Σ delay did). */
function NumCell(props: {
  value: () => number;
  digits: number;
  step: number;
  min?: number;
  max?: number;
  label: string;
  commit: (v: number) => void;
}) {
  let timer: ReturnType<typeof setTimeout> | null = null;
  onCleanup(() => { if (timer) clearTimeout(timer); });
  const clamp = (v: number) => clampNum(v, props.min ?? -Infinity, props.max ?? Infinity);
  const later = (v: number) => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => { timer = null; props.commit(v); }, 250);
  };
  return (
    <input
      type="number"
      class="delay-input"
      step={props.step}
      aria-label={props.label}
      value={props.value().toFixed(props.digits)}
      onChange={(e) => {
        const v = parseFloat(e.currentTarget.value);
        if (Number.isFinite(v)) { const c = clamp(v); e.currentTarget.value = c.toFixed(props.digits); later(c); }
        else e.currentTarget.value = props.value().toFixed(props.digits);
      }}
      ref={(el) => {
        let draft: number | null = null;
        attachFieldWheel(el, {
          onSteps: (steps, coarse) => {
            const k = Math.pow(10, props.digits);
            draft = clamp(Math.round(((draft ?? props.value()) + steps * props.step * (coarse ? 10 : 1)) * k) / k);
            el.value = draft.toFixed(props.digits);
            later(draft);
          },
          onEnd: () => { draft = null; },
        });
      }}
    />
  );
}

export default function SumStrip(props: SumStripProps) {
  const bands = () => appState.bands;
  const [autoAligning, setAutoAligning] = createSignal(false);

  // ---- curve visibility cells (the plot's own legend entries) -------------
  const Swatch = (p: { e: SumLegendCell; dash: boolean; title: string }) => (
    <button class={`legend-item ${p.e.visible ? "" : "legend-off"}`} title={p.title} aria-pressed={p.e.visible}
      onClick={() => props.toggleCell(p.e)}>
      <span class={`legend-swatch ${p.dash ? "legend-swatch-dash" : ""}`}
        style={{ "background-color": p.dash ? "transparent" : p.e.color, "border-color": p.e.color }} />
    </button>
  );
  const VisCell = (p: { col: string; cat: SumCat }) => {
    const name = () => `${p.col} · ${catLabel(p.cat)}`;
    return (
      <div class="ss-cell">
        <Show when={props.visEnabled()} fallback={<span class="ss-none">·</span>}>
          <Show when={props.pairMode()} fallback={
            <Show when={props.cell(p.col, p.cat)} fallback={<span class="ss-none">·</span>}>
              {(e) => <Swatch e={e()} dash={e().dash} title={name()} />}
            </Show>
          }>
            <Show when={props.cellPair(p.col, p.cat).ir}>{(e) => <Swatch e={e()} dash={false} title={`${name()} · IR`} />}</Show>
            <Show when={props.cellPair(p.col, p.cat).step}>{(e) => <Swatch e={e()} dash={true} title={`${name()} · Step`} />}</Show>
            <Show when={!props.cellPair(p.col, p.cat).ir && !props.cellPair(p.col, p.cat).step}><span class="ss-none">·</span></Show>
          </Show>
        </Show>
      </div>
    );
  };

  // ---- auto-align (was the «AUTO» button of the Σ legend) ------------------
  const alignReady = () => {
    const xo = bands().filter((b) => b.measurement?.phase && b.measurement.phase.length > 0
      && b.targetEnabled && (b.target.high_pass || b.target.low_pass));
    return xo.length >= 2 && xo.every((b) => b.peqBands && b.peqBands.length > 0);
  };
  async function autoAlign() {
    if (!alignReady() || autoAligning()) return;
    const withPhase = bands().filter((b) => b.measurement?.phase && b.measurement.phase.length > 0);
    if (withPhase.length < 2) return;
    setAutoAligning(true);
    try {
      // Store bands go straight in (the evaluator snapshots; a clone would miss its cache).
      const result = await computeAutoAlign(withPhase as BandState[], exportSampleRate());
      batch(() => { for (const [id, delay] of Object.entries(result.delays)) setAlignmentDelay(id, delay); });
    } finally {
      setAutoAligning(false);
    }
  }

  // ---- which filters stand between band i and band i+1 ---------------------
  /** One editor for both sides: linked, and both sections exist. */
  const merged = (i: number) => {
    const a = bands()[i], b = bands()[i + 1];
    return !!a && !!b && a.linkedToNext && !!a.target.low_pass && !!b.target.high_pass;
  };

  // ---- leader lines: point on the plot → its card --------------------------
  let rowRef: HTMLDivElement | undefined;
  let svgRef: SVGSVGElement | undefined;
  const [leads, setLeads] = createSignal<{ id: string; x1: number; x2: number }[]>([]);
  let raf = 0;
  const measure = () => {
    raf = 0;
    if (!rowRef || !svgRef) return;
    const xs = props.xoX();
    const left = svgRef.getBoundingClientRect().left;
    const out: { id: string; x1: number; x2: number }[] = [];
    rowRef.querySelectorAll<HTMLElement>("[data-xo-band]").forEach((el) => {
      const id = el.dataset.xoBand!;
      const x = xs[id];
      if (x == null) return;
      const r = el.getBoundingClientRect();
      out.push({ id, x1: x - left, x2: r.left + r.width / 2 - left });
    });
    setLeads(out);
  };
  const schedule = () => { if (!raf) raf = requestAnimationFrame(measure); };
  createEffect(() => {
    props.xoX();
    // Card layout: a link or a section switched on/off moves the cards.
    for (const b of bands()) { void b.linkedToNext; void !!b.target.high_pass; void !!b.target.low_pass; void b.name; }
    schedule();
  });
  onMount(() => {
    if (!rowRef) return;
    // jsdom (tests) has no ResizeObserver; the scroll listener is enough there.
    const ro = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(schedule);
    ro?.observe(rowRef);
    rowRef.addEventListener("scroll", schedule, { passive: true });
    onCleanup(() => { ro?.disconnect(); rowRef?.removeEventListener("scroll", schedule); if (raf) cancelAnimationFrame(raf); });
  });

  const peqText = (b: BandState) => {
    const n = b.peqBands?.length ?? 0;
    if (n === 0) return b.target.high_pass || b.target.low_pass ? L("нет", "none") : "—";
    return `${n}${peqStale(b) ? " ⚠" : ""}`;
  };
  const peqTitle = (b: BandState) => {
    const n = b.peqBands?.length ?? 0;
    if (n === 0) return L("PEQ не оптимизирован. Клик — открыть полосу.", "PEQ not optimized. Click — open the band.");
    const what = `${n} ${plural(n, ["фильтр", "фильтра", "фильтров"], ["filter", "filters"])}`;
    return peqStale(b)
      ? L(`${what}, устарел: цель изменена после оптимизации. Клик — открыть полосу.`, `${what}, stale: the target changed since optimization. Click — open the band.`)
      : L(`${what}. Клик — открыть полосу.`, `${what}. Click — open the band.`);
  };

  return (
    <div class="sum-strip">
      <svg class="ss-leads" ref={svgRef} aria-hidden="true">
        <For each={leads()}>
          {(l) => <path d={`M${l.x1} 0 L${l.x1} 3 L${l.x2} 11 L${l.x2} 14`}
            classList={{ "ss-lead": true, "ss-lead-hot": props.hotXo() === l.id }} />}
        </For>
      </svg>
      <div class="ss-row" ref={rowRef}>
        {/* Row names — a click on a curve name toggles it for every band */}
        <div class="ss-col ss-labels">
          <div class="ss-cell ss-head" />
          <For each={SUM_CATS}>
            {(cat) => (
              <button class="ss-cell ss-label ss-label-btn" title={catTitle(cat)} disabled={!props.visEnabled()}
                onClick={() => props.toggleCategory(cat)}>{catLabel(cat)}</button>
            )}
          </For>
          <div class="ss-cell ss-label" title={L("Уровень цели полосы", "The band's target level")}>{L("Уровень, дБ", "Level, dB")}</div>
          <div class="ss-cell ss-label" title={L("Полярность полосы: NOR — нормальная, INV — инвертированная. Записывается в WAV.", "Band polarity: NOR — normal, INV — inverted. Written to the WAV.")}>{L("Полярность", "Polarity")}</div>
          <div class="ss-cell ss-label" title={L("Задержка выравнивания: плюс — полоса играет позже. Вводится в конвольвере, в WAV не пишется.", "Alignment delay: plus — the band plays later. Entered in the convolver, not written to the WAV.")}>{L("Задержка, мс", "Delay, ms")}</div>
          <div class="ss-cell ss-label">PEQ</div>
        </div>

        {/* Σ — the summed curves */}
        <div class="ss-col ss-sigma">
          <button class="ss-cell ss-head ss-head-btn" disabled={!props.visEnabled()}
            title={L("Суммарные кривые. Клик — показать/скрыть все.", "Summed curves. Click — show/hide all.")}
            onClick={() => props.toggleColumn(SIGMA)}>{SIGMA}</button>
          <For each={SUM_CATS}>{(cat) => <VisCell col={SIGMA} cat={cat} />}</For>
          <div class="ss-cell" />
          <div class="ss-cell" />
          <div class="ss-cell">
            <button class="auto-align-btn" disabled={!alignReady() || autoAligning()} aria-busy={autoAligning()}
              title={alignReady()
                ? L("Автоматически подобрать задержки полос", "Auto-fit band delays")
                : L("Нужна PEQ-оптимизация всех полос кроссовера", "All crossover bands need PEQ optimization")}
              style={{ cursor: autoAligning() ? "progress" : alignReady() ? "pointer" : "not-allowed", opacity: alignReady() ? 1 : 0.4 }}
              onClick={autoAlign}>
              <span class="btn-label-stack">
                <span classList={{ "btn-label-off": autoAligning() }}>{L("Авто", "Auto")}</span>
                <span classList={{ "btn-label-off": !autoAligning() }}>…</span>
              </span>
            </button>
          </div>
          <div class="ss-cell" />
        </div>

        {/* The first band's high-pass has no neighbour on its left */}
        <Show when={bands()[0]}>
          {(b) => <div class="ss-xo ss-edge"><BandFilterBlock bandId={b().id} side="hp" /></div>}
        </Show>

        <For each={bands()}>
          {(band, i) => (
            <>
              <div class="ss-col ss-band">
                <div class="ss-cell ss-head">
                  <button class="ss-band-name" title={L(`${band.name} — открыть полосу`, `${band.name} — open the band`)}
                    onClick={() => setActiveBand(band.id)}>
                    <span class="ss-dot" style={{ background: band.color }} />
                    <span class="ss-band-text">{band.name}</span>
                  </button>
                  <button class="ss-eye" disabled={!props.visEnabled()}
                    title={L(`Показать/скрыть все кривые полосы ${band.name}`, `Show/hide every curve of ${band.name}`)}
                    aria-label={L(`Показать/скрыть все кривые полосы ${band.name}`, `Show/hide every curve of ${band.name}`)}
                    onClick={() => props.toggleColumn(band.name)}>◉</button>
                </div>
                <For each={SUM_CATS}>{(cat) => <VisCell col={band.name} cat={cat} />}</For>
                <div class="ss-cell">
                  <NumCell value={() => band.target.reference_level_db ?? 0} digits={1} step={0.5} min={-20} max={20}
                    label={L(`${band.name}: уровень, дБ`, `${band.name}: level, dB`)}
                    commit={(v) => { if (v !== (band.target.reference_level_db ?? 0)) setBandReferenceLevel(band.id, v); }} />
                </div>
                <div class="ss-cell">
                  <button class={`fb-toggle invert-toggle ${band.inverted ? "on" : ""}`} aria-pressed={band.inverted}
                    aria-label={L(`${band.name}: полярность`, `${band.name}: polarity`)}
                    onClick={() => toggleBandInverted(band.id)}>{band.inverted ? "INV" : "NOR"}</button>
                </div>
                <div class="ss-cell">
                  <NumCell value={() => (band.alignmentDelay ?? 0) * 1000} digits={2} step={0.01}
                    label={L(`${band.name}: задержка, мс`, `${band.name}: delay, ms`)}
                    commit={(ms) => setAlignmentDelay(band.id, ms / 1000)} />
                  {/* The WAV shortfall keeps its slot, so the cell never changes width */}
                  <span class="delay-conv"
                    style={{ visibility: (props.convDelays()[band.id] ?? 0) >= 5e-6 ? "visible" : "hidden" }}
                    title={L("Добавьте к задержке в конвольвере: хвост фильтра этой полосы не уместился " +
                      "в половину файла, и его WAV начинается раньше остальных на это время. " +
                      `В конвольвер: ${(((band.alignmentDelay ?? 0) + (props.convDelays()[band.id] ?? 0)) * 1000).toFixed(2)} ms.`,
                      "Add to the convolver delay: this band's filter tail did not fit " +
                      "in half the file, so its WAV starts this much earlier than the others. " +
                      `Convolver: ${(((band.alignmentDelay ?? 0) + (props.convDelays()[band.id] ?? 0)) * 1000).toFixed(2)} ms.`)}
                  >+{((props.convDelays()[band.id] ?? 0) * 1000).toFixed(2)}</span>
                </div>
                <div class="ss-cell">
                  <button class="ss-peq" classList={{ "ss-peq-bad": (band.peqBands?.length ?? 0) === 0 && !!(band.target.high_pass || band.target.low_pass), "ss-peq-stale": peqStale(band) }}
                    title={peqTitle(band)} onClick={() => setActiveBand(band.id)}>{peqText(band)}</button>
                </div>
              </div>

              {/* Between this band and the next: their crossover */}
              <Show when={i() < bands().length - 1}>
                <Show when={merged(i())} fallback={
                  <div class="ss-xo ss-xo-split">
                    <BandFilterBlock bandId={band.id} side="lp" />
                    <BandFilterBlock bandId={bands()[i() + 1]?.id ?? ""} side="hp" />
                  </div>
                }>
                  <div class="ss-xo" data-xo-band={band.id} classList={{ "ss-xo-hot": props.hotXo() === band.id }}
                    onMouseEnter={() => props.onHoverXo(band.id)} onMouseLeave={() => props.onHoverXo(null)}>
                    <BandFilterBlock bandId={band.id} side="lp" title={L("Раздел", "Crossover")} />
                  </div>
                </Show>
              </Show>
            </>
          )}
        </For>

        {/* The last band's low-pass has no neighbour on its right */}
        <Show when={bands().length > 0}>
          <div class="ss-xo ss-edge"><BandFilterBlock bandId={bands()[bands().length - 1].id} side="lp" /></div>
        </Show>
      </div>
    </div>
  );
}

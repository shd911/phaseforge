// b141.37: Export metrics bar. Each metric opens a drop-down with what it
// measures and how to read it — the pre-ring number used to mix ringing
// with the truncation floor, and nothing on screen said what it counted.
import { createSignal, For, Show, onCleanup, onMount, type JSX } from "solid-js";
import type { ExportMetrics } from "../stores/bands";
import { STATUS_GOOD, STATUS_WARN, STATUS_BAD } from "../lib/plot-helpers";
import { FLOOR_BAD_DB, FLOOR_GOOD_DB, FLOOR_MARGIN_DB, PRE_RING_DB } from "../lib/export-metrics";
import { L, isEn, plural } from "../lib/i18n";

const POPOVER_W = 340;
/** Below this much room under the label the popover opens upward. */
const POPOVER_MIN_H = 240;

function fmtHz(hz: number): string {
  return hz >= 1000
    ? `${(hz / 1000).toFixed(hz >= 10000 ? 1 : 2)} ${L("кГц", "kHz")}`
    : `${Math.round(hz)} ${L("Гц", "Hz")}`;
}
const fmtDb = (db: number) => (db <= -150 ? "≤ −150" : db.toFixed(0).replace("-", "−"));

interface Item {
  id: string;
  label: () => JSX.Element;
  color?: () => string | undefined;
  show?: () => boolean;
  title: () => string;
  body: () => JSX.Element;
}

export default function ExportMetricsBar(props: { m: ExportMetrics }) {
  const m = () => props.m;
  const [open, setOpen] = createSignal<string | null>(null);
  // Fixed position from the clicked label: the plot containers clip overflow.
  // b141.119 (UI audit §9): opens upward when the room below is short — the
  // bar sits at the bottom of the plot and the text shrank to a few lines.
  const [pos, setPos] = createSignal({ top: 0, bottom: 0, left: 0, up: false });
  let root: HTMLDivElement | undefined;

  const onDown = (e: MouseEvent) => {
    if (root && !root.contains(e.target as Node)) setOpen(null);
  };
  const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(null); };
  const close = () => setOpen(null);
  onMount(() => {
    document.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    window.addEventListener("resize", close);
  });
  onCleanup(() => {
    document.removeEventListener("mousedown", onDown);
    window.removeEventListener("keydown", onKey);
    window.removeEventListener("resize", close);
  });

  const toggle = (id: string, el: HTMLElement) => {
    if (open() === id) { setOpen(null); return; }
    const r = el.getBoundingClientRect();
    const below = window.innerHeight - r.bottom - 14;
    const above = r.top - 14;
    setPos({
      top: r.bottom + 6,
      bottom: window.innerHeight - r.top + 6,
      left: Math.max(8, Math.min(r.left, window.innerWidth - POPOVER_W - 8)),
      up: below < POPOVER_MIN_H && above > below,
    });
    setOpen(id);
  };

  const floorColor = () => {
    const f = m().floorDb;
    if (f == null) return STATUS_WARN;
    return f <= FLOOR_GOOD_DB ? STATUS_GOOD : f <= FLOOR_BAD_DB ? STATUS_WARN : STATUS_BAD;
  };
  const zoneText = () => m().ringZoneHz == null
    ? L(`${m().ringZoneMs.toFixed(0)} мс (у полосы нет ФВЧ/ФНЧ)`,
        `${m().ringZoneMs.toFixed(0)} ms (band has no HP/LP)`)
    : L(`${m().ringZoneMs.toFixed(1)} мс — по ${m().ringZoneLinear ? "linear-phase " : ""}фильтру ${fmtHz(m().ringZoneHz!)} и его крутизне`,
        `${m().ringZoneMs.toFixed(1)} ms — from the ${m().ringZoneLinear ? "linear-phase " : ""}filter at ${fmtHz(m().ringZoneHz!)} and its slope`);
  const band = () => `${fmtHz(m().passbandLoHz)} – ${fmtHz(m().passbandHiHz)}`;

  const items: Item[] = [
    {
      id: "causal",
      // b141.123 (external audit 2026-10-29): no traffic light at all. b141.119
      // graded min-phase bands against ~100 %, but a steep section's peak comes
      // after its rise: a correct LR4 band reads 42–72 %, and it went red.
      label: () => <>{L("После пика", "After peak")}: {m().causality}%</>,
      title: () => L("Энергия на пике и после него", "Energy at and after the peak"),
      body: () => isEn() ? <>
        <p>The share of the impulse energy at the peak and after it.</p>
        <p>A reference number, not colour-graded. Linear-phase — about 50%: the impulse
          is symmetric. Min-phase — 40 to 100%: a steep filter peaks after its rise
          begins, and part of the energy rightly falls before the peak
          (LR4 in its passband — 40–70%, HP at 30 Hz — 100%).</p>
        <p>Pre-ringing and floor, which also put energy before the peak, are shown
          separately: “Pre-ring” and “Floor”.</p>
      </> : <>
        <p>Доля энергии импульса, которая приходится на пик и время после него.</p>
        <p>Справочное число, цветом не оценивается. Linear-phase — около 50%: импульс
          симметричен. Min-phase — от 40 до 100%: у крутого фильтра пик наступает позже
          начала нарастания, и часть энергии законно приходится на время до пика
          (LR4 в рабочей полосе — 40–70%, ФВЧ на 30 Гц — 100%).</p>
        <p>Предзвон и фон, которые тоже дают энергию до пика, показаны отдельно:
          «Предзвон» и «Фон».</p>
      </>,
    },
    {
      id: "prering",
      show: () => m().preRingMs > 0 || m().preRingLimited,
      label: () => <>{L("Предзвон", "Pre-ring")}: {m().preRingLimited ? "≥ " : ""}{m().preRingMs} {L("мс", "ms")}</>,
      color: () => m().preRingLimited ? STATUS_WARN : undefined,
      title: () => L("Предзвон", "Pre-ringing"),
      body: () => isEn() ? <>
        <p>Time from the moment the impulse first rises above {PRE_RING_DB} dB
          re the peak, to the peak itself.</p>
        <p>It comes from the band's linear-phase HP/LP and is set by their frequency and
          slope. It does not depend on the tap count: a longer filter does not shorten it.
          Min-phase bands do not ring before the peak — here they show only their rise time.</p>
        <p>Ringing zone: {zoneText()}. Anything farther from the peak is floor, not ringing.</p>
        <Show when={m().preRingLimited}>
          <p style={{ color: STATUS_WARN }}>The floor ({fmtDb(m().floorDb!)} dB) is louder than the threshold,
            so the threshold is raised to {fmtDb(m().preRingThresholdDb)} dB (floor + {FLOOR_MARGIN_DB} dB).
            Part of the ringing hides under the floor — the true pre-ringing is at least the value shown.
            Increase taps until the “≥” goes away.</p>
        </Show>
      </> : <>
        <p>Время от момента, когда импульс впервые поднимается выше {PRE_RING_DB} дБ
          от пика, до самого пика.</p>
        <p>Его создают linear-phase ФВЧ/ФНЧ полосы, и он определяется их частотой и
          крутизной. От числа отсчётов он не зависит: длинный фильтр его не укорачивает.
          Min-phase полосы не звенят до пика — у них здесь только время нарастания.</p>
        <p>Зона звона: {zoneText()}. Всё, что дальше от пика, — не звон, а фон.</p>
        <Show when={m().preRingLimited}>
          <p style={{ color: STATUS_WARN }}>Фон ({fmtDb(m().floorDb!)} дБ) громче порога,
            поэтому порог поднят до {fmtDb(m().preRingThresholdDb)} дБ (фон + {FLOOR_MARGIN_DB} дБ).
            Часть звона под фоном не видна — настоящий предзвон не меньше показанного.
            Увеличьте отсчёты, пока «≥» не пропадёт.</p>
        </Show>
      </>,
    },
    {
      id: "floor",
      label: () => <>{L("Фон", "Floor")}: {m().floorDb == null ? "—" : `${fmtDb(m().floorDb!)} ${L("дБ", "dB")}`}</>,
      color: floorColor,
      title: () => L("Фон до пика", "Floor before the peak"),
      body: () => isEn() ? <>
        <p>The loudest level of the impulse before the peak outside the ringing zone
          (more than {m().ringZoneMs.toFixed(1)} ms from the peak), in dB re the peak.</p>
        <p>The band's filters no longer ring there. What remains did not fit the FIR length:
          long tails of LF correction and narrow PEQ. Not fitting after the peak, they
          wrap to the start of the filter and land before the peak. This floor is written to the WAV.</p>
        <p>
          <span style={{ color: STATUS_GOOD }}>below {FLOOR_GOOD_DB} dB</span> — the tails fit;{" "}
          <span style={{ color: STATUS_WARN }}>{FLOOR_GOOD_DB}…{FLOOR_BAD_DB} dB</span> — borderline;{" "}
          <span style={{ color: STATUS_BAD }}>above {FLOOR_BAD_DB} dB</span> — the filter is too short for
          this correction.
        </p>
        <p>Fixed by more taps. Time matters, not the count: at 352.8 kHz you need
          7.35 times more taps than at 48 kHz.</p>
        <Show when={m().floorDb == null}>
          <p style={{ color: STATUS_WARN }}>The filter has less than {m().ringZoneMs.toFixed(1)} ms before the peak —
            the floor cannot be told from ringing. Increase taps.</p>
        </Show>
      </> : <>
        <p>Самый громкий уровень импульса до пика за пределами зоны звона
          (дальше {m().ringZoneMs.toFixed(1)} мс от пика), в дБ от пика.</p>
        <p>Фильтры полосы там уже не звенят. Остаётся то, что не уместилось в длину FIR:
          долгие хвосты НЧ-коррекции и узких PEQ. Не поместившись после пика, они
          заворачиваются в начало фильтра и ложатся перед пиком. Этот фон записывается в WAV.</p>
        <p>
          <span style={{ color: STATUS_GOOD }}>ниже {FLOOR_GOOD_DB} дБ</span> — хвосты уместились;{" "}
          <span style={{ color: STATUS_WARN }}>{FLOOR_GOOD_DB}…{FLOOR_BAD_DB} дБ</span> — пограничный;{" "}
          <span style={{ color: STATUS_BAD }}>выше {FLOOR_BAD_DB} дБ</span> — фильтр короток для
          этой коррекции.
        </p>
        <p>Лечится увеличением отсчётов. Важно время, а не число: при 352.8 кГц отсчётов
          нужно в 7.35 раза больше, чем при 48 кГц.</p>
        <Show when={m().floorDb == null}>
          <p style={{ color: STATUS_WARN }}>До пика в фильтре меньше {m().ringZoneMs.toFixed(1)} мс —
            фон нельзя отделить от звона. Увеличьте отсчёты.</p>
        </Show>
      </>,
    },
    {
      id: "magerr",
      label: () => <>{L("Ошибка АЧХ", "Mag error")}: {m().maxMagErr} {L("дБ", "dB")}</>,
      color: () => m().maxMagErr <= 0.5 ? STATUS_GOOD : m().maxMagErr <= 1.5 ? STATUS_WARN : STATUS_BAD,
      title: () => L("Ошибка АЧХ", "Magnitude error"),
      body: () => isEn() ? <>
        <p>The largest magnitude deviation of the finished FIR from the model (target + PEQ)
          in the passband {band()}.</p>
        <p>Grows when the filter is too short for steep slopes and LF correction.
          Guide: up to 0.5 dB — good, up to 1.5 dB — acceptable.</p>
      </> : <>
        <p>Наибольшее расхождение АЧХ готового FIR с моделью (цель + PEQ) в полосе
          пропускания {band()}.</p>
        <p>Растёт, когда длины фильтра не хватает на крутые склоны и НЧ-коррекцию.
          Ориентиры: до 0.5 дБ — хорошо, до 1.5 дБ — терпимо.</p>
      </>,
    },
    {
      id: "gd",
      // b141.119 (UI audit §4): a span, not an error — no traffic light.
      label: () => <>{L("Размах ГЗ", "GD span")}: {m().gdRippleMs} {L("мс", "ms")}</>,
      title: () => L("Размах групповой задержки", "Group delay span"),
      body: () => isEn() ? <>
        <p>The difference between the largest and smallest group delay of the FIR in the
          passband {band()}.</p>
        <p>Min-phase correction and PEQ rightly raise the delay toward low frequencies — that
          counts too. So the metric is more useful for comparing variants of one
          band (taps, window) than as an absolute score.</p>
      </> : <>
        <p>Разница между наибольшей и наименьшей групповой задержкой FIR в полосе
          пропускания {band()}.</p>
        <p>Min-phase коррекция и PEQ законно дают рост задержки к низким частотам — это
          тоже войдёт в число. Поэтому метрика полезнее для сравнения вариантов одной
          полосы (отсчёты, окно), чем как абсолютная оценка.</p>
      </>,
    },
    {
      id: "uslp",
      show: () => m().ultrasonicLpHz != null,
      label: () => <>{L("УЗ-ФНЧ", "US LP")}: {fmtHz(m().ultrasonicLpHz ?? 0)}</>,
      title: () => L("Ультразвуковой ФНЧ", "Ultrasonic low-pass"),
      body: () => isEn() ? <>
        <p>At export rates from 88.2 kHz every FIR has a built-in 8th-order Butterworth
          low-pass (48 dB/oct) at {fmtHz(m().ultrasonicLpHz ?? 0)} with zero phase.</p>
        <p>It limits ultrasound that would otherwise reach the tweeter and amplifier:
          noise bandwidth above 24 kHz is 6.6 kHz vs 17 kHz for the old 40 kHz cutoff
          and 161 kHz for the unlimited analytical route (at 352.8 kHz). The worst output
          peak is ~4 dB lower, and the 40 kHz cutoff's ringing no longer enters the pre-ringing.</p>
        <p>Zero phase adds no delay: no band shifts, summation
          and alignment are unchanged. Loss up to 20 kHz is 0.007 dB; on the plot the FIR
          rolls off smoothly at the right edge (−3 dB at {fmtHz(m().ultrasonicLpHz ?? 0)}).</p>
      </> : <>
        <p>При частоте экспорта от 88.2 кГц в каждый FIR встроен ФНЧ Баттерворта
          8-го порядка (48 дБ/окт) на {fmtHz(m().ultrasonicLpHz ?? 0)} с нулевой фазой.</p>
        <p>Он ограничивает ультразвук, который иначе уходит в твитер и усилитель:
          шумовая полоса выше 24 кГц — 6.6 кГц против 17 кГц у прежнего обрыва на 40 кГц
          и 161 кГц у аналитического пути без ограничения (при 352.8 кГц). Худший пик
          на выходе ниже на ~4 дБ, звон обрыва на 40 кГц больше не попадает в предзвон.</p>
        <p>Нулевая фаза не вносит задержку: ни одна полоса не сдвигается, сведение
          и выравнивание не меняются. В полосе до 20 кГц потеря 0.007 дБ; на графике FIR
          плавно уходит вниз у правого края (−3 дБ на {fmtHz(m().ultrasonicLpHz ?? 0)}).</p>
      </>,
    },
    {
      id: "norm",
      label: () => <>{L("Нормировка", "Norm")}: {m().normDb.toFixed(1)} {L("дБ", "dB")}</>,
      title: () => L("Нормировка", "Normalization"),
      body: () => isEn() ? <>
        <p>How much the filter is attenuated so the file's magnitude peaks at 0 dB: target
          and PEQ boosts alone do not drive the file past full scale. This does not replace
          input headroom.</p>
        <p>Each band is normalized separately, by design. Levels between bands
          are set in the player or DSP host.</p>
      </> : <>
        <p>Насколько ослаблен фильтр, чтобы максимум АЧХ файла был 0 дБ: подъёмы
          цели и PEQ сами по себе не выводят файл за полную шкалу. Запас по уровню
          на входе сигнала это не заменяет.</p>
        <p>Каждая полоса нормируется отдельно, так задумано. Уровни полос между собой
          выставляются в плеере или DSP-хосте.</p>
      </>,
    },
  ];

  return (
    <div ref={root} class="export-metrics" style={{
      display: "flex", "flex-wrap": "wrap", gap: "6px 14px",
      padding: "3px 8px", "font-size": "var(--fs-sm)", color: "#b0b0bc",
      "border-top": "1px solid #2a2a35",
    }}>
      <span>{m().taps} {plural(m().taps, ["отсчёт", "отсчёта", "отсчётов"], ["tap", "taps"])}</span>
      <span>{m().sampleRate / 1000} {L("кГц", "kHz")}</span>
      <span>{m().window}</span>
      <span>{m().phaseLabel}</span>
      <For each={items.slice(0, -2)}>{(it) => <MetricItem it={it} />}</For>
      <Show when={m().peqCount > 0}>
        <span>PEQ: {m().peqCount}</span>
      </Show>
      <For each={items.slice(-2)}>{(it) => <MetricItem it={it} />}</For>
    </div>
  );

  function MetricItem(p: { it: Item }) {
    const it = p.it;
    return (
      <Show when={it.show ? it.show() : true}>
        <span>
          <span
            role="button"
            tabIndex={0}
            onClick={(e) => toggle(it.id, e.currentTarget)}
            onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(it.id, e.currentTarget); } }}
            style={{
              color: it.color?.(), cursor: "help",
              "border-bottom": "1px dotted currentColor",
            }}
          >
            {it.label()}
          </span>
          <Show when={open() === it.id}>
            <div style={{
              position: "fixed", left: `${pos().left}px`,
              ...(pos().up ? { bottom: `${pos().bottom}px` } : { top: `${pos().top}px` }),
              width: `min(${POPOVER_W}px, calc(100vw - 16px))`, "z-index": 1000,
              "max-height": pos().up
                ? `calc(100vh - ${pos().bottom + 8}px)`
                : `calc(100vh - ${pos().top + 8}px)`,
              "overflow-y": "auto",
              background: "var(--bg-surface)", border: "1px solid var(--border)",
              "border-radius": "6px", padding: "10px 12px",
              "box-shadow": "0 8px 24px rgba(0,0,0,0.5)",
              color: "var(--text-primary)", "font-size": "12px", "line-height": "1.45",
              "white-space": "normal",
            }}>
              <div style={{ "font-weight": 600, "margin-bottom": "6px" }}>{it.title()}</div>
              <div class="metric-popover-body">{it.body()}</div>
            </div>
          </Show>
        </span>
      </Show>
    );
  }
}

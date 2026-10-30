// b141.37: Export metrics bar. Each metric opens a drop-down with what it
// measures and how to read it — the pre-ring number used to mix ringing
// with the truncation floor, and nothing on screen said what it counted.
import { createSignal, For, Show, onCleanup, onMount, type JSX } from "solid-js";
import type { ExportMetrics } from "../stores/bands";
import { STATUS_GOOD, STATUS_WARN, STATUS_BAD } from "../lib/plot-helpers";
import { FLOOR_BAD_DB, FLOOR_GOOD_DB, FLOOR_MARGIN_DB, PRE_RING_DB } from "../lib/export-metrics";
import { pluralRu } from "../lib/plural";

const POPOVER_W = 340;
/** Below this much room under the label the popover opens upward. */
const POPOVER_MIN_H = 240;

function fmtHz(hz: number): string {
  return hz >= 1000 ? `${(hz / 1000).toFixed(hz >= 10000 ? 1 : 2)} кГц` : `${Math.round(hz)} Гц`;
}
const fmtDb = (db: number) => (db <= -150 ? "≤ −150" : db.toFixed(0).replace("-", "−"));

interface Item {
  id: string;
  label: () => JSX.Element;
  color?: () => string | undefined;
  show?: () => boolean;
  title: string;
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
    ? `${m().ringZoneMs.toFixed(0)} мс (у полосы нет ФВЧ/ФНЧ)`
    : `${m().ringZoneMs.toFixed(1)} мс — по ${m().ringZoneLinear ? "linear-phase " : ""}фильтру ${fmtHz(m().ringZoneHz!)} и его крутизне`;
  const band = () => `${fmtHz(m().passbandLoHz)} – ${fmtHz(m().passbandHiHz)}`;

  const items: Item[] = [
    {
      id: "causal",
      // b141.123 (external audit 2026-10-29): no traffic light at all. b141.119
      // graded min-phase bands against ~100 %, but a steep section's peak comes
      // after its rise: a correct LR4 band reads 42–72 %, and it went red.
      label: () => <>После пика: {m().causality}%</>,
      title: "Энергия на пике и после него",
      body: () => <>
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
      label: () => <>Предзвон: {m().preRingLimited ? "≥ " : ""}{m().preRingMs} мс</>,
      color: () => m().preRingLimited ? STATUS_WARN : undefined,
      title: "Предзвон",
      body: () => <>
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
      label: () => <>Фон: {m().floorDb == null ? "—" : `${fmtDb(m().floorDb!)} дБ`}</>,
      color: floorColor,
      title: "Фон до пика",
      body: () => <>
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
      label: () => <>Ошибка АЧХ: {m().maxMagErr} дБ</>,
      color: () => m().maxMagErr <= 0.5 ? STATUS_GOOD : m().maxMagErr <= 1.5 ? STATUS_WARN : STATUS_BAD,
      title: "Ошибка АЧХ",
      body: () => <>
        <p>Наибольшее расхождение АЧХ готового FIR с моделью (цель + PEQ) в полосе
          пропускания {band()}.</p>
        <p>Растёт, когда длины фильтра не хватает на крутые склоны и НЧ-коррекцию.
          Ориентиры: до 0.5 дБ — хорошо, до 1.5 дБ — терпимо.</p>
      </>,
    },
    {
      id: "gd",
      // b141.119 (UI audit §4): a span, not an error — no traffic light.
      label: () => <>Размах ГЗ: {m().gdRippleMs} мс</>,
      title: "Размах групповой задержки",
      body: () => <>
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
      label: () => <>УЗ-ФНЧ: {fmtHz(m().ultrasonicLpHz ?? 0)}</>,
      title: "Ультразвуковой ФНЧ",
      body: () => <>
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
      label: () => <>Нормировка: {m().normDb.toFixed(1)} дБ</>,
      title: "Нормировка",
      body: () => <>
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
      <span>{m().taps} {pluralRu(m().taps, "отсчёт", "отсчёта", "отсчётов")}</span>
      <span>{m().sampleRate / 1000} кГц</span>
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
              <div style={{ "font-weight": 600, "margin-bottom": "6px" }}>{it.title}</div>
              <div class="metric-popover-body">{it.body()}</div>
            </div>
          </Show>
        </span>
      </Show>
    );
  }
}

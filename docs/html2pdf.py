#!/usr/bin/env python3
"""Convert PhaseForge HTML manuals to PDF using fpdf2 with Unicode support."""

import sys
import os

# Add user site-packages
user_site = os.path.expanduser("~/Library/Python/3.13/lib/python/site-packages")
if os.path.isdir(user_site):
    sys.path.insert(0, user_site)
# Also try 3.12, 3.11, etc.
for v in ["3.13", "3.12", "3.11"]:
    p = os.path.expanduser(f"~/Library/Python/{v}/lib/python/site-packages")
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from fpdf import FPDF

class ManualPDF(FPDF):
    def __init__(self, lang="en"):
        super().__init__()
        self.lang = lang
        # Use built-in Helvetica (supports basic Latin)
        # For Cyrillic we need a Unicode font
        font_dir = "/System/Library/Fonts"
        # Try to add a system font with Cyrillic support
        for fname in ["Helvetica.ttc", "SFPro.ttf", "Arial Unicode.ttf"]:
            fpath = os.path.join(font_dir, fname)
            if os.path.exists(fpath):
                try:
                    self.add_font("UniFont", "", fpath, uni=True)
                    self.add_font("UniFont", "B", fpath, uni=True)
                    self.has_uni = True
                    return
                except:
                    pass
        # Try supplemental
        font_dir2 = "/System/Library/Fonts/Supplemental"
        for fname in ["Arial Unicode.ttf", "Arial.ttf", "Tahoma.ttf"]:
            fpath = os.path.join(font_dir2, fname)
            if os.path.exists(fpath):
                try:
                    self.add_font("UniFont", "", fpath, uni=True)
                    self.add_font("UniFont", "B", fpath, uni=True)
                    self.has_uni = True
                    return
                except:
                    pass
        # Last resort: try Apple's SF font
        for d in ["/System/Library/Fonts", "/Library/Fonts", os.path.expanduser("~/Library/Fonts")]:
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.endswith(".ttf") and ("arial" in f.lower() or "sf" in f.lower()):
                    fpath = os.path.join(d, f)
                    try:
                        self.add_font("UniFont", "", fpath, uni=True)
                        self.add_font("UniFont", "B", fpath, uni=True)
                        self.has_uni = True
                        return
                    except:
                        pass
        self.has_uni = False

    def uf(self, style=""):
        if self.has_uni:
            return "UniFont", style
        return "Helvetica", style

    def header(self):
        fn, st = self.uf("")
        self.set_font(fn, st, 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 5, "PhaseForge v0.1.0-b89", align="R")
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        fn, st = self.uf("")
        self.set_font(fn, st, 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def chapter_title(self, title):
        fn, st = self.uf("B")
        self.set_font(fn, st, 16)
        self.set_text_color(74, 158, 255)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 200, 200)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def section_title(self, title):
        fn, st = self.uf("B")
        self.set_font(fn, st, 13)
        self.set_text_color(50, 50, 50)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text):
        fn, st = self.uf("")
        self.set_font(fn, st, 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet(self, text):
        fn, st = self.uf("")
        self.set_font(fn, st, 10)
        self.set_text_color(30, 30, 30)
        x = self.l_margin
        self.set_x(x)
        self.cell(8, 5.5, "  " + chr(8226) + " ")
        self.multi_cell(self.w - self.r_margin - self.get_x(), 5.5, text)
        self.set_x(x)

    def table_row(self, cells, header=False):
        fn, st = self.uf("B" if header else "")
        self.set_font(fn, st, 9)
        col_w = (self.w - self.l_margin - self.r_margin) / len(cells)
        if header:
            self.set_fill_color(240, 244, 248)
        self.set_text_color(30, 30, 30)
        for i, c in enumerate(cells):
            w = col_w
            self.cell(w, 6, c, border=1, fill=header)
        self.ln()

    def table_row_wide(self, cells, widths, header=False):
        fn, st = self.uf("B" if header else "")
        self.set_font(fn, st, 9)
        if header:
            self.set_fill_color(240, 244, 248)
        self.set_text_color(30, 30, 30)
        for i, c in enumerate(cells):
            self.cell(widths[i], 6, c, border=1, fill=header)
        self.ln()

    def note_box(self, text):
        self.set_fill_color(255, 248, 225)
        self.set_draw_color(255, 193, 7)
        x = self.get_x()
        y = self.get_y()
        w = self.w - self.l_margin - self.r_margin
        self.rect(x, y, w, 14, "DF")
        self.line(x, y, x, y + 14)
        fn, st = self.uf("")
        self.set_font(fn, st, 9)
        self.set_text_color(80, 60, 0)
        self.set_xy(x + 4, y + 2)
        self.multi_cell(w - 8, 5, text)
        self.ln(4)


def build_en_pdf(output_path):
    pdf = ManualPDF("en")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # Cover page
    pdf.add_page()
    pdf.ln(40)
    fn, st = pdf.uf("B")
    pdf.set_font(fn, st, 32)
    pdf.set_text_color(74, 158, 255)
    pdf.cell(0, 15, "PhaseForge", align="C", new_x="LMARGIN", new_y="NEXT")
    fn, st = pdf.uf("")
    pdf.set_font(fn, st, 14)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "Loudspeaker Correction & FIR Filter Design", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    pdf.set_font(fn, st, 11)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 8, "Version 0.1.0-b89 - User Manual", align="C", new_x="LMARGIN", new_y="NEXT")

    # 1. Overview
    pdf.add_page()
    pdf.chapter_title("1. Overview")
    pdf.body_text("PhaseForge is a desktop application for loudspeaker measurement analysis, target curve design, parametric EQ optimization, and FIR filter generation. It supports multi-band crossover systems with linked HP/LP filters, interactive PEQ editing, and export of correction filters as WAV impulse responses.")
    pdf.section_title("Key Features")
    for f in [
        "Import REW (.txt) and FRD measurements with magnitude and phase",
        "Near-field + far-field measurement merge with baffle step correction",
        "Configurable target curves: HP/LP (Butterworth, Bessel, LR, Gaussian), tilt, shelving",
        "Automatic PEQ optimization (greedy peak-matching + LMA refinement)",
        "Interactive PEQ editing: drag bands, scroll for Q, double-click to create",
        "Multi-band crossover system with linked filters",
        "FIR filter generation with iterative WLS optimization",
        "Configurable narrowband boost limiting and frequency weighting",
        "Export FIR as WAV (mono, float64)",
        "Project save/load with full state preservation",
    ]:
        pdf.bullet(f)

    # 2. Getting Started
    pdf.ln(4)
    pdf.chapter_title("2. Getting Started")
    for i, step in enumerate([
        "Launch PhaseForge. The Welcome dialog offers New Project or Open Project.",
        "Create a new project: enter name, choose band count (1-8).",
        "On the Measurement tab, click Import to load a .txt or .frd file.",
        "Switch to Target tab to configure target response (HP, LP, tilt, shelves).",
        "Click Optimize in the PEQ sidebar to auto-generate PEQ correction.",
        "Switch to Export tab to preview and export the FIR filter.",
    ], 1):
        pdf.body_text(f"{i}. {step}")

    # 3. Interface Layout
    pdf.chapter_title("3. Interface Layout")
    for item in [
        "Top toolbar - file menu, project name, strategy toggle, Settings, Optimize All",
        "Band tabs - SUM + individual band tabs (draggable to reorder)",
        "Plot area - upper: frequency response; lower: impulse / PEQ / export impulse",
        "Control panel (right) - measurement, target, or export controls",
        "PEQ sidebar (right, Target tab only) - PEQ band table and optimizer",
        "Status bar (bottom) - band count, current view",
    ]:
        pdf.bullet(item)

    # 4. Toolbar
    pdf.add_page()
    pdf.chapter_title("4. Toolbar")
    w = [40, pdf.w - pdf.l_margin - pdf.r_margin - 40]
    pdf.table_row_wide(["Element", "Description"], w, header=True)
    for row in [
        ["File menu", "New, Open, Save, Save As, Recent Projects"],
        ["Project name", "Current project. Shows (modified) when unsaved"],
        ["Standard/Hybrid", "Strategy toggle. Standard=PEQ, Hybrid=phase-aware"],
        ["Settings", "Opens Optimization Settings dialog (FIR parameters)"],
        ["Optimize All", "Run PEQ optimizer for all bands simultaneously"],
    ]:
        pdf.table_row_wide(row, w)

    # 5. Band Tabs
    pdf.ln(4)
    pdf.chapter_title("5. Band Tabs & SUM View")
    pdf.section_title("Band Management")
    for item in [
        "SUM tab (always first) - shows combined response of all bands",
        "Band N tabs - individual band workspaces",
        "+ button - add new band",
        "x on tab - delete band (with confirmation)",
        "Drag tabs to reorder bands",
    ]:
        pdf.bullet(item)
    pdf.section_title("SUM View Features")
    for item in [
        "All band target curves and corrected responses overlaid",
        "Crossover markers at band boundaries",
        "Drag crossover markers to adjust frequency",
        "Double-click crossover to open Crossover Dialog",
    ]:
        pdf.bullet(item)

    # 6. Measurement Tab
    pdf.add_page()
    pdf.chapter_title("6. Measurement Tab")
    for item in [
        "Import File - load .txt (REW) or .frd measurement",
        "Merge NF+FF - blend near-field and far-field measurements",
        "Smoothing - Off, 1/3, 1/6, 1/12, 1/24 octave, or Variable",
        "Delay compensation - auto-detects propagation delay; toggle to remove",
        "Polarity invert - NOR/INV toggle (+180 deg phase shift)",
        "Floor bounce - reflection simulation (speaker height, mic height, distance)",
    ]:
        pdf.bullet(item)

    # 7. Target Tab
    pdf.ln(4)
    pdf.chapter_title("7. Target Tab")
    pdf.section_title("General")
    for item in ["ON/OFF toggle for target curve", "Level (dB) - reference level", "Tilt (dB/octave) - frequency slope for house curve"]:
        pdf.bullet(item)
    pdf.section_title("HP / LP Filters")
    w = [40, pdf.w - pdf.l_margin - pdf.r_margin - 40]
    pdf.table_row_wide(["Parameter", "Values"], w, header=True)
    for row in [
        ["Filter type", "Butterworth, Bessel, Linkwitz-Riley, Gaussian"],
        ["Order", "1-8 (LR: 2, 4, 8 only)"],
        ["Frequency", "10-20,000 Hz"],
        ["Linear Phase", "Checkbox (always on for Gaussian)"],
        ["M (Gaussian)", "Shape coefficient, 0.5-10"],
    ]:
        pdf.table_row_wide(row, w)
    pdf.ln(2)
    pdf.body_text("Band Linking: Click the link button on LP to connect with next band's HP. Changes to one automatically mirror the other.")

    # 8. Export Tab
    pdf.add_page()
    pdf.chapter_title("8. Export Tab")
    pdf.section_title("FIR Configuration")
    w = [40, pdf.w - pdf.l_margin - pdf.r_margin - 40]
    pdf.table_row_wide(["Parameter", "Options"], w, header=True)
    for row in [
        ["Sample Rate", "44.1k, 48k, 88.2k, 96k, 176.4k, 192k Hz"],
        ["Taps", "4K, 8K, 16K, 32K, 64K, 128K, 256K"],
        ["Window", "20+ types: Hann, Blackman-Harris, Kaiser, etc."],
    ]:
        pdf.table_row_wide(row, w)
    pdf.section_title("Export Display")
    for item in [
        "Upper plot: Model mag/phase vs FIR realized mag/phase",
        "Lower plot: FIR impulse response (time domain)",
        "Status: taps, sample rate, window, phase mode, normalization, causality %",
        "SNAP/CLR: Snapshot curves for comparison overlay",
        "Export WAV: Save FIR impulse response to file",
    ]:
        pdf.bullet(item)

    # 9. Plots
    pdf.ln(4)
    pdf.chapter_title("9. Plots & Interactions")
    pdf.section_title("Frequency Response Plot")
    w = [50, pdf.w - pdf.l_margin - pdf.r_margin - 50]
    pdf.table_row_wide(["Action", "Result"], w, header=True)
    for row in [
        ["Scroll wheel", "Zoom frequency axis"],
        ["Shift + scroll", "Zoom magnitude axis"],
        ["Click + drag", "Pan"],
        ["Right-click", "Undo last zoom"],
        ["Ctrl + drag", "Zoom box selection"],
        ["Double-click", "Add PEQ band (Target tab)"],
    ]:
        pdf.table_row_wide(row, w)
    pdf.section_title("PEQ Response Plot (Target tab)")
    pdf.table_row_wide(["Action", "Result"], w, header=True)
    for row in [
        ["Drag band marker", "Move frequency and gain"],
        ["Shift+scroll on marker", "Adjust Q (bandwidth)"],
        ["Double-click", "Add new PEQ band"],
    ]:
        pdf.table_row_wide(row, w)

    # 10. PEQ Sidebar
    pdf.add_page()
    pdf.chapter_title("10. PEQ Sidebar")
    pdf.body_text("Visible on the Target tab (non-SUM view).")
    pdf.section_title("Auto-Fit Controls")
    for item in [
        "Tolerance (dB) - error margin for optimization (0.5-3.0)",
        "Max bands - limit PEQ band count (1-60)",
        "Optimize - run auto-optimizer for current band",
        "Clear - remove all PEQ bands",
    ]:
        pdf.bullet(item)
    pdf.section_title("Band Table Columns")
    for item in [
        "Checkbox - enable/disable band",
        "Freq (Hz) - center frequency, 20-20,000. Scroll wheel to adjust",
        "Gain (dB) - +/-18 (Standard) or +/-60 (Hybrid). Color-coded",
        "Q - quality factor, 0.1-20. Scroll wheel to adjust",
        "x - remove band",
    ]:
        pdf.bullet(item)

    # 11. Dialogs
    pdf.ln(4)
    pdf.chapter_title("11. Dialogs")
    pdf.section_title("Optimization Settings")
    pdf.body_text("Opened via the Settings button in the toolbar.")
    w3 = [35, 25, pdf.w - pdf.l_margin - pdf.r_margin - 60]
    pdf.table_row_wide(["Parameter", "Range", "Description"], w3, header=True)
    for row in [
        ["WLS iterations", "0-20", "Error correction passes. 0=off, 3-5 optimal"],
        ["Freq weighting", "on/off", "Priority for crossover/speech bands (200-4k)"],
        ["NB limiting", "on/off", "Clamp sharp peaks above smoothed curve"],
        ["NB smoothing", "0.05-2 oct", "Smoothing window width"],
        ["NB max excess", "1-24 dB", "Max boost above smoothed correction"],
        ["Max boost", "0-60 dB", "Global correction boost limit"],
        ["Noise floor", "-200..-40", "Ignore correction below this level"],
    ]:
        pdf.table_row_wide(row, w3)

    pdf.section_title("Crossover Dialog")
    pdf.body_text("Double-click crossover marker in SUM view. Configure filter type, order, frequency, linear phase for LP/HP pair.")
    pdf.section_title("Merge NF+FF Dialog")
    for item in ["Select NF and FF files", "Splice frequency (50-1000 Hz)", "Blend width (0.5-3.0 octaves)", "Auto level offset or manual", "Optional baffle step correction"]:
        pdf.bullet(item)
    pdf.section_title("Baffle Step Dialog")
    pdf.body_text("Configure baffle dimensions and driver position. Live preview shows correction curve with f3 and edge diffraction points.")

    # 12. Project Management
    pdf.add_page()
    pdf.chapter_title("12. Project Management")
    pdf.section_title("Project Structure")
    for item in [
        "project-name.pfproj - JSON metadata file",
        "inbox/ - imported measurement files",
        "All band configs, PEQ, targets, optimization settings",
    ]:
        pdf.bullet(item)
    pdf.section_title("Workflow")
    w = [50, pdf.w - pdf.l_margin - pdf.r_margin - 50]
    pdf.table_row_wide(["Action", "Shortcut"], w, header=True)
    for row in [["New Project", "Cmd+N"], ["Open Project", "Cmd+O"], ["Save", "Cmd+S"], ["Save As", "Shift+Cmd+S"]]:
        pdf.table_row_wide(row, w)
    pdf.ln(4)
    pdf.note_box("Note: FIR filters are not saved in the project - they are recomputed on demand. Use Export WAV to save.")

    # 13. Supported Formats
    pdf.ln(4)
    pdf.chapter_title("13. Supported Formats")
    pdf.section_title("Import")
    w3 = [30, 25, pdf.w - pdf.l_margin - pdf.r_margin - 55]
    pdf.table_row_wide(["Format", "Extension", "Columns"], w3, header=True)
    pdf.table_row_wide(["REW Text", ".txt", "Freq(Hz), SPL(dB), Phase(deg) - 2 or 3 cols"], w3)
    pdf.table_row_wide(["FRD", ".frd", "Same as REW"], w3)
    pdf.body_text("Lines starting with * or # are comments. Frequency must be monotonically increasing.")
    pdf.section_title("Export")
    w = [40, pdf.w - pdf.l_margin - pdf.r_margin - 40]
    pdf.table_row_wide(["Format", "Description"], w, header=True)
    pdf.table_row_wide(["FIR WAV", "Mono, 64-bit float, selected sample rate"], w)
    pdf.table_row_wide(["Target TXT", "REW-compatible text file"], w)

    # 14. Shortcuts
    pdf.ln(4)
    pdf.chapter_title("14. Keyboard Shortcuts")
    w = [50, pdf.w - pdf.l_margin - pdf.r_margin - 50]
    pdf.table_row_wide(["Shortcut", "Action"], w, header=True)
    for row in [
        ["Cmd+N", "New Project"],
        ["Cmd+O", "Open Project"],
        ["Cmd+S", "Save Project"],
        ["Shift+Cmd+S", "Save As"],
        ["Scroll", "Zoom frequency axis"],
        ["Shift+Scroll", "Zoom magnitude axis"],
        ["Right-click", "Undo last zoom"],
        ["Ctrl+Drag", "Zoom box"],
        ["Double-click", "Add PEQ band (on chart)"],
        ["Escape", "Close dialog"],
        ["Enter", "Confirm dialog"],
    ]:
        pdf.table_row_wide(row, w)

    pdf.output(output_path)
    print(f"EN PDF saved: {output_path}")


def build_ru_pdf(output_path):
    pdf = ManualPDF("ru")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # Cover
    pdf.add_page()
    pdf.ln(40)
    fn, st = pdf.uf("B")
    pdf.set_font(fn, st, 32)
    pdf.set_text_color(74, 158, 255)
    pdf.cell(0, 15, "PhaseForge", align="C", new_x="LMARGIN", new_y="NEXT")
    fn, st = pdf.uf("")
    pdf.set_font(fn, st, 14)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "Koрrekция AС и проектирование FIR-фильтров", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    pdf.set_font(fn, st, 11)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 8, "Версия 0.1.0-b89 - Руководство пользователя", align="C", new_x="LMARGIN", new_y="NEXT")

    # 1
    pdf.add_page()
    pdf.chapter_title("1. Обзор")
    pdf.body_text("PhaseForge - десктопное приложение для анализа измерений акустических систем, проектирования целевых кривых, автоматической оптимизации параметрического эквалайзера и генерации FIR-фильтров коррекции. Поддерживает многополосные кроссоверные системы со связанными фильтрами HP/LP, интерактивное редактирование PEQ и экспорт импульсных характеристик в WAV.")
    pdf.section_title("Ключевые возможности")
    for f in [
        "Импорт измерений REW (.txt) и FRD с амплитудой и фазой",
        "Сшивка ближнего и дальнего поля (NF+FF) с коррекцией баффл-степа",
        "Целевые кривые: HP/LP (Butterworth, Bessel, LR, Gaussian), наклон, полки",
        "Автоматическая оптимизация PEQ (жадный поиск + LMA-уточнение)",
        "Интерактивное PEQ: перетаскивание, прокрутка для Q, двойной клик",
        "Многополосная кроссоверная система со связанными фильтрами",
        "FIR с итеративной WLS-оптимизацией",
        "Настраиваемое ограничение узкополосного буста",
        "Экспорт FIR в WAV (моно, float64)",
        "Сохранение/загрузка проектов",
    ]:
        pdf.bullet(f)

    # 2
    pdf.ln(4)
    pdf.chapter_title("2. Быстрый старт")
    for i, step in enumerate([
        "Запустите PhaseForge. Приветственное окно: New Project или Open Project.",
        "Создайте проект: имя, количество полос (1-8).",
        "Вкладка Measurement: Import для загрузки .txt или .frd.",
        "Вкладка Target: настройка целевой кривой (HP, LP, наклон, полки).",
        "Нажмите Optimize в панели PEQ для автоматической коррекции.",
        "Вкладка Export: просмотр и экспорт FIR-фильтра.",
    ], 1):
        pdf.body_text(f"{i}. {step}")

    # 3
    pdf.chapter_title("3. Компоновка интерфейса")
    for item in [
        "Верхняя панель - меню, имя проекта, стратегия, Settings, Optimize All",
        "Вкладки полос - SUM + отдельные полосы (перетаскиваемые)",
        "Область графиков - верх: АЧХ/ФЧХ; низ: импульс / PEQ / экспорт",
        "Панель управления (справа) - измерения, цель или экспорт",
        "Панель PEQ (справа, Target) - таблица PEQ и оптимизатор",
        "Строка состояния (внизу) - число полос, режим",
    ]:
        pdf.bullet(item)

    # 4
    pdf.add_page()
    pdf.chapter_title("4. Панель инструментов")
    w = [40, pdf.w - pdf.l_margin - pdf.r_margin - 40]
    pdf.table_row_wide(["Элемент", "Описание"], w, header=True)
    for row in [
        ["File menu", "New, Open, Save, Save As, Recent Projects"],
        ["Имя проекта", "Текущий проект. (modified) при несохранённых изм."],
        ["Standard/Hybrid", "Стратегия. Standard=PEQ, Hybrid=фазо-ориентир."],
        ["Settings", "Настройки оптимизации (параметры FIR)"],
        ["Optimize All", "Оптимизация PEQ для всех полос"],
    ]:
        pdf.table_row_wide(row, w)

    # 5
    pdf.ln(4)
    pdf.chapter_title("5. Вкладки полос и SUM")
    pdf.section_title("Управление полосами")
    for item in [
        "SUM (первая вкладка) - суммарная характеристика всех полос",
        "Band N - рабочие области отдельных полос",
        "+ - добавить полосу; x - удалить (с подтверждением)",
        "Перетаскивание вкладок для изменения порядка",
    ]:
        pdf.bullet(item)
    pdf.section_title("Режим SUM")
    for item in [
        "Наложение целевых и скорректированных кривых всех полос",
        "Маркеры кроссоверов на границах",
        "Перетаскивание маркеров для настройки частоты",
        "Двойной клик - диалог кроссовера",
    ]:
        pdf.bullet(item)

    # 6
    pdf.add_page()
    pdf.chapter_title("6. Вкладка Measurement")
    for item in [
        "Import File - загрузка .txt (REW) или .frd",
        "Merge NF+FF - сшивка ближнего и дальнего поля",
        "Smoothing - Off, 1/3, 1/6, 1/12, 1/24 октавы, Variable",
        "Компенсация задержки - автоопределение по фазе",
        "Инверсия полярности - NOR/INV (+180 град)",
        "Floor bounce - моделирование отражений",
    ]:
        pdf.bullet(item)

    # 7
    pdf.ln(4)
    pdf.chapter_title("7. Вкладка Target")
    pdf.section_title("Общие параметры")
    for item in ["ON/OFF целевой кривой", "Level (дБ) - опорный уровень", "Tilt (дБ/окт) - наклон (house curve)"]:
        pdf.bullet(item)
    pdf.section_title("Фильтры HP / LP")
    w = [40, pdf.w - pdf.l_margin - pdf.r_margin - 40]
    pdf.table_row_wide(["Параметр", "Значения"], w, header=True)
    for row in [
        ["Тип", "Butterworth, Bessel, Linkwitz-Riley, Gaussian"],
        ["Порядок", "1-8 (LR: 2, 4, 8)"],
        ["Частота", "10-20 000 Гц"],
        ["Linear Phase", "Чекбокс (всегда для Gaussian)"],
        ["M (Gaussian)", "Коэффициент формы, 0.5-10"],
    ]:
        pdf.table_row_wide(row, w)
    pdf.ln(2)
    pdf.body_text("Связка полос: кнопка связи на LP соединяет с HP следующей полосы. Изменения зеркалятся автоматически.")

    # 8
    pdf.add_page()
    pdf.chapter_title("8. Вкладка Export")
    pdf.section_title("Конфигурация FIR")
    w = [40, pdf.w - pdf.l_margin - pdf.r_margin - 40]
    pdf.table_row_wide(["Параметр", "Варианты"], w, header=True)
    for row in [
        ["Sample Rate", "44.1k, 48k, 88.2k, 96k, 176.4k, 192k Гц"],
        ["Taps", "4K, 8K, 16K, 32K, 64K, 128K, 256K"],
        ["Window", "20+ типов: Hann, BH, Kaiser и др."],
    ]:
        pdf.table_row_wide(row, w)
    pdf.section_title("Отображение")
    for item in [
        "Верх: модельная АЧХ/ФЧХ vs реализованная FIR",
        "Низ: импульсная характеристика FIR",
        "Статус: тапы, SR, окно, фаза, нормализация, каузальность %",
        "SNAP/CLR: снимки для сравнения",
        "Export WAV: сохранение FIR в файл",
    ]:
        pdf.bullet(item)

    # 9
    pdf.ln(4)
    pdf.chapter_title("9. Графики и взаимодействие")
    pdf.section_title("График АЧХ")
    w = [50, pdf.w - pdf.l_margin - pdf.r_margin - 50]
    pdf.table_row_wide(["Действие", "Результат"], w, header=True)
    for row in [
        ["Колесо мыши", "Масштаб по частоте"],
        ["Shift + колесо", "Масштаб по амплитуде"],
        ["Клик + тащить", "Панорамирование"],
        ["Правый клик", "Отмена зума"],
        ["Ctrl + тащить", "Прямоугольный зум"],
        ["Двойной клик", "Добавить PEQ (вкл. Target)"],
    ]:
        pdf.table_row_wide(row, w)
    pdf.section_title("График PEQ (вкладка Target)")
    pdf.table_row_wide(["Действие", "Результат"], w, header=True)
    for row in [
        ["Тащить маркер", "Изменение частоты и усиления"],
        ["Shift+колесо на маркере", "Регулировка Q"],
        ["Двойной клик", "Новая PEQ-полоса"],
    ]:
        pdf.table_row_wide(row, w)

    # 10
    pdf.add_page()
    pdf.chapter_title("10. Панель PEQ")
    pdf.body_text("Видна на вкладке Target (не SUM).")
    pdf.section_title("Авто-подбор")
    for item in [
        "Tolerance (дБ) - допуск ошибки (0.5-3.0)",
        "Max bands - лимит полос (1-60)",
        "Optimize - запуск оптимизатора",
        "Clear - удалить все PEQ",
    ]:
        pdf.bullet(item)
    pdf.section_title("Столбцы таблицы")
    for item in [
        "Чекбокс - вкл/выкл полосы",
        "Freq (Гц) - центральная частота. Колесо для настройки",
        "Gain (дБ) - +/-18 (Standard) или +/-60 (Hybrid)",
        "Q - добротность, 0.1-20. Колесо для настройки",
        "x - удалить полосу",
    ]:
        pdf.bullet(item)

    # 11
    pdf.ln(4)
    pdf.chapter_title("11. Диалоговые окна")
    pdf.section_title("Настройки оптимизации")
    pdf.body_text("Кнопка Settings на панели инструментов.")
    w3 = [35, 25, pdf.w - pdf.l_margin - pdf.r_margin - 60]
    pdf.table_row_wide(["Параметр", "Диапазон", "Описание"], w3, header=True)
    for row in [
        ["WLS iterations", "0-20", "Итерации коррекции. 0=выкл, 3-5 оптим."],
        ["Freq weighting", "вкл/выкл", "Приоритет кроссовера/речи (200-4k)"],
        ["NB limiting", "вкл/выкл", "Ограничение острых пиков"],
        ["NB smoothing", "0.05-2 окт", "Ширина окна сглаживания"],
        ["NB max excess", "1-24 дБ", "Макс. буст над сглаженной"],
        ["Max boost", "0-60 дБ", "Глобальный лимит усиления"],
        ["Noise floor", "-200..-40", "Игнорировать коррекцию ниже"],
    ]:
        pdf.table_row_wide(row, w3)

    pdf.section_title("Диалог кроссовера")
    pdf.body_text("Двойной клик по маркеру в SUM. Тип фильтра, порядок, частота, линейная фаза.")
    pdf.section_title("Сшивка NF+FF")
    for item in ["Выбор файлов NF и FF", "Частота сплайса (50-1000 Гц)", "Ширина бленда (0.5-3.0 окт)", "Авто/ручной уровень", "Опциональный баффл-степ"]:
        pdf.bullet(item)

    # 12
    pdf.add_page()
    pdf.chapter_title("12. Управление проектами")
    pdf.section_title("Структура")
    for item in ["project-name.pfproj - JSON метаданные", "inbox/ - импортированные файлы", "Конфигурации полос, PEQ, цели, настройки оптимизации"]:
        pdf.bullet(item)
    w = [50, pdf.w - pdf.l_margin - pdf.r_margin - 50]
    pdf.table_row_wide(["Действие", "Клавиша"], w, header=True)
    for row in [["Новый проект", "Cmd+N"], ["Открыть", "Cmd+O"], ["Сохранить", "Cmd+S"], ["Сохранить как", "Shift+Cmd+S"]]:
        pdf.table_row_wide(row, w)
    pdf.ln(4)
    pdf.note_box("FIR не сохраняется в проекте - пересчитывается по запросу. Используйте Export WAV.")

    # 13
    pdf.ln(4)
    pdf.chapter_title("13. Поддерживаемые форматы")
    pdf.section_title("Импорт")
    w3 = [30, 25, pdf.w - pdf.l_margin - pdf.r_margin - 55]
    pdf.table_row_wide(["Формат", "Расш.", "Столбцы"], w3, header=True)
    pdf.table_row_wide(["REW Text", ".txt", "Freq(Гц), SPL(дБ), Phase(град) - 2 или 3"], w3)
    pdf.table_row_wide(["FRD", ".frd", "Аналогично REW"], w3)
    pdf.body_text("Строки с * или # - комментарии. Частота монотонно возрастающая.")
    pdf.section_title("Экспорт")
    w = [40, pdf.w - pdf.l_margin - pdf.r_margin - 40]
    pdf.table_row_wide(["Формат", "Описание"], w, header=True)
    pdf.table_row_wide(["FIR WAV", "Моно, 64-бит float"], w)
    pdf.table_row_wide(["Target TXT", "REW-совместимый текст"], w)

    # 14
    pdf.ln(4)
    pdf.chapter_title("14. Горячие клавиши")
    w = [50, pdf.w - pdf.l_margin - pdf.r_margin - 50]
    pdf.table_row_wide(["Клавиша", "Действие"], w, header=True)
    for row in [
        ["Cmd+N", "Новый проект"],
        ["Cmd+O", "Открыть"],
        ["Cmd+S", "Сохранить"],
        ["Shift+Cmd+S", "Сохранить как"],
        ["Колесо", "Масштаб по частоте"],
        ["Shift+Колесо", "Масштаб по амплитуде"],
        ["Правый клик", "Отмена зума"],
        ["Ctrl+Drag", "Прямоугольный зум"],
        ["Двойной клик", "Добавить PEQ"],
        ["Escape", "Закрыть диалог"],
        ["Enter", "Подтвердить диалог"],
    ]:
        pdf.table_row_wide(row, w)

    pdf.output(output_path)
    print(f"RU PDF saved: {output_path}")


if __name__ == "__main__":
    docs_dir = os.path.dirname(os.path.abspath(__file__))
    build_en_pdf(os.path.join(docs_dir, "PhaseForge-Manual-EN.pdf"))
    build_ru_pdf(os.path.join(docs_dir, "PhaseForge-Manual-RU.pdf"))

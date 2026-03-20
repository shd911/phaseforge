#!/usr/bin/env python3
"""Generate PhaseForge Optimization Strategies & Methods PDF (EN + RU)."""

import sys, os
for v in ["3.14", "3.13", "3.12", "3.11"]:
    p = os.path.expanduser(f"~/Library/Python/{v}/lib/python/site-packages")
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from fpdf import FPDF

class DocPDF(FPDF):
    def __init__(self):
        super().__init__()
        font_dir = "/System/Library/Fonts"
        self.has_uni = False
        for fname in ["Helvetica.ttc"]:
            fpath = os.path.join(font_dir, fname)
            if os.path.exists(fpath):
                try:
                    self.add_font("UF", "", fpath, uni=True)
                    self.add_font("UF", "B", fpath, uni=True)
                    self.add_font("UF", "I", fpath, uni=True)
                    self.has_uni = True
                    return
                except: pass
        for d in ["/System/Library/Fonts/Supplemental", "/Library/Fonts"]:
            if not os.path.isdir(d): continue
            for f in os.listdir(d):
                if f.endswith(".ttf") and "arial" in f.lower():
                    fpath = os.path.join(d, f)
                    try:
                        self.add_font("UF", "", fpath, uni=True)
                        self.add_font("UF", "B", fpath, uni=True)
                        self.add_font("UF", "I", fpath, uni=True)
                        self.has_uni = True
                        return
                    except: pass

    def f(self, style=""): return ("UF" if self.has_uni else "Helvetica", style)

    def header(self):
        fn, st = self.f("")
        self.set_font(fn, st, 8)
        self.set_text_color(150,150,150)
        self.cell(0, 5, "PhaseForge v0.1.0-b89 Optimization Reference", align="R")
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        fn, st = self.f("")
        self.set_font(fn, st, 8)
        self.set_text_color(150,150,150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def h1(self, t):
        fn, st = self.f("B")
        self.set_font(fn, st, 18)
        self.set_text_color(74, 158, 255)
        self.cell(0, 12, t, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(74, 158, 255)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(5)

    def h2(self, t):
        fn, st = self.f("B")
        self.set_font(fn, st, 14)
        self.set_text_color(50, 80, 150)
        self.cell(0, 9, t, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200,200,200)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def h3(self, t):
        fn, st = self.f("B")
        self.set_font(fn, st, 11)
        self.set_text_color(60, 60, 60)
        self.cell(0, 7, t, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def p(self, t):
        fn, st = self.f("")
        self.set_font(fn, st, 10)
        self.set_text_color(30,30,30)
        self.multi_cell(0, 5, t)
        self.ln(2)

    def pi(self, t):
        fn, st = self.f("I")
        self.set_font(fn, st, 9)
        self.set_text_color(80,80,80)
        self.multi_cell(0, 4.5, t)
        self.ln(2)

    def b(self, t):
        fn, st = self.f("")
        self.set_font(fn, st, 10)
        self.set_text_color(30,30,30)
        x = self.l_margin
        self.set_x(x)
        self.cell(8, 5, "  " + chr(8226) + " ")
        self.multi_cell(self.w - self.r_margin - self.get_x(), 5, t)
        self.set_x(x)

    def code(self, t):
        fn, st = self.f("")
        self.set_font(fn, st, 8)
        self.set_fill_color(240, 244, 248)
        self.set_text_color(50,50,50)
        x = self.l_margin + 4
        self.set_x(x)
        for line in t.strip().split("\n"):
            self.set_x(x)
            self.cell(self.w - self.r_margin - x - 2, 4.2, line, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def tr(self, cells, widths, header=False):
        fn, st = self.f("B" if header else "")
        self.set_font(fn, st, 9)
        if header: self.set_fill_color(240, 244, 248)
        self.set_text_color(30,30,30)
        for i, c in enumerate(cells):
            self.cell(widths[i], 6, c, border=1, fill=header)
        self.ln()

    def note(self, t):
        self.set_fill_color(255, 248, 225)
        self.set_draw_color(255, 193, 7)
        x, y = self.get_x(), self.get_y()
        w = self.w - self.l_margin - self.r_margin
        self.rect(x, y, w, 12, "DF")
        fn, st = self.f("")
        self.set_font(fn, st, 9)
        self.set_text_color(80,60,0)
        self.set_xy(x+4, y+2)
        self.multi_cell(w-8, 4.5, t)
        self.ln(4)


def build_en(path):
    pdf = DocPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # Cover
    pdf.add_page()
    pdf.ln(35)
    fn, st = pdf.f("B")
    pdf.set_font(fn, st, 30)
    pdf.set_text_color(74, 158, 255)
    pdf.cell(0, 14, "PhaseForge", align="C", new_x="LMARGIN", new_y="NEXT")
    fn, st = pdf.f("")
    pdf.set_font(fn, st, 16)
    pdf.set_text_color(80,80,80)
    pdf.cell(0, 10, "Optimization Strategies & Methods", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    pdf.set_font(fn, st, 11)
    pdf.set_text_color(150,150,150)
    pdf.cell(0, 8, "Technical Reference v0.1.0-b89", align="C", new_x="LMARGIN", new_y="NEXT")

    # TOC
    pdf.add_page()
    pdf.h1("Contents")
    for item in [
        "1. Correction Pipeline Overview",
        "2. PEQ Optimization (Levenberg-Marquardt)",
        "3. FIR Filter Generation",
        "4. Phase Computation (Hilbert Transform)",
        "5. Iterative WLS Refinement",
        "6. Narrowband Boost Limiting",
        "7. Frequency-Dependent Weighting",
        "8. Standard vs Hybrid Strategy",
        "9. Window Functions",
        "10. Causality Metric",
        "11. Target Curve Evaluation",
        "12. Parameter Reference",
    ]:
        pdf.b(item)

    # 1. Pipeline
    pdf.add_page()
    pdf.h1("1. Correction Pipeline Overview")
    pdf.p("PhaseForge implements a multi-stage audio correction pipeline that transforms raw loudspeaker measurements into precision FIR correction filters.")
    pdf.h3("Signal Flow")
    pdf.code("Measurement (freq-domain mag + phase)\n  |\n  v\nPhase Processing: unwrap, delay estimation, delay removal\n  |\n  v\nTarget Curve: HP/LP filters, tilt, shelves\n  |\n  v\nPEQ Optimization: LMA auto-fit (greedy init + simultaneous refinement)\n  |\n  v\nFIR Generation: correction = target - (measurement + PEQ)\n  Hilbert min-phase -> IFFT -> windowing -> WLS refinement\n  |\n  v\nExport: WAV impulse response (mono, float64)")
    pdf.p("Each stage can be configured independently. The PEQ corrects broadband deviations with parametric filters (implementable on DSP hardware), while FIR handles fine correction and crossover phase alignment.")

    # 2. PEQ
    pdf.add_page()
    pdf.h1("2. PEQ Optimization")
    pdf.h2("2.1 Levenberg-Marquardt Algorithm (LMA)")
    pdf.p("All PEQ band parameters [frequency, gain, Q] are optimized simultaneously using the Levenberg-Marquardt method - a damped Newton solver for nonlinear least-squares problems.")
    pdf.h3("Cost Function")
    pdf.code("J = sum_i [ sqrt(W_i) * sqrt(bias_i) * (meas[i] + correction[i] - target[i]) ]^2\n    + Q_penalties\n\nW_i     = frequency weight (ERB-inspired)\nbias_i  = sqrt(peak_bias) if error > 0, else 1.0\nQ_pen   = penalizes high-Q bands above crossover")
    pdf.h3("Damping Strategy")
    pdf.p("Lambda starts at 1.0. On accepted step: lambda /= 2 (more Newton-like). On rejected step: lambda *= 2 (more gradient-like). Stops if lambda > 1e6 (stuck) or parameter change < 1e-4 (converged).")
    pdf.h3("Jacobian")
    pdf.p("Computed via numerical finite differences. Step size: 1e-3 relative for frequency, 0.01 absolute for gain and Q. Normal equations solved via Cholesky decomposition.")

    pdf.h2("2.2 Greedy Initialization")
    pdf.p("Before LMA, a greedy algorithm provides warm start:")
    pdf.b("Find largest error peak in smoothed residual")
    pdf.b("Estimate Q from peak width at -6 dB points")
    pdf.b("Place PEQ band: freq = peak, gain = -error, Q = estimated")
    pdf.b("Apply band and mark exclusion zone (+/-1/3 octave)")
    pdf.b("Repeat until max_bands or error < tolerance")

    pdf.h2("2.3 Band Addition Loop")
    pdf.p("After initial LMA convergence, up to 3 additional rounds check for remaining residuals. If worst weighted error > 1.5x tolerance, a new band is added and LMA re-runs on all bands together.")

    pdf.h2("2.4 Merging & Pruning")
    pdf.p("Bands within 1/3 octave are merged: frequency averaged (weighted by gain magnitude), gains summed, Q set to widest. Bands with |gain| < 0.1 dB are pruned.")

    # 3. FIR
    pdf.add_page()
    pdf.h1("3. FIR Filter Generation")
    pdf.h2("3.1 Correction Computation")
    pdf.code("correction_dB = effective_target - (measurement + PEQ)\n\nWithin HP..LP: effective_target = target curve\nOutside: effective_target = smoothed measurement (1/2 oct)\nBlend zones: +/-0.5 octave sigmoid transitions")
    pdf.p("The correction is clamped to [noise_floor_dB, max_boost_dB] and optionally limited by narrowband protection (see section 6).")

    pdf.h2("3.2 Impulse Synthesis")
    pdf.b("Compute minimum-phase from correction magnitude via Hilbert transform")
    pdf.b("Assemble complex spectrum: S[k] = 10^(mag/20) * exp(j*phase)")
    pdf.b("Mirror for conjugate symmetry (ensures real-valued impulse)")
    pdf.b("IFFT to obtain time-domain impulse")
    pdf.b("Phase-dependent reordering: min-phase = causal; linear-phase = centered")
    pdf.b("Apply window function (half-window for min-phase, full for linear)")
    pdf.b("Iterative WLS refinement (if iterations > 0)")
    pdf.b("Passband normalization to 0 dB peak")

    pdf.h2("3.3 Tap Count Selection")
    pdf.p("Recommended taps = 3 x sample_rate / lowest_freq, rounded to next power of 2. Available: 4K to 256K. Higher tap count improves low-frequency resolution and phase accuracy at the cost of latency.")

    # 4. Phase
    pdf.add_page()
    pdf.h1("4. Phase Computation")
    pdf.h2("4.1 Minimum Phase via Hilbert Transform")
    pdf.p("The Hilbert transform computes the minimum-phase response corresponding to a given magnitude spectrum (Oppenheim & Schafer method).")
    pdf.h3("Algorithm")
    pdf.code("1. ln_mag[k] = correction_dB[k] * ln(10) / 20\n2. Create symmetric signal for FFT\n3. X = FFT(ln_mag)\n4. Apply Hilbert window:\n   DC: x1   Positive: x2   Nyquist: x1   Negative: x0\n5. time_domain = IFFT(X)\n6. phase_rad[k] = -imag(time_domain[k]) / N")
    pdf.pi("Property: minimum-phase has all poles/zeros inside unit circle - causal, stable, minimum group delay for given magnitude response.")

    pdf.h2("4.2 Delay Estimation (IR-based)")
    pdf.p("Propagation delay is estimated by reconstructing the impulse response via IFFT and finding the peak position:")
    pdf.code("1. FFT size = next_power_of_2(8 * N_points)\n2. Interpolate mag + phase to linear freq grid\n3. Build complex spectrum, mirror for symmetry\n4. IFFT -> impulse response\n5. delay = argmax(|IR|) / sample_rate")

    pdf.h2("4.3 Delay Removal")
    pdf.code("phase_new[i] = phase[i] + 360 * freq[i] * delay_seconds")
    pdf.p("Compensates linear phase slope caused by propagation delay. Essential for accurate minimum-phase computation.")

    # 5. WLS
    pdf.add_page()
    pdf.h1("5. Iterative WLS Refinement")
    pdf.p("Windowing distorts the frequency-domain response of the FIR filter. Iterative Weighted Least-Squares correction compensates this error.")
    pdf.h3("Algorithm (per iteration)")
    pdf.code("1. FFT(impulse) -> realized_mag_dB\n2. For each bin k (f > 10 Hz):\n   error[k] = desired_dB[k] - realized_dB[k]\n   weight[k] = frequency_weight(f_k)\n   refined[k] += error[k] * weight[k] * 0.7\n3. If max_error < 0.05 dB: STOP\n4. Rebuild: Hilbert(refined) -> IFFT -> window -> new impulse")
    pdf.p("Damping factor 0.7 prevents oscillation. Typically 3-5 iterations reduce max error from 2-3 dB to < 0.5 dB. Convergence is logged per iteration (max_err, rms_err).")
    pdf.note("Iterations = 0 disables refinement. Useful for quick preview or when measurement quality is low.")

    # 6. Narrowband
    pdf.add_page()
    pdf.h1("6. Narrowband Boost Limiting")
    pdf.p("Prevents the filter from aggressively boosting narrow measurement dips caused by interference, comb filtering, or microphone position artifacts.")
    pdf.h3("Algorithm")
    pdf.code("For each FFT bin k (f > 20 Hz):\n  f_lo = f_center / 2^(smoothing/2)\n  f_hi = f_center * 2^(smoothing/2)\n  smoothed[k] = mean(correction[k_lo..k_hi])\n  limit = smoothed[k] + max_excess_dB\n  if correction[k] > limit:\n    correction[k] = limit")
    pdf.p("Default: 1/3-octave smoothing, 6 dB max excess. For high-quality near-field measurements, reduce smoothing (e.g. 1/6 oct) and increase max excess (e.g. 12 dB) to preserve detail.")
    pdf.note("Smoothing width is critical: 1/3 oct works well for far-field. For near-field bass measurements, use 1/6 or smaller to avoid over-smoothing room modes.")

    # 7. Freq weighting
    pdf.h1("7. Frequency-Dependent Weighting")
    pdf.p("Applied during iterative WLS refinement to prioritize perceptually important frequency regions.")
    pdf.h3("Weight Map")
    w = [45, 20, pdf.w - pdf.l_margin - pdf.r_margin - 65]
    pdf.tr(["Frequency Band", "Weight", "Rationale"], w, header=True)
    for row in [
        ["< 200 Hz", "1.0", "Room modes, limited correction value"],
        ["200 - 4000 Hz", "2.0", "Peak auditory sensitivity, speech band"],
        ["4000 - 8000 Hz", "1.5", "Transition region"],
        ["> 8000 Hz", "0.5", "Masking reduces requirements"],
        ["+/- 0.5 oct of HP/LP", "3.0", "Critical for crossover phase alignment"],
    ]:
        pdf.tr(row, w)
    pdf.ln(2)
    pdf.p("Weights are multiplicative with the damping factor (0.7). Higher weight = more correction effort in that band.")

    # 8. Standard vs Hybrid
    pdf.add_page()
    pdf.h1("8. Standard vs Hybrid Strategy")
    pdf.h2("8.1 Standard Strategy")
    pdf.p("PEQ flattens the measurement. FIR applies the complete target curve (HP, LP, tilt, shelves) as a single correction filter.")
    pdf.code("PEQ target = flat line at passband average\nFIR correction = target - (measurement + PEQ)\nFIR phase = Hilbert(correction)  [minimum-phase]")
    pdf.b("Pros: Simple, predictable, works without phase data")
    pdf.b("Cons: Entire target shaping done in FIR phase domain")

    pdf.h2("8.2 Hybrid Strategy")
    pdf.p("Decomposes the correction into two components with different phase behaviors:")
    pdf.code("Correction (min-phase): flattens measurement to reference level\n  correction_dB = ref_level - measurement\n  phase = Hilbert(correction_dB)\n\nFilter (linear-phase): applies target shaping\n  filter_dB = target - ref_level\n  phase = 0  (zero phase, symmetric impulse)\n\nTotal magnitude: correction + filter = target - measurement\nTotal phase: Hilbert(correction) + 0 = min-phase of correction only")
    pdf.b("Pros: Ideal crossover response, minimal phase deviation at crossover")
    pdf.b("Pros: Driver's natural phase compensated, filter shaping adds no phase artifacts")
    pdf.b("Cons: Requires accurate phase measurement")

    pdf.h2("8.3 When to Use Each")
    w2 = [45, pdf.w - pdf.l_margin - pdf.r_margin - 45]
    pdf.tr(["Scenario", "Recommended Strategy"], w2, header=True)
    pdf.tr(["No phase data", "Standard (only option)"], w2)
    pdf.tr(["Single full-range driver", "Standard (simpler)"], w2)
    pdf.tr(["Multi-way crossover system", "Hybrid (better phase at XO)"], w2)
    pdf.tr(["Subwoofer integration", "Hybrid (phase-accurate blending)"], w2)
    pdf.tr(["Quick preview / testing", "Standard (faster)"], w2)

    # 9. Windows
    pdf.add_page()
    pdf.h1("9. Window Functions")
    pdf.p("The window function shapes the FIR impulse in time domain, controlling the trade-off between main-lobe width (frequency resolution) and side-lobe level (spectral leakage).")
    pdf.h3("Available Windows (20+ types)")
    w2 = [42, pdf.w - pdf.l_margin - pdf.r_margin - 42]
    pdf.tr(["Window", "Characteristics"], w2, header=True)
    for row in [
        ["Hann", "Good general purpose, -31 dB sidelobes"],
        ["Blackman-Harris", "Excellent sidelobe suppression (-92 dB)"],
        ["Kaiser (b=10)", "Parametric, adjustable main-lobe/sidelobe trade-off"],
        ["Dolph-Chebyshev", "Equiripple sidelobes (-100 dB)"],
        ["Nuttall4", "Very low sidelobes (-98 dB), wide main lobe"],
        ["FlatTop", "Best amplitude accuracy, wide main lobe"],
        ["Gaussian (s=2.5)", "Smooth taper, good time-domain behavior"],
        ["Tukey (a=0.5)", "Blend between rectangular and Hann"],
    ]:
        pdf.tr(row, w2)
    pdf.ln(2)
    pdf.h3("Half-Window (Minimum-Phase)")
    pdf.p("For minimum-phase FIR, only the right half of the window is used: starts at 1.0 (peak), tapers to 0. This preserves causality while controlling time-domain ringing.")
    pdf.h3("Full Window (Linear-Phase)")
    pdf.p("For linear-phase FIR, the full symmetric window is applied with the impulse centered at N/2. Pre-ringing equals post-ringing.")

    # 10. Causality
    pdf.h1("10. Causality Metric")
    pdf.p("Quantifies the ratio of post-peak energy to total energy in the impulse response.")
    pdf.code("peak_idx = argmax(|impulse[i]|)\npost_peak_energy = sum(impulse[i]^2) for i >= peak_idx\ntotal_energy = sum(impulse[i]^2) for all i\ncausality = post_peak / total")
    w2 = [30, pdf.w - pdf.l_margin - pdf.r_margin - 30]
    pdf.tr(["Value", "Interpretation"], w2, header=True)
    pdf.tr(["1.0", "Perfectly causal (all energy after peak, no pre-ringing)"], w2)
    pdf.tr(["0.95-0.99", "Minimum-phase FIR (minimal pre-ringing)"], w2)
    pdf.tr(["~0.50", "Linear-phase FIR (symmetric, equal pre/post ringing)"], w2)
    pdf.tr(["< 0.50", "Non-causal (excessive pre-ringing, check config)"], w2)

    # 11. Target
    pdf.add_page()
    pdf.h1("11. Target Curve Evaluation")
    pdf.h2("11.1 Filter Types")
    w3 = [35, 25, pdf.w - pdf.l_margin - pdf.r_margin - 60]
    pdf.tr(["Type", "Orders", "Properties"], w3, header=True)
    pdf.tr(["Butterworth", "1-8", "Maximally flat passband, smooth rolloff"], w3)
    pdf.tr(["Bessel", "1-8", "Best group delay flatness (near-linear phase)"], w3)
    pdf.tr(["Linkwitz-Riley", "2,4,8", "Power-complementary, cascaded Butterworth"], w3)
    pdf.tr(["Gaussian", "M=0.5-10", "Perfect complementarity, true linear phase"], w3)
    pdf.ln(2)
    pdf.h2("11.2 Gaussian Crossover")
    pdf.p("Gaussian filters have perfect LP+HP complementarity (LP + HP = 1.0 at all frequencies) and zero phase deviation. Parameter M controls slope steepness: M=1 is gentle (~6 dB/oct), M=4+ gives near-brick-wall response.")

    pdf.h2("11.3 Evaluation Order")
    pdf.code("1. Start: magnitude = reference_level, phase = 0\n2. Apply tilt: mag += tilt_dB/oct * log2(f/f_ref)\n3. Apply high-pass filter (mag + phase)\n4. Apply low-pass filter (mag + phase)\n5. Apply low shelf (mag + phase)\n6. Apply high shelf (mag + phase)\n7. Wrap phase to [-180, 180]")

    # 12. Params
    pdf.add_page()
    pdf.h1("12. Parameter Reference")
    pdf.h2("PEQ Optimizer")
    w3 = [40, 22, pdf.w - pdf.l_margin - pdf.r_margin - 62]
    pdf.tr(["Parameter", "Default", "Description"], w3, header=True)
    for row in [
        ["Tolerance", "1.0 dB", "Convergence threshold for optimization"],
        ["Max bands", "20", "Maximum number of PEQ bands"],
        ["Peak bias", "1.5", "Weight ratio for peaks vs dips"],
        ["Max boost", "6 dB", "Maximum single-band boost (Standard)"],
        ["Max cut", "18 dB", "Maximum single-band cut"],
        ["Freq range", "HP..LP", "Optimization frequency range"],
    ]:
        pdf.tr(row, w3)

    pdf.h2("FIR Generator (Settings Dialog)")
    pdf.tr(["Parameter", "Default", "Description"], w3, header=True)
    for row in [
        ["WLS iterations", "3", "Error correction passes (0=off, 3-5 optimal)"],
        ["Freq weighting", "ON", "Priority for speech/crossover bands"],
        ["NB limiting", "ON", "Clamp sharp boost peaks"],
        ["NB smoothing", "1/3 oct", "Narrowband detection window width"],
        ["NB max excess", "6 dB", "Max boost above smoothed correction"],
        ["Max boost", "24 dB", "Global correction boost limit"],
        ["Noise floor", "-150 dB", "Ignore corrections below this level"],
    ]:
        pdf.tr(row, w3)

    pdf.h2("FIR Export")
    pdf.tr(["Parameter", "Default", "Description"], w3, header=True)
    for row in [
        ["Sample rate", "48 kHz", "44.1k to 192k Hz"],
        ["Taps", "16384", "4K to 256K (resolution vs latency)"],
        ["Window", "Hann", "20+ types available"],
        ["Phase mode", "Auto", "Min-phase, linear, or hybrid"],
    ]:
        pdf.tr(row, w3)

    pdf.output(path)
    print(f"EN: {path}")


def build_ru(path):
    pdf = DocPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # Cover
    pdf.add_page()
    pdf.ln(35)
    fn, st = pdf.f("B")
    pdf.set_font(fn, st, 30)
    pdf.set_text_color(74, 158, 255)
    pdf.cell(0, 14, "PhaseForge", align="C", new_x="LMARGIN", new_y="NEXT")
    fn, st = pdf.f("")
    pdf.set_font(fn, st, 16)
    pdf.set_text_color(80,80,80)
    pdf.cell(0, 10, "Стратегии и методы оптимизации", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    pdf.set_font(fn, st, 11)
    pdf.set_text_color(150,150,150)
    pdf.cell(0, 8, "Техническое описание v0.1.0-b89", align="C", new_x="LMARGIN", new_y="NEXT")

    # TOC
    pdf.add_page()
    pdf.h1("Содержание")
    for item in [
        "1. Обзор конвейера коррекции",
        "2. Оптимизация PEQ (Левенберг-Марквардт)",
        "3. Генерация FIR-фильтра",
        "4. Вычисление фазы (преобразование Гильберта)",
        "5. Итеративное WLS-уточнение",
        "6. Ограничение узкополосного буста",
        "7. Частотно-зависимое взвешивание",
        "8. Стратегии: Standard vs Hybrid",
        "9. Оконные функции",
        "10. Метрика каузальности",
        "11. Вычисление целевой кривой",
        "12. Справочник параметров",
    ]:
        pdf.b(item)

    # 1
    pdf.add_page()
    pdf.h1("1. Обзор конвейера коррекции")
    pdf.p("PhaseForge реализует многоступенчатый конвейер коррекции акустических систем, преобразующий измерения в прецизионные FIR-фильтры.")
    pdf.h3("Поток сигнала")
    pdf.code("Измерение (АЧХ + ФЧХ в частотной области)\n  |\nОбработка фазы: развёртка, оценка задержки, удаление\n  |\nЦелевая кривая: HP/LP фильтры, наклон, полки\n  |\nОптимизация PEQ: LMA (жадная инициализация + совместное уточнение)\n  |\nГенерация FIR: коррекция = цель - (измерение + PEQ)\n  Гильберт мин-фаза -> ОБПФ -> окно -> WLS-уточнение\n  |\nЭкспорт: WAV импульсная характеристика (моно, float64)")
    pdf.p("PEQ корректирует широкополосные отклонения параметрическими фильтрами (реализуемыми на DSP-оборудовании), FIR обеспечивает тонкую коррекцию и фазовое выравнивание кроссовера.")

    # 2
    pdf.add_page()
    pdf.h1("2. Оптимизация PEQ")
    pdf.h2("2.1 Алгоритм Левенберга-Марквардта (LMA)")
    pdf.p("Все параметры PEQ-полос [частота, усиление, Q] оптимизируются одновременно методом Левенберга-Марквардта - демпфированным решателем нелинейных наименьших квадратов.")
    pdf.h3("Функция стоимости")
    pdf.code("J = sum_i [ sqrt(W_i) * sqrt(bias_i) * (meas[i] + corr[i] - target[i]) ]^2\n    + Q_penalties\n\nW_i     = частотный вес (ERB-подобный)\nbias_i  = sqrt(peak_bias) если ошибка > 0, иначе 1.0\nQ_pen   = штраф высокой добротности выше кроссовера")
    pdf.h3("Стратегия демпфирования")
    pdf.p("Lambda начинается с 1.0. Принятый шаг: lambda /= 2 (ближе к Ньютону). Отклонённый: lambda *= 2 (ближе к градиенту). Остановка при lambda > 1e6 или изменении параметров < 1e-4.")

    pdf.h2("2.2 Жадная инициализация")
    pdf.b("Поиск наибольшего пика ошибки в сглаженном остатке")
    pdf.b("Оценка Q по ширине пика на -6 дБ")
    pdf.b("Установка полосы: freq = пик, gain = -ошибка, Q = оценённый")
    pdf.b("Зона исключения +/-1/3 октавы вокруг каждой полосы")
    pdf.b("Повтор до max_bands или ошибка < tolerance")

    pdf.h2("2.3 Цикл добавления полос")
    pdf.p("После сходимости LMA, до 3 дополнительных раундов проверяют остатки. Если худшая взвешенная ошибка > 1.5x tolerance, добавляется полоса и LMA перезапускается.")

    pdf.h2("2.4 Слияние и очистка")
    pdf.p("Полосы в пределах 1/3 октавы сливаются: частота усредняется (вес по усилению), усиления суммируются, Q берётся наименьший. Полосы с |gain| < 0.1 дБ удаляются.")

    # 3
    pdf.add_page()
    pdf.h1("3. Генерация FIR-фильтра")
    pdf.h2("3.1 Вычисление коррекции")
    pdf.code("correction_dB = effective_target - (measurement + PEQ)\n\nВ полосе HP..LP: effective_target = целевая кривая\nВне полосы: effective_target = сглаженное измерение (1/2 окт)\nПереходные зоны: +/-0.5 октавы, сигмоидный бленд")
    pdf.p("Коррекция ограничивается [noise_floor, max_boost] и опционально узкополосной защитой (раздел 6).")

    pdf.h2("3.2 Синтез импульса")
    pdf.b("Минимальная фаза из АЧХ коррекции через преобразование Гильберта")
    pdf.b("Сборка комплексного спектра: S[k] = 10^(mag/20) * exp(j*phase)")
    pdf.b("Зеркалирование для сопряжённой симметрии (реальный импульс)")
    pdf.b("ОБПФ для получения импульса во временной области")
    pdf.b("Фазо-зависимая перестановка: мин-фаза = каузальный; лин-фаза = центрированный")
    pdf.b("Оконная функция (полу-окно для мин-фазы, полное для линейной)")
    pdf.b("Итеративное WLS-уточнение (если iterations > 0)")
    pdf.b("Нормализация полосы пропускания до 0 дБ пик")

    pdf.h2("3.3 Выбор числа тапов")
    pdf.p("Рекомендация: 3 x sample_rate / lowest_freq, округление до степени 2. Доступно: 4K-256K. Больше тапов = лучшее НЧ-разрешение и точность фазы за счёт задержки.")

    # 4
    pdf.add_page()
    pdf.h1("4. Вычисление фазы")
    pdf.h2("4.1 Минимальная фаза через Гильберта")
    pdf.p("Преобразование Гильберта вычисляет минимально-фазовый отклик из АЧХ (метод Оппенгейма-Шафера).")
    pdf.code("1. ln_mag[k] = correction_dB[k] * ln(10) / 20\n2. Симметричный сигнал для БПФ\n3. X = БПФ(ln_mag)\n4. Гильбертово окно:\n   DC: x1   Положит. частоты: x2   Найквист: x1   Отрицат: x0\n5. time_domain = ОБПФ(X)\n6. phase_rad[k] = -imag(time_domain[k]) / N")
    pdf.pi("Свойство: минимально-фазовая система имеет все полюса/нули внутри единичного круга - каузальная, устойчивая, минимальная групповая задержка.")

    pdf.h2("4.2 Оценка задержки (метод ИХ)")
    pdf.code("1. Размер БПФ = next_pow2(8 * N_точек)\n2. Интерполяция АЧХ+ФЧХ на линейную сетку\n3. Комплексный спектр, зеркалирование\n4. ОБПФ -> импульсная характеристика\n5. delay = argmax(|ИХ|) / sample_rate")

    pdf.h2("4.3 Удаление задержки")
    pdf.code("phase_new[i] = phase[i] + 360 * freq[i] * delay_seconds")
    pdf.p("Компенсирует линейный наклон фазы от задержки распространения.")

    # 5
    pdf.add_page()
    pdf.h1("5. Итеративное WLS-уточнение")
    pdf.p("Оконная функция искажает частотную характеристику FIR. Итеративная взвешенная коррекция наименьших квадратов компенсирует эту ошибку.")
    pdf.code("На каждой итерации:\n1. БПФ(импульс) -> realized_mag_dB\n2. Для каждого бина k (f > 10 Гц):\n   error[k] = desired[k] - realized[k]\n   refined[k] += error[k] * weight[k] * 0.7\n3. Если max_error < 0.05 дБ: СТОП\n4. Пересборка: Гильберт(refined) -> ОБПФ -> окно")
    pdf.p("Коэффициент демпфирования 0.7 предотвращает осцилляции. 3-5 итераций снижают максимальную ошибку с 2-3 дБ до < 0.5 дБ.")
    pdf.note("Iterations = 0 отключает уточнение. Полезно для быстрого превью или при низком качестве измерения.")

    # 6
    pdf.h1("6. Ограничение узкополосного буста")
    pdf.p("Предотвращает агрессивное заполнение узких провалов, вызванных интерференцией, гребенчатой фильтрацией или позицией микрофона.")
    pdf.code("Для каждого бина k (f > 20 Гц):\n  smoothed[k] = среднее correction в окне +/-smoothing/2 окт\n  limit = smoothed[k] + max_excess_dB\n  if correction[k] > limit:\n    correction[k] = limit")
    pdf.p("По умолчанию: 1/3 октавы сглаживание, 6 дБ макс. превышение. Для качественных ближнепольных измерений уменьшите сглаживание (1/6 окт) и увеличьте порог (12 дБ).")
    pdf.note("Ширина сглаживания критична: 1/3 окт хороша для дальнего поля. Для ближнего поля на басах используйте 1/6 или меньше.")

    # 7
    pdf.add_page()
    pdf.h1("7. Частотно-зависимое взвешивание")
    pdf.p("Применяется в WLS-уточнении для приоритизации перцептуально важных частотных областей.")
    w = [45, 20, pdf.w - pdf.l_margin - pdf.r_margin - 65]
    pdf.tr(["Полоса частот", "Вес", "Обоснование"], w, header=True)
    for row in [
        ["< 200 Гц", "1.0", "Комнатные моды, ограниченная ценность коррекции"],
        ["200 - 4000 Гц", "2.0", "Пиковая слуховая чувствительность, речь"],
        ["4000 - 8000 Гц", "1.5", "Переходная область"],
        ["> 8000 Гц", "0.5", "Маскировка снижает требования"],
        ["+/- 0.5 окт HP/LP", "3.0", "Критично для фазы кроссовера"],
    ]:
        pdf.tr(row, w)

    # 8
    pdf.add_page()
    pdf.h1("8. Standard vs Hybrid")
    pdf.h2("8.1 Standard")
    pdf.p("PEQ выравнивает измерение. FIR применяет полную целевую кривую как единый корректирующий фильтр.")
    pdf.code("Цель PEQ = плоская линия на уровне полосы пропускания\nКоррекция FIR = цель - (измерение + PEQ)\nФаза FIR = Гильберт(коррекция)  [минимально-фазовая]")
    pdf.b("Плюсы: Простой, предсказуемый, работает без фазы")
    pdf.b("Минусы: Вся формовка цели в фазовой области FIR")

    pdf.h2("8.2 Hybrid")
    pdf.p("Разделяет коррекцию на два компонента с разным фазовым поведением:")
    pdf.code("Коррекция (мин-фаза): выравнивает до опорного уровня\n  correction_dB = ref_level - measurement\n  phase = Гильберт(correction)\n\nФильтр (лин-фаза): формирует целевую кривую\n  filter_dB = target - ref_level\n  phase = 0  (нулевая фаза)\n\nИтого АЧХ: correction + filter = target - measurement\nИтого ФЧХ: Гильберт(correction) + 0")
    pdf.b("Плюсы: Идеальный кроссовер, минимальное отклонение фазы")
    pdf.b("Плюсы: Компенсация фазы динамика, формовка без фазовых артефактов")
    pdf.b("Минусы: Требуется точное измерение фазы")

    pdf.h2("8.3 Когда использовать")
    w2 = [50, pdf.w - pdf.l_margin - pdf.r_margin - 50]
    pdf.tr(["Сценарий", "Рекомендация"], w2, header=True)
    pdf.tr(["Нет данных фазы", "Standard (единственный вариант)"], w2)
    pdf.tr(["Один широкополосный динамик", "Standard (проще)"], w2)
    pdf.tr(["Многополосная система", "Hybrid (лучшая фаза на XO)"], w2)
    pdf.tr(["Интеграция сабвуфера", "Hybrid (фазо-точное сведение)"], w2)
    pdf.tr(["Быстрый превью", "Standard (быстрее)"], w2)

    # 9
    pdf.add_page()
    pdf.h1("9. Оконные функции")
    pdf.p("Оконная функция формирует импульс FIR во временной области, управляя компромиссом между шириной главного лепестка (частотное разрешение) и уровнем боковых лепестков (утечка спектра).")
    w2 = [42, pdf.w - pdf.l_margin - pdf.r_margin - 42]
    pdf.tr(["Окно", "Характеристики"], w2, header=True)
    for row in [
        ["Hann", "Универсальное, бок. лепестки -31 дБ"],
        ["Blackman-Harris", "Отличное подавление (-92 дБ)"],
        ["Kaiser (b=10)", "Параметрическое, настраиваемый компромисс"],
        ["Dolph-Chebyshev", "Равноволновые бок. лепестки (-100 дБ)"],
        ["Nuttall4", "Очень низкие бок. лепестки (-98 дБ)"],
        ["FlatTop", "Лучшая точность амплитуды"],
        ["Gaussian (s=2.5)", "Плавный спад, хорошее поведение во времени"],
        ["Tukey (a=0.5)", "Смесь прямоугольного и Hann"],
    ]:
        pdf.tr(row, w2)
    pdf.ln(2)
    pdf.p("Мин-фаза: полу-окно (правая половина, спад от 1 к 0). Лин-фаза: полное симметричное окно с импульсом в центре.")

    # 10
    pdf.h1("10. Метрика каузальности")
    pdf.code("peak_idx = argmax(|impulse[i]|)\npost_energy = sum(impulse[i]^2) for i >= peak_idx\ntotal_energy = sum(impulse[i]^2)\ncausality = post_energy / total_energy")
    w2 = [30, pdf.w - pdf.l_margin - pdf.r_margin - 30]
    pdf.tr(["Значение", "Интерпретация"], w2, header=True)
    pdf.tr(["1.0", "Идеально каузальный (нет пре-звона)"], w2)
    pdf.tr(["0.95-0.99", "Мин-фаза FIR (минимальный пре-звон)"], w2)
    pdf.tr(["~0.50", "Лин-фаза FIR (симметричный, равный пре/пост)"], w2)
    pdf.tr(["< 0.50", "Не-каузальный (чрезмерный пре-звон)"], w2)

    # 11
    pdf.add_page()
    pdf.h1("11. Целевая кривая")
    pdf.h2("Типы фильтров")
    w3 = [35, 25, pdf.w - pdf.l_margin - pdf.r_margin - 60]
    pdf.tr(["Тип", "Порядки", "Свойства"], w3, header=True)
    pdf.tr(["Butterworth", "1-8", "Макс. плоская полоса, плавный спад"], w3)
    pdf.tr(["Bessel", "1-8", "Лучшая плоскость групповой задержки"], w3)
    pdf.tr(["Linkwitz-Riley", "2,4,8", "Комплементарные по мощности"], w3)
    pdf.tr(["Gaussian", "M=0.5-10", "Идеальная комплементарность, лин. фаза"], w3)
    pdf.ln(2)
    pdf.p("Gaussian: LP + HP = 1.0 на всех частотах, нулевое отклонение фазы. M управляет крутизной: M=1 мягкий, M=4+ почти идеальный.")

    # 12
    pdf.add_page()
    pdf.h1("12. Справочник параметров")
    pdf.h2("Оптимизатор PEQ")
    w3 = [40, 22, pdf.w - pdf.l_margin - pdf.r_margin - 62]
    pdf.tr(["Параметр", "Дефолт", "Описание"], w3, header=True)
    for row in [
        ["Tolerance", "1.0 дБ", "Порог сходимости оптимизации"],
        ["Max bands", "20", "Максимум PEQ-полос"],
        ["Peak bias", "1.5", "Соотношение весов пиков vs провалов"],
        ["Max boost", "6 дБ", "Макс. буст одной полосы (Standard)"],
        ["Max cut", "18 дБ", "Макс. подавление одной полосы"],
    ]:
        pdf.tr(row, w3)

    pdf.h2("Генератор FIR (диалог Settings)")
    pdf.tr(["Параметр", "Дефолт", "Описание"], w3, header=True)
    for row in [
        ["WLS iterations", "3", "Проходы коррекции (0=выкл, 3-5 оптим.)"],
        ["Freq weighting", "ВКЛ", "Приоритет речевых/кроссоверных полос"],
        ["NB limiting", "ВКЛ", "Ограничение острых пиков буста"],
        ["NB smoothing", "1/3 окт", "Окно обнаружения узких пиков"],
        ["NB max excess", "6 дБ", "Макс. буст над сглаженной коррекцией"],
        ["Max boost", "24 дБ", "Глобальный лимит буста коррекции"],
        ["Noise floor", "-150 дБ", "Игнорировать коррекцию ниже уровня"],
    ]:
        pdf.tr(row, w3)

    pdf.h2("Экспорт FIR")
    pdf.tr(["Параметр", "Дефолт", "Описание"], w3, header=True)
    for row in [
        ["Sample rate", "48 кГц", "44.1k - 192k Гц"],
        ["Taps", "16384", "4K - 256K (разрешение vs задержка)"],
        ["Window", "Hann", "20+ типов доступно"],
        ["Phase mode", "Авто", "Мин-фаза, линейная или гибрид"],
    ]:
        pdf.tr(row, w3)

    pdf.output(path)
    print(f"RU: {path}")


if __name__ == "__main__":
    d = os.path.dirname(os.path.abspath(__file__))
    build_en(os.path.join(d, "PhaseForge-Optimization-EN.pdf"))
    build_ru(os.path.join(d, "PhaseForge-Optimization-RU.pdf"))

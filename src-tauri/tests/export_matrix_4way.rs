// b141.108: exported-WAV matrix on a synthetic 4-way system.
//
// Four bands with a flat magnitude and zero phase measurement (so no PEQ),
// crossovers 100 / 500 / 2000 Hz. Every filter family the UI offers
// (LR / BW / Bessel 1–8, «Произвольный Q», Gaussian M), every phase mode
// (all min, all linear, HP min + LP lin, HP lin + LP min), Gaussian
// subsonic, the export rates/taps and windows. Each band goes through the
// app's own path — `generate_band_fir` → `export_wav_f64` — the WAV is read
// back FROM DISK and checked independently of the pipeline's own analysis:
//
//   • header (IEEE float 64, rate, length) and finite samples;
//   • leading zeros = the reported WAV delay = N/2 (WAV delay convention);
//   • FFT of the file vs the designed response (target::evaluate + the
//     min-phase Hilbert terms the plot uses), magnitude where the target is
//     above −30 dB, phase where it is above −20 dB;
//   • the sum of the four files vs the sum of the four designs (with the
//     polarity that makes the design flattest — what the user sets with INV).
//
// A report is written to target/export_matrix_report.md.
use phaseforge_lib::fir::band::{generate_band_fir, target_hilbert_phase, BandFirSettings};
use phaseforge_lib::fir::{export_wav_f64, WindowType};
use phaseforge_lib::fir::ultrasonic::ultrasonic_lp_gain;
use phaseforge_lib::target::{self, FilterConfig, FilterType, TargetCurve};
use rustfft::{num_complex::Complex64, FftPlanner};
use std::f64::consts::PI;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Mutex;

const XO: [f64; 3] = [100.0, 500.0, 2000.0];

static REPORT: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());
static FILE_SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[derive(Clone, Copy, Debug)]
enum Mode { Min, Lin, HpMinLpLin, HpLinLpMin }

impl Mode {
    fn name(self) -> &'static str {
        match self { Mode::Min => "min", Mode::Lin => "lin", Mode::HpMinLpLin => "ФВЧ min+ФНЧ lin", Mode::HpLinLpMin => "ФВЧ lin+ФНЧ min" }
    }
    fn lin(self, is_hp: bool) -> bool {
        match self { Mode::Min => false, Mode::Lin => true, Mode::HpMinLpLin => !is_hp, Mode::HpLinLpMin => is_hp }
    }
}

#[derive(Clone)]
struct Family { ty: FilterType, order: u8, shape: Option<f64>, q: Option<f64>, subsonic: bool }

impl Family {
    fn label(&self) -> String {
        let base = match self.ty {
            FilterType::LinkwitzRiley => format!("LR{} ({} дБ/окт)", self.order, 12 * self.order as u32),
            FilterType::Butterworth => format!("BW{}", self.order),
            FilterType::Bessel => format!("Bessel{}", self.order),
            FilterType::Custom => format!("Q{:.2}·{}", self.q.unwrap(), self.order),
            FilterType::Gaussian => format!("Gauss M{}", self.shape.unwrap()),
        };
        if self.subsonic { format!("{base} +subsonic") } else { base }
    }
}

struct Run { sr: f64, taps: usize, window: WindowType }

const DEFAULT_RUN: Run = Run { sr: 48_000.0, taps: 65_536, window: WindowType::Blackman };

fn filt(f: &Family, fc: f64, lin: bool, is_hp: bool) -> FilterConfig {
    FilterConfig { filter_type: f.ty.clone(), order: f.order, freq_hz: fc, shape: f.shape, linear_phase: lin, q: f.q,
        subsonic_protect: if f.subsonic && is_hp { Some(true) } else { None } }
}

fn bands(f: &Family, m: Mode) -> Vec<TargetCurve> {
    (0..4).map(|i| {
        let hp = (i > 0).then(|| filt(f, XO[i - 1], m.lin(true), true));
        let lp = (i < 3).then(|| filt(f, XO[i], m.lin(false), false));
        TargetCurve { reference_level_db: 0.0, tilt_db_per_octave: 0.0, tilt_ref_freq: 1000.0,
            high_pass: hp, low_pass: lp, low_shelf: None, high_shelf: None }
    }).collect()
}

fn read_wav_f64(p: &std::path::Path) -> Result<(u32, Vec<f64>), String> {
    let b = std::fs::read(p).map_err(|e| e.to_string())?;
    if &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" { return Err("not RIFF/WAVE".into()); }
    let (mut i, mut sr, mut fmt_ok, mut data) = (12usize, 0u32, false, None);
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let len = u32::from_le_bytes(b[i + 4..i + 8].try_into().unwrap()) as usize;
        let body = &b[i + 8..(i + 8 + len).min(b.len())];
        if id == b"fmt " {
            let tag = u16::from_le_bytes(body[0..2].try_into().unwrap());
            let ch = u16::from_le_bytes(body[2..4].try_into().unwrap());
            sr = u32::from_le_bytes(body[4..8].try_into().unwrap());
            let bits = u16::from_le_bytes(body[14..16].try_into().unwrap());
            fmt_ok = tag == 3 && ch == 1 && bits == 64;
        } else if id == b"data" {
            data = Some(body.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>());
        }
        i += 8 + len + (len & 1);
    }
    if !fmt_ok { return Err("fmt is not mono IEEE float 64".into()); }
    data.map(|d| (sr, d)).ok_or_else(|| "no data chunk".into())
}

fn spectrum(h: &[f64]) -> Vec<Complex64> {
    let mut buf: Vec<Complex64> = h.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    FftPlanner::new().plan_fft_forward(buf.len()).process(&mut buf);
    buf
}

/// H(f) of a spectrum with `delay` samples removed, linearly interpolated
/// between bins (smooth once the bulk delay is gone).
fn at(spec: &[Complex64], f: f64, sr: f64, delay: f64) -> Complex64 {
    let n = spec.len();
    let k = f / sr * n as f64;
    let (k0, t) = (k.floor() as usize, k - k.floor());
    let rot = |kk: usize| spec[kk] * Complex64::from_polar(1.0, 2.0 * PI * kk as f64 * delay / n as f64);
    rot(k0) * (1.0 - t) + rot(k0 + 1) * t
}

fn db(c: Complex64) -> f64 { 20.0 * c.norm().max(1e-30).log10() }
fn wrap(d: f64) -> f64 { (d + 180.0).rem_euclid(360.0) - 180.0 }

struct Outcome { fails: Vec<String>, row: String }

fn run(f: &Family, m: Mode, r: &Run, tag: &str) -> Outcome {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/export_matrix");
    std::fs::create_dir_all(&dir).unwrap();
    let grid: Vec<f64> = (0..300).map(|i| 10.0 * (2000f64.min(r.sr * 0.45 / 10.0)).powf(i as f64 / 299.0)).collect();
    let s = BandFirSettings { taps: r.taps, sample_rate: r.sr, window: r.window.clone(), max_boost_db: 24.0,
        noise_floor_db: -150.0, iterations: 3, freq_weighting: true, narrowband_limit: true,
        nb_smoothing_oct: 0.333, nb_max_excess_db: 6.0 };
    let mut fails = Vec::new();
    let (mut wav_sum, mut exp_sum) = (vec![Complex64::new(0.0, 0.0); grid.len()], Vec::new());
    let (mut worst_mag, mut worst_ph, mut routes) = (0.0f64, 0.0f64, String::new());
    let (mut hf_mag, mut hf_ph) = (0.0f64, 0.0f64);
    let mut shortfall: Vec<String> = Vec::new();
    let mut exps = Vec::new();
    let mut wavs = Vec::new();
    let lab = format!("{} · {} · {}", f.label(), m.name(), tag);
    for (i, tc) in bands(f, m).iter().enumerate() {
        let res = match generate_band_fir(tc, &[], &s, false) {
            Ok(v) => v,
            Err(e) => { fails.push(format!("{lab}: полоса {}: ошибка FIR: {e}", i + 1)); return Outcome { fails, row: format!("| {lab} | ОШИБКА | | | | |") }; }
        };
        routes.push(if format!("{:?}", res.route).starts_with('I') { 'I' } else { 'C' });
        // Unique per write: the family tests run in parallel with the same tags.
        let path = dir.join(format!("{}_{}.wav", std::process::id(), FILE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
        export_wav_f64(&res.fir.impulse, r.sr, &path).unwrap();
        let (sr_read, h) = read_wav_f64(&path).unwrap_or_else(|e| panic!("{lab}: {e}"));
        let _ = std::fs::remove_file(&path);
        if sr_read as f64 != r.sr { fails.push(format!("{lab}: полоса {}: в WAV {sr_read} Гц", i + 1)); }
        if h.len() != r.taps { fails.push(format!("{lab}: полоса {}: длина {} вместо {}", i + 1, h.len(), r.taps)); }
        if !h.iter().all(|v| v.is_finite()) { fails.push(format!("{lab}: полоса {}: NaN/Inf в WAV", i + 1)); continue; }
        let delay = res.fir.wav_delay_samples;
        let sections: Vec<&FilterConfig> = [tc.high_pass.as_ref(), tc.low_pass.as_ref()].into_iter().flatten().collect();
        let all_min = sections.iter().all(|c| !c.linear_phase);
        let spec = spectrum(&h);
        // A min-phase file is causal after its N/2 zeros (a linear-phase one is
        // symmetric around N/2 — its pre-ring is by design). Causality is taken
        // BEFORE the zero-phase ultrasonic LP (b141.40) — its symmetric kernel
        // smears the onset back by design, so it is divided out first.
        let causal: Vec<f64> = match res.fir.ultrasonic_lp_hz {
            Some(c) => {
                let n = spec.len();
                let mut b: Vec<Complex64> = spec.iter().enumerate().map(|(k, v)| {
                    let fk = (k.min(n - k)) as f64 * r.sr / n as f64;
                    v / ultrasonic_lp_gain(fk, c).max(1e-12)
                }).collect();
                FftPlanner::new().plan_fft_inverse(n).process(&mut b);
                b.iter().map(|v| v.re / n as f64).collect()
            }
            None => h.clone(),
        };
        let peak = causal.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let lead = causal.iter().position(|v| v.abs() > 1e-9 * peak).unwrap_or(h.len());
        if all_min && lead + 1 < delay { fails.push(format!("{lab}: полоса {}: мин-фаза, но сигнал с отсчёта {lead} при задержке {delay}", i + 1)); }
        // Delay = N/2, except the honest shortfall of a min-phase tail that
        // does not fit (b141.14/19: shown to the user as «конв. X» and in
        // the export toast) — then the tail runs to the last sample.
        let last = h.iter().rposition(|v| v.abs() > 1e-5 * peak).unwrap_or(0);
        let short = delay < r.taps / 2 && all_min && last + 1 >= r.taps;
        if short { shortfall.push(format!("п{} −{:.0} мс", i + 1, (r.taps / 2 - delay) as f64 / r.sr * 1000.0)); }
        else if delay != r.taps / 2 { fails.push(format!("{lab}: полоса {}: задержка WAV {delay} ≠ N/2 {}", i + 1, r.taps / 2)); }
        // Two references. ANALOG = what the plot draws (target::evaluate +
        // the min-phase Hilbert terms). DIGITAL = what the design realises: in
        // a band whose sections are ALL min-phase, an LR/BW/Custom/Bessel section
        // is a bilinear cascade pre-warped at its corner (IIR route, or the
        // cepstral digital section b141.51), i.e. the analog section at
        // fc·tan(πf/sr)/tan(πfc/sr). A mixed band (one section linear) is
        // realised from the analog model as is; linear sections and
        // Gaussian are magnitude targets (Hilbert for min phase), not warped.
        let hil = target_hilbert_phase(&grid, tc.high_pass.as_ref(), tc.low_pass.as_ref(), None, r.sr).unwrap();
        let e = target::evaluate(tc, &grid);
        let analog: Vec<Complex64> = (0..grid.len()).map(|k| Complex64::from_polar(10f64.powf(e.magnitude[k] / 20.0),
            (e.phase[k] + hil[k]).to_radians())).collect();
        let mut digital = vec![Complex64::from_polar(1.0, 0.0); grid.len()];
        for (k, d) in digital.iter_mut().enumerate() { *d = Complex64::from_polar(1.0, hil[k].to_radians()); }
        for (si, c) in sections.iter().enumerate() {
            let is_hp = si == 0 && tc.high_pass.is_some();
            let one = TargetCurve { reference_level_db: 0.0, tilt_db_per_octave: 0.0, tilt_ref_freq: 1000.0,
                high_pass: is_hp.then(|| (*c).clone()), low_pass: (!is_hp).then(|| (*c).clone()), low_shelf: None, high_shelf: None };
            let warped = all_min && !c.linear_phase && matches!(c.filter_type, FilterType::LinkwitzRiley | FilterType::Butterworth | FilterType::Custom | FilterType::Bessel);
            let fe: Vec<f64> = if warped {
                let tw = (PI * c.freq_hz / r.sr).tan();
                grid.iter().map(|&fq| c.freq_hz * (PI * fq / r.sr).tan() / tw).collect()
            } else { grid.clone() };
            let ev = target::evaluate(&one, &fe);
            for k in 0..grid.len() { digital[k] *= Complex64::from_polar(10f64.powf(ev.magnitude[k] / 20.0), ev.phase[k].to_radians()); }
        }
        // The export normalises each band's passband to 0 dB (`norm_db`; the
        // host sets the band's level) — undo it to compare with the design.
        let gain = 10f64.powf(res.fir.norm_db / 20.0);
        let got: Vec<Complex64> = grid.iter().map(|&fq| at(&spec, fq, r.sr, delay as f64) * gain).collect();
        let (mut bm, mut bp, mut fm, mut fp) = (0.0f64, 0.0f64, 0.0, 0.0);
        for k in 0..grid.len() {
            let (ed, ea) = (db(digital[k]), db(analog[k]));
            let dm = (db(got[k]) - ed).abs();
            let dp = wrap(got[k].arg().to_degrees() - digital[k].arg().to_degrees()).abs();
            if ea > -30.0 { hf_mag = hf_mag.max((db(got[k]) - ea).abs()); }
            if ea > -20.0 { hf_ph = hf_ph.max(wrap(got[k].arg().to_degrees() - analog[k].arg().to_degrees()).abs()); }
            // The BW8 subsonic at fc/8 rings longer than a short file holds
            // (0.68 dB at 64K, 0.03 at 256K): checked from 2·fc/8 below 256K.
            let sub_lf = tc.high_pass.as_ref().is_some_and(|h| h.subsonic_protect == Some(true) && r.taps < 262_144 && grid[k] < h.freq_hz / 4.0);
            // The plot's Hilbert grid starts at 5 Hz — 2–4° at 10–26 Hz deep in
            // a slope (Gaussian M 0.5, whose magnitude has a kink at DC).
            let low_edge = grid[k] < 30.0 && ed < -15.0;
            // An analog phase term (the subsonic BW8 on a linear main) is not
            // zero at Nyquist, the realised discrete one is: 2.4° at 20 kHz.
            let near_nyq = tc.high_pass.as_ref().is_some_and(|h| h.subsonic_protect == Some(true)) && grid[k] > r.sr / 4.0;
            if ed > -30.0 && dm > bm && !sub_lf { bm = dm; fm = grid[k]; }
            if ed > -20.0 && dp > bp && !low_edge && !near_nyq { bp = dp; fp = grid[k]; }
        }
        worst_mag = worst_mag.max(bm); worst_ph = worst_ph.max(bp);
        // A tail that did not fit (the shortfall above) is cut at the end of
        // the file: Gaussian M 0.5 never fits (1/t decay) — 2.2° at 33 Hz.
        let ph_tol = if short { 3.0 } else { 2.0 };
        if bp > ph_tol { fails.push(format!("{lab}: полоса {}: фаза файла отличается от расчётной на {bp:.1}° @ {fp:.0} Гц", i + 1)); }
        if bm > 0.3 { fails.push(format!("{lab}: полоса {}: АЧХ файла отличается от расчётной на {bm:.2} дБ @ {fm:.0} Гц", i + 1)); }
        if std::env::var("PF_DIAG").is_ok() {
            println!("    полоса {}: norm {:+.2} дБ; АЧХ {bm:.3} дБ @ {fm:.0} Гц; фаза {bp:.2}° @ {fp:.0} Гц", i + 1, res.fir.norm_db);
        }
        let exp = digital;
        exps.push(exp); wavs.push(got);
    }
    if exps.len() < 4 { return Outcome { fails, row: format!("| {lab} | — | | | | |") }; }
    // Polarity that makes the design flattest (what INV is for).
    let flat = |pol: [f64; 4]| -> (f64, f64) {
        let v: Vec<f64> = (0..grid.len()).filter(|&k| grid[k] >= 20.0)
            .map(|k| db((0..4).map(|b| exps[b][k] * pol[b]).sum::<Complex64>())).collect();
        (v.iter().cloned().fold(f64::MAX, f64::min), v.iter().cloned().fold(f64::MIN, f64::max))
    };
    let pols = [[1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 1.0, -1.0]];
    let pol = *pols.iter().min_by(|a, b| { let (x, y) = (flat(**a), flat(**b)); (x.1 - x.0).partial_cmp(&(y.1 - y.0)).unwrap() }).unwrap();
    let (emin, emax) = flat(pol);
    let (mut sum_mag, mut sum_ph) = (0.0f64, 0.0f64);
    let (mut wmin, mut wmax) = (f64::MAX, f64::MIN);
    for k in 0..grid.len() {
        let e: Complex64 = (0..4).map(|b| exps[b][k] * pol[b]).sum();
        let w: Complex64 = (0..4).map(|b| wavs[b][k] * pol[b]).sum();
        exp_sum.push(e); wav_sum[k] = w;
        if grid[k] < 20.0 { continue; }
        wmin = wmin.min(db(w)); wmax = wmax.max(db(w));
        if db(e) > -20.0 {
            sum_mag = sum_mag.max((db(w) - db(e)).abs());
            // Same analog-vs-discrete limit of the subsonic phase near Nyquist.
            if !(f.subsonic && grid[k] > r.sr / 4.0) {
                sum_ph = sum_ph.max(wrap(w.arg().to_degrees() - e.arg().to_degrees()).abs());
            }
        }
    }
    let inv = if pol[1] < 0.0 { "2,4 INV" } else { "—" };
    let inv = if shortfall.is_empty() { inv.to_string() } else { format!("{inv}; недобор {}", shortfall.join(", ")) };
    if sum_mag > 0.3 || sum_ph > 2.0 { fails.push(format!("{lab}: сумма файлов отличается от расчётной: {sum_mag:.2} дБ / {sum_ph:.1}°")); }
    let row = format!("| {lab} | {routes} | {worst_mag:.3} / {worst_ph:.2}° | {hf_mag:.2} / {hf_ph:.1}° | {sum_mag:.3} / {sum_ph:.2}° | {:+.2}…{:+.2} | {:+.2}…{:+.2} | {inv} |",
        emin, emax, wmin, wmax);
    Outcome { fails, row }
}

fn families(ty: FilterType) -> Vec<Family> {
    match ty {
        FilterType::Gaussian => [0.5, 1.0, 2.0, 4.0, 10.0].iter()
            .flat_map(|&mm| [false, true].map(|sub| Family { ty: FilterType::Gaussian, order: 1, shape: Some(mm), q: None, subsonic: sub }))
            .collect(),
        FilterType::Custom => [2u8, 4].iter().flat_map(|&o| [0.5, 0.707, 1.0, 2.0].map(|q| Family { ty: FilterType::Custom, order: o, shape: None, q: Some(q), subsonic: false })).collect(),
        t => (1..=8).map(|o| Family { ty: t.clone(), order: o, shape: None, q: None, subsonic: false }).collect(),
    }
}

fn sweep(ty: FilterType, section: &str) {
    let mut fails = Vec::new();
    let mut rows = String::new();
    for f in families(ty) {
        for m in [Mode::Min, Mode::Lin, Mode::HpMinLpLin, Mode::HpLinLpMin] {
            let o = run(&f, m, &DEFAULT_RUN, "48k/64K");
            println!("{}", o.row);
            writeln!(rows, "{}", o.row).unwrap();
            fails.extend(o.fails);
        }
    }
    REPORT.lock().unwrap().push((section.to_string(), rows));
    write_report();
    assert!(fails.is_empty(), "{} нарушений:\n{}", fails.len(), fails.join("\n"));
}

fn write_report() {
    let rep = REPORT.lock().unwrap();
    let mut s = String::from("# Экспорт WAV: 4 полосы, разделы 100/500/2000 Гц\n\n\
        Плоский замер, без PEQ. Файлы прочитаны с диска. Колонки: путь FIR по полосам (I — аналитический, C — кепстральный); \
        худшее отклонение файла от цифрового расчёта (мин-фазные LR/BW/Q — билинейный каскад с предыскажением на частоте среза) по АЧХ (где цель > −30 дБ) / фазе (где > −20 дБ), и отдельно — от аналоговой модели, которую рисует график (разница = билинейное искажение); то же для суммы четырёх файлов; уровень каждого файла приведён обратно на `norm_db` (экспорт нормирует пассбанд к 0 дБ, уровень ставит хост); \
        неравномерность суммы расчётной и из файлов (20 Гц – 20 кГц); полярность для ровной суммы.\n\n");
    for (sec, rows) in rep.iter() {
        writeln!(s, "## {sec}\n\n| Вариант | Путь | Файл vs цифровой расчёт | Файл vs график (аналог) | Сумма vs расчёт | Сумма расчётная, дБ | Сумма файлов, дБ | Полярность; недобор задержки |\n|---|---|---|---|---|---|---|---|\n{rows}").unwrap();
    }
    std::fs::write(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/export_matrix_report.md"), s).unwrap();
}

#[test] fn matrix_linkwitz_riley() { sweep(FilterType::LinkwitzRiley, "Linkwitz-Riley"); }
#[test] fn matrix_butterworth() { sweep(FilterType::Butterworth, "Butterworth"); }
#[test] fn matrix_bessel() { sweep(FilterType::Bessel, "Bessel"); }
#[test] fn matrix_custom_q() { sweep(FilterType::Custom, "Произвольный Q"); }
#[test] fn matrix_gaussian() { sweep(FilterType::Gaussian, "Gaussian"); }

/// Export sets and windows on LR2 (min and linear).
#[test]
fn matrix_rates_taps_windows() {
    let f = Family { ty: FilterType::LinkwitzRiley, order: 2, shape: None, q: None, subsonic: false };
    let mut fails = Vec::new();
    let mut rows = String::new();
    for (sr, taps) in [(44_100.0, 16_384), (48_000.0, 32_768), (96_000.0, 131_072), (192_000.0, 262_144), (384_000.0, 1_048_576)] {
        for m in [Mode::Min, Mode::Lin] {
            let o = run(&f, m, &Run { sr, taps, window: WindowType::Blackman }, &format!("{}k/{}K", sr / 1000.0, taps / 1024));
            println!("{}", o.row); writeln!(rows, "{}", o.row).unwrap(); fails.extend(o.fails);
        }
    }
    for w in [WindowType::Rectangular, WindowType::Bartlett, WindowType::Hann, WindowType::Hamming, WindowType::Blackman,
        WindowType::ExactBlackman, WindowType::BlackmanHarris, WindowType::Nuttall3, WindowType::Nuttall4, WindowType::FlatTop,
        WindowType::Kaiser, WindowType::DolphChebyshev, WindowType::Gaussian, WindowType::Tukey, WindowType::Lanczos,
        WindowType::Poisson, WindowType::HannPoisson, WindowType::Bohman, WindowType::Cauchy, WindowType::Riesz] {
        let o = run(&f, Mode::Lin, &Run { sr: 48_000.0, taps: 65_536, window: w.clone() }, &format!("48k/64K {w:?}"));
        println!("{}", o.row); writeln!(rows, "{}", o.row).unwrap(); fails.extend(o.fails);
    }
    REPORT.lock().unwrap().push(("Частоты, отсчёты, окна (LR2)".into(), rows));
    write_report();
    assert!(fails.is_empty(), "{} нарушений:\n{}", fails.len(), fails.join("\n"));
}

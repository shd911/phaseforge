//! FFT engine abstraction — compile-time dispatch to platform-optimal backend.
//!
//! - macOS: Apple Accelerate vDSP (AMX/NEON acceleration, 3-5x vs rustfft)
//! - Other: rustfft (pure Rust, Cooley-Tukey radix-2/4)

use num_complex::Complex64;

// ───────────────────────────────────────────────────────────────────────────
// Public API
// ───────────────────────────────────────────────────────────────────────────

/// FFT engine with cached plans/setups. Create once, reuse for multiple transforms.
pub struct FftEngine {
    inner: Backend,
}

impl FftEngine {
    pub fn new() -> Self {
        Self { inner: Backend::new() }
    }

    /// In-place forward FFT. `buf.len()` must be power-of-2.
    pub fn fft_forward(&mut self, buf: &mut [Complex64]) {
        self.inner.fft_forward(buf);
    }

    /// In-place inverse FFT. `buf.len()` must be power-of-2.
    /// Does NOT normalize — caller must multiply by 1/n.
    pub fn fft_inverse(&mut self, buf: &mut [Complex64]) {
        self.inner.fft_inverse(buf);
    }
}

// ───────────────────────────────────────────────────────────────────────────
// macOS: Apple Accelerate vDSP backend
// ───────────────────────────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
mod vdsp {
    use num_complex::Complex64;
    use std::collections::HashMap;

    /// Apple vDSP split-complex format for f64.
    #[repr(C)]
    struct DSPDoubleSplitComplex {
        realp: *mut f64,
        imagp: *mut f64,
    }

    const FFT_FORWARD: i32 = 1;   // kFFTDirection_Forward
    const FFT_INVERSE: i32 = -1;  // kFFTDirection_Inverse
    const FFT_RADIX2: i32 = 0;    // kFFTRadix2

    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        fn vDSP_create_fftsetupD(log2n: u64, radix: i32) -> *mut std::ffi::c_void;
        fn vDSP_destroy_fftsetupD(setup: *mut std::ffi::c_void);
        fn vDSP_fft_zipD(
            setup: *mut std::ffi::c_void,
            c: *mut DSPDoubleSplitComplex,
            stride: i64,
            log2n: u64,
            direction: i32,
        );
    }

    pub struct VdspBackend {
        setups: HashMap<u64, *mut std::ffi::c_void>,
        re_buf: Vec<f64>,
        im_buf: Vec<f64>,
    }

    // FFT setup pointers are safe to hold — vDSP setups are thread-local-safe
    unsafe impl Send for VdspBackend {}

    impl VdspBackend {
        pub fn new() -> Self {
            Self {
                setups: HashMap::new(),
                re_buf: Vec::new(),
                im_buf: Vec::new(),
            }
        }

        fn get_setup(&mut self, log2n: u64) -> *mut std::ffi::c_void {
            *self.setups.entry(log2n).or_insert_with(|| {
                unsafe { vDSP_create_fftsetupD(log2n, FFT_RADIX2) }
            })
        }

        fn ensure_buffers(&mut self, n: usize) {
            if self.re_buf.len() < n {
                self.re_buf.resize(n, 0.0);
                self.im_buf.resize(n, 0.0);
            }
        }

        pub fn fft_forward(&mut self, buf: &mut [Complex64]) {
            self.transform(buf, FFT_FORWARD);
        }

        pub fn fft_inverse(&mut self, buf: &mut [Complex64]) {
            self.transform(buf, FFT_INVERSE);
        }

        fn transform(&mut self, buf: &mut [Complex64], direction: i32) {
            let n = buf.len();
            assert!(n.is_power_of_two(), "FFT size must be power of 2, got {n}");
            let log2n = n.trailing_zeros() as u64;

            let setup = self.get_setup(log2n);
            self.ensure_buffers(n);

            // Deinterleave Complex64 → split real/imag
            for i in 0..n {
                self.re_buf[i] = buf[i].re;
                self.im_buf[i] = buf[i].im;
            }

            let mut split = DSPDoubleSplitComplex {
                realp: self.re_buf.as_mut_ptr(),
                imagp: self.im_buf.as_mut_ptr(),
            };

            unsafe {
                vDSP_fft_zipD(setup, &mut split, 1, log2n, direction);
            }

            // Interleave back to Complex64
            for i in 0..n {
                buf[i] = Complex64::new(self.re_buf[i], self.im_buf[i]);
            }
        }
    }

    impl Drop for VdspBackend {
        fn drop(&mut self) {
            for (_, setup) in self.setups.drain() {
                unsafe { vDSP_destroy_fftsetupD(setup); }
            }
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Fallback: rustfft backend
// ───────────────────────────────────────────────────────────────────────────

#[cfg(not(target_os = "macos"))]
mod rustfft_backend {
    use num_complex::Complex64;
    use rustfft::FftPlanner;

    pub struct RustFftBackend {
        planner: FftPlanner<f64>,
    }

    impl RustFftBackend {
        pub fn new() -> Self {
            Self { planner: FftPlanner::new() }
        }

        pub fn fft_forward(&mut self, buf: &mut [Complex64]) {
            let n = buf.len();
            let fft = self.planner.plan_fft_forward(n);
            fft.process(buf);
        }

        pub fn fft_inverse(&mut self, buf: &mut [Complex64]) {
            let n = buf.len();
            let fft = self.planner.plan_fft_inverse(n);
            fft.process(buf);
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Backend selection via cfg
// ───────────────────────────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
type Backend = vdsp::VdspBackend;

#[cfg(not(target_os = "macos"))]
type Backend = rustfft_backend::RustFftBackend;

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_roundtrip_fft_ifft() {
        let n = 1024;
        let mut engine = FftEngine::new();

        // Create a real signal: sum of two cosines
        let mut buf: Vec<Complex64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                Complex64::new(
                    (2.0 * PI * 3.0 * t).cos() + 0.5 * (2.0 * PI * 7.0 * t).cos(),
                    0.0,
                )
            })
            .collect();

        let original: Vec<f64> = buf.iter().map(|c| c.re).collect();

        // Forward then inverse
        engine.fft_forward(&mut buf);
        engine.fft_inverse(&mut buf);

        // Normalize and compare
        let norm = 1.0 / n as f64;
        for i in 0..n {
            let restored = buf[i].re * norm;
            assert!(
                (restored - original[i]).abs() < 1e-10,
                "Mismatch at {i}: expected {}, got {restored}",
                original[i]
            );
        }
    }

    #[test]
    fn test_fft_dirac() {
        let n = 256;
        let mut engine = FftEngine::new();

        // Dirac delta: all energy at DC after FFT should be 1.0+0i for all bins
        let mut buf = vec![Complex64::new(0.0, 0.0); n];
        buf[0] = Complex64::new(1.0, 0.0);

        engine.fft_forward(&mut buf);

        for i in 0..n {
            assert!((buf[i].re - 1.0).abs() < 1e-12, "Real mismatch at bin {i}");
            assert!(buf[i].im.abs() < 1e-12, "Imag nonzero at bin {i}");
        }
    }
}

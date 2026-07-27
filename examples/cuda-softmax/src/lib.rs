// SPDX-License-Identifier: MPL-2.0
//! Host side of the CUDA softmax example.
//!
//! The evolvable function returns a [`KernelPlan`] — CUDA C source plus the
//! geometry to launch it with — and everything that needs `unsafe`, a GPU, or
//! a driver handle lives here, behind [`Gpu`]. That split is what lets agent
//! code stay inside symbiont's policy (no `unsafe`, no statics, no FFI) while
//! still programming the GPU: the dylib emits *text* and typed launch
//! parameters, the host compiles, verifies, and measures them.

#![allow(
    unused_crate_dependencies,
    reason = "symbiont, tokio and tracing are used by this package's binary target."
)]

mod gpu;
mod isolate;

pub use gpu::{
    DeviceInfo,
    Gpu,
    GpuError,
    KernelFailure,
    Measurement,
};
pub use isolate::{
    EVAL_ENV,
    Isolated,
    ROOFLINE_ENV,
    evaluator_main,
};

/// The kernel entry point the host looks up in every compiled module.
pub const KERNEL_NAME: &str = "softmax";

/// The C signature every candidate kernel must have.
pub const KERNEL_SIGNATURE: &str =
    "extern \"C\" __global__ void softmax(const float* input, float* output, int rows, int cols)";

/// A CUDA kernel and the geometry to launch it with.
///
/// The host compiles [`KernelPlan::source`] with NVRTC, looks up
/// [`KERNEL_NAME`], and launches it with the given grid/block/shared-memory
/// configuration. Both halves matter: the same source at the wrong block size
/// can be an order of magnitude slower, so the launch geometry is part of what
/// gets evolved rather than something the host picks.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KernelPlan {
    /// CUDA C source defining [`KERNEL_SIGNATURE`].
    ///
    /// Compiled by NVRTC at evaluation time, so it may use any device-side
    /// CUDA C++ the installed NVRTC understands: shared memory, warp shuffles,
    /// vectorized `float4` loads, fast-math intrinsics such as `__expf`.
    pub source: String,

    /// Grid dimensions, in blocks.
    pub grid: (u32, u32, u32),

    /// Block dimensions, in threads. The product must not exceed the device's
    /// maximum threads per block (1024 on every current architecture).
    pub block: (u32, u32, u32),

    /// Dynamic shared memory per block, in bytes — the size that
    /// `extern __shared__` arrays get. Leave at 0 when the kernel declares its
    /// shared memory statically.
    pub shared_bytes: u32,
}

impl KernelPlan {
    /// Total threads per block.
    #[must_use]
    pub fn threads_per_block(&self) -> u64 {
        u64::from(self.block.0) * u64::from(self.block.1) * u64::from(self.block.2)
    }

    /// Total blocks in the grid.
    #[must_use]
    pub fn blocks(&self) -> u64 {
        u64::from(self.grid.0) * u64::from(self.grid.1) * u64::from(self.grid.2)
    }
}

/// Rows and columns of the benchmark matrix.
///
/// One softmax per row over `cols` contiguous `f32`s. `cols` is a multiple of
/// 4, so `float4` loads are always legal.
pub const ROWS: usize = 4096;
/// See [`ROWS`].
pub const COLS: usize = 1024;

/// The naive kernel the evolvable function starts from: one thread per row.
///
/// Correct, and about as slow as a softmax can reasonably be. Each thread
/// walks an entire row on its own, so the 32 threads of a warp touch addresses
/// `cols * 4` bytes apart and every load pulls in a fresh cache line to use
/// four bytes of it. The three passes then re-read the row from DRAM twice.
pub const NAIVE_KERNEL: &str = r#"
extern "C" __global__ void softmax(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* in = input + (size_t)row * cols;
    float* out = output + (size_t)row * cols;

    float maximum = -INFINITY;
    for (int i = 0; i < cols; ++i) {
        maximum = fmaxf(maximum, in[i]);
    }
    float sum = 0.0f;
    for (int i = 0; i < cols; ++i) {
        sum += expf(in[i] - maximum);
    }
    for (int i = 0; i < cols; ++i) {
        out[i] = expf(in[i] - maximum) / sum;
    }
}
"#;

/// Deterministic benchmark input: `rows * cols` values in `[bias, bias + 8]`
/// with a per-row `bias` of up to ~90.
///
/// The bias is the point. `expf(90.0f)` overflows to `inf` in single
/// precision, so a kernel that skips the max-subtraction — a tempting way to
/// delete a whole pass over the row — produces `inf / inf = NaN` and is caught
/// by the correctness gate instead of winning the benchmark.
#[must_use]
pub fn benchmark_input(rows: usize, cols: usize) -> Vec<f32> {
    let mut data = Vec::with_capacity(rows * cols);
    // SplitMix64, inlined: a fixed stream keeps every run comparable without
    // pulling in an RNG dependency.
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        f64::from((z ^ (z >> 31)) as u32) / f64::from(u32::MAX)
    };
    for _ in 0..rows {
        let bias = next() * 90.0;
        for _ in 0..cols {
            data.push((bias + next() * 8.0) as f32);
        }
    }
    data
}

/// Row-wise softmax on the CPU, used as the correctness oracle.
///
/// Accumulated in `f64` so the tolerance the kernels are held to measures
/// *their* error rather than the reference's.
#[must_use]
pub fn reference_softmax(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * cols];
    for row in 0..rows {
        let src = &input[row * cols..(row + 1) * cols];
        let dst = &mut out[row * cols..(row + 1) * cols];
        let maximum = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f64 = src
            .iter()
            .map(|&v| (f64::from(v) - f64::from(maximum)).exp())
            .sum();
        for (d, &v) in dst.iter_mut().zip(src.iter()) {
            *d = ((f64::from(v) - f64::from(maximum)).exp() / sum) as f32;
        }
    }
    out
}

/// Prelude imported by the generated dylib through [`symbiont::DylibConfig`].
pub mod prelude {
    pub use crate::{
        KERNEL_NAME,
        KERNEL_SIGNATURE,
        KernelPlan,
        NAIVE_KERNEL,
    };
}

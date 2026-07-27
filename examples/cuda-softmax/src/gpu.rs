// SPDX-License-Identifier: MPL-2.0
//! The host-owned GPU façade: NVRTC compilation, correctness gate, timing.
//!
//! Every `unsafe` block in this example lives in this file. Agent code never
//! sees a device pointer or a launch: it emits a [`KernelPlan`] and this side
//! decides whether that plan is even launchable, whether its output matches
//! the CPU oracle, and how fast it is.

use std::{
    sync::Arc,
    time::{
        Duration,
        Instant,
    },
};

use cudarc::{
    driver::{
        CudaContext,
        CudaSlice,
        CudaStream,
        DriverError,
        LaunchConfig,
        PushKernelArg,
        sys::CUdevice_attribute,
    },
    nvrtc::{
        CompileError,
        CompileOptions,
        compile_ptx_with_opts,
    },
};

use crate::{
    KERNEL_NAME,
    KernelPlan,
    benchmark_input,
    reference_softmax,
};

/// How long to keep launching before the timed runs start.
///
/// Long enough to matter: a fresh context runs at idle clocks, and these
/// kernels are tens of microseconds each, so a handful of warmup launches
/// would time the GPU on its way up to its boost clock. Measured ceilings
/// varied by 3x before this was time-based rather than a fixed count.
const WARMUP: Duration = Duration::from_millis(100);
/// Timed launches per candidate. Averaged, not median: the kernels here run in
/// tens of microseconds, where a single sync-per-run would dominate, so the
/// whole batch is timed between two synchronizations.
const TIMED_RUNS: usize = 50;
/// Largest absolute deviation from the CPU reference a candidate may have.
///
/// Outputs are probabilities over 1024 columns, so the values themselves are
/// around 1e-3. This tolerance accepts the fast-math `__expf` intrinsic and
/// any sane reduction order, and rejects a missing max-subtraction outright
/// (that produces `inf`/`NaN`, not a small error).
const TOLERANCE: f32 = 1e-6;

/// Prepended to every candidate before compilation.
///
/// NVRTC compiles without the CUDA toolkit headers, so `<math.h>` staples like
/// `INFINITY` and `FLT_MAX` are simply undefined — the most common way for an
/// otherwise fine kernel to fail to compile, and a distraction the search
/// should not spend rounds on. Every definition is guarded, so a candidate
/// that brings its own wins.
///
/// The trailing `#line 1` resets NVRTC's line counter, so the line numbers in
/// its diagnostics refer to the agent's source rather than to this prelude.
const NVRTC_PRELUDE: &str = "#ifndef INFINITY\n\
     #define INFINITY __int_as_float(0x7f800000)\n\
     #endif\n\
     #ifndef NAN\n\
     #define NAN __int_as_float(0x7fffffff)\n\
     #endif\n\
     #ifndef FLT_MAX\n\
     #define FLT_MAX 3.402823466e+38f\n\
     #endif\n\
     #ifndef FLT_MIN\n\
     #define FLT_MIN 1.175494351e-38f\n\
     #endif\n\
     #line 1\n";

/// Static properties of the device the benchmark runs on.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Marketing name, e.g. `NVIDIA GeForce RTX 4090`.
    pub name: String,
    /// Compute capability as `(major, minor)`.
    pub compute_capability: (i32, i32),
    /// Number of streaming multiprocessors.
    pub multiprocessors: i32,
    /// Maximum threads in a single block.
    pub max_threads_per_block: i32,
    /// Maximum statically declared shared memory per block, in bytes.
    pub max_shared_memory_per_block: i32,
    /// Threads per warp (32 on every current architecture).
    pub warp_size: i32,
}

/// What a candidate kernel achieved.
#[derive(Debug, Clone, Copy)]
pub struct Measurement {
    /// Mean kernel duration in microseconds.
    pub micros: f64,
    /// Effective memory throughput: one read plus one write of the matrix,
    /// divided by the kernel duration.
    pub gb_per_s: f64,
    /// [`Measurement::gb_per_s`] as a percentage of the device-to-device copy
    /// throughput measured on the same buffers — the practical roofline for a
    /// memory-bound kernel.
    pub pct_of_roofline: f64,
    /// Largest absolute deviation from the CPU reference.
    pub max_abs_err: f32,
}

/// Why a candidate kernel did not produce a measurement.
#[derive(Debug, Clone)]
pub enum KernelFailure {
    /// The launch geometry was rejected before reaching the driver.
    Geometry(String),
    /// NVRTC refused the source; carries the compiler log.
    Compile(String),
    /// The module compiled but did not export [`KERNEL_NAME`].
    MissingSymbol(String),
    /// The driver rejected the launch, or the kernel faulted while running.
    Launch(String),
    /// The kernel ran but its output does not match the CPU reference.
    Wrong {
        /// Largest absolute deviation found.
        max_abs_err: f32,
        /// Where the first bad element was and what it looked like.
        detail: String,
    },
}

impl KernelFailure {
    /// Whether this failure leaves the CUDA context unusable.
    ///
    /// An illegal memory access is a *sticky* error: every subsequent call
    /// fails too, and not even a brand new context can be created in this
    /// process afterwards, so the only recovery is to replace the process
    /// ([`crate::Isolated`]). Compile errors and wrong answers, by contrast,
    /// leave the device perfectly healthy — which is why they are separate
    /// variants rather than one `Error(String)`.
    #[must_use]
    pub fn poisons_context(&self) -> bool {
        matches!(self, Self::Launch(_))
    }

    /// One-line label for report tables.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Geometry(_) => "bad geometry",
            Self::Compile(_) => "nvrtc error",
            Self::MissingSymbol(_) => "missing symbol",
            Self::Launch(_) => "launch fault",
            Self::Wrong { .. } => "wrong output",
        }
    }
}

impl std::fmt::Display for KernelFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Geometry(msg) | Self::MissingSymbol(msg) | Self::Launch(msg) => {
                write!(f, "{}: {msg}", self.kind())
            }
            Self::Compile(log) => write!(f, "nvrtc error:\n{log}"),
            Self::Wrong {
                max_abs_err,
                detail,
            } if max_abs_err.is_finite() => {
                write!(
                    f,
                    "wrong output (max abs error {max_abs_err:.3e}): {detail}"
                )
            }
            Self::Wrong { detail, .. } => write!(f, "wrong output: {detail}"),
        }
    }
}

impl KernelFailure {
    /// The message without the [`std::fmt::Display`] prefix, for a wire format
    /// that carries the kind separately — otherwise a failure round-tripped
    /// through a child process ends up labelled twice.
    #[must_use]
    pub fn detail(&self) -> &str {
        match self {
            Self::Geometry(msg)
            | Self::Compile(msg)
            | Self::MissingSymbol(msg)
            | Self::Launch(msg)
            | Self::Wrong { detail: msg, .. } => msg,
        }
    }
}

/// Failure to set up the device itself, as opposed to a bad
/// candidate kernel.
#[derive(Debug)]
pub enum GpuError {
    /// `libcuda` could not be loaded: no driver installed.
    DriverUnavailable,
    /// A driver call failed while setting up the benchmark.
    Driver(DriverError),
    /// The host's own reference kernel misbehaved — a bug in this example,
    /// not in anything the agent produced.
    Internal(String),
}

impl From<DriverError> for GpuError {
    fn from(err: DriverError) -> Self {
        Self::Driver(err)
    }
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DriverUnavailable => write!(
                f,
                "the CUDA driver library could not be loaded (no NVIDIA driver installed?)"
            ),
            Self::Driver(err) => write!(f, "CUDA driver error: {err}"),
            Self::Internal(msg) => write!(f, "internal error: {msg}"),
        }
    }
}

impl std::error::Error for GpuError {}

/// Owns the CUDA context, the benchmark buffers and the correctness oracle.
///
/// Lives in the *child* process for the duration of one candidate (see
/// [`crate::Isolated`]), and in the parent only to report the device and the
/// copy ceiling. The evolvable function never touches it.
#[derive(Debug)]
pub struct Gpu {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    input: CudaSlice<f32>,
    output: CudaSlice<f32>,
    rows: usize,
    cols: usize,
    reference: Vec<f32>,
    /// Written into the output buffer before the correctness run, so a kernel
    /// that leaves elements untouched fails the comparison instead of reading
    /// as "the previous candidate's answer" or "zero".
    poison: Vec<f32>,
    info: DeviceInfo,
    /// `compute_XX`, leaked once per context because
    /// [`CompileOptions::arch`] wants a `&'static str`.
    arch: Option<&'static str>,
    roofline_gb_per_s: f64,
}

impl Gpu {
    /// Bring up device 0 and stage a `rows` x `cols` benchmark on it.
    ///
    /// # Errors
    ///
    /// [`GpuError::DriverUnavailable`] when there is no CUDA driver to load,
    /// so a caller can skip the example instead of dying, and
    /// [`GpuError::Driver`] for a device that is present but unusable.
    pub fn new(rows: usize, cols: usize) -> Result<Self, GpuError> {
        if !driver_present() {
            return Err(GpuError::DriverUnavailable);
        }
        let ctx = CudaContext::new(0)?;
        let info = device_info(&ctx)?;
        let arch = arch_flag(info.compute_capability);
        let host_input = benchmark_input(rows, cols);
        let reference = reference_softmax(&host_input, rows, cols);
        let poison = vec![f32::NAN; rows * cols];

        let stream = ctx.default_stream();
        let input = stream.clone_htod(&host_input)?;
        let output = stream.alloc_zeros::<f32>(rows * cols)?;

        let mut gpu = Self {
            ctx,
            stream,
            input,
            output,
            rows,
            cols,
            reference,
            poison,
            info,
            arch,
            roofline_gb_per_s: 0.0,
        };
        gpu.roofline_gb_per_s = match inherited_roofline() {
            Some(gb_per_s) => gb_per_s,
            None => gpu.measure_copy_roofline()?,
        };
        Ok(gpu)
    }

    /// Static properties of the device.
    #[must_use]
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// Device-to-device copy throughput over the benchmark buffers: the
    /// practical bandwidth ceiling a memory-bound kernel is measured against.
    #[must_use]
    pub fn roofline_gb_per_s(&self) -> f64 {
        self.roofline_gb_per_s
    }

    /// Bytes a correct kernel must move at minimum: read the matrix once,
    /// write it once.
    #[must_use]
    pub fn traffic_bytes(&self) -> u64 {
        2 * (self.rows * self.cols * size_of::<f32>()) as u64
    }

    /// Compile, verify and time one candidate.
    ///
    /// # Errors
    ///
    /// A [`KernelFailure`] describing exactly which stage rejected the
    /// candidate, phrased so it can be fed straight back to the agent.
    pub fn evaluate(&mut self, plan: &KernelPlan) -> Result<Measurement, KernelFailure> {
        self.check_geometry(plan)?;
        let ptx = compile_ptx_with_opts(
            format!("{NVRTC_PRELUDE}{}", plan.source),
            CompileOptions {
                arch: self.arch,
                ..Default::default()
            },
        )
        .map_err(|err| KernelFailure::Compile(compile_log(&err)))?;

        let module = self
            .ctx
            .load_module(ptx)
            .map_err(|err| KernelFailure::Compile(format!("PTX failed to load: {err}")))?;
        let func = module.load_function(KERNEL_NAME).map_err(|err| {
            KernelFailure::MissingSymbol(format!(
                "the module does not export `{KERNEL_NAME}` ({err}); \
                 the kernel must be declared `extern \"C\"`"
            ))
        })?;

        let cfg = LaunchConfig {
            grid_dim: plan.grid,
            block_dim: plan.block,
            shared_mem_bytes: plan.shared_bytes,
        };
        let rows = i32::try_from(self.rows).expect("row count fits in i32");
        let cols = i32::try_from(self.cols).expect("column count fits in i32");

        // Poison the output so untouched elements read as NaN.
        self.stream
            .memcpy_htod(&self.poison, &mut self.output)
            .map_err(|err| KernelFailure::Launch(format!("could not reset the output: {err}")))?;

        // Split the borrow: the builder needs `&input` and `&mut output` at
        // the same time, which is only expressible per field.
        let Self {
            stream,
            input,
            output,
            ..
        } = self;
        let mut launch = stream.launch_builder(&func);
        launch.arg(&*input).arg(&mut *output).arg(&rows).arg(&cols);

        // SAFETY: nothing about a machine-written kernel is safe, which is the
        // whole point of running it behind a correctness gate and a context we
        // are prepared to throw away. The arguments do match the signature the
        // agent is required to implement, and `check_geometry` has already
        // ruled out the launch configurations the driver would reject.
        unsafe { launch.launch(cfg) }
            .map_err(|err| KernelFailure::Launch(format!("launch rejected: {err}")))?;
        stream
            .synchronize()
            .map_err(|err| KernelFailure::Launch(format!("kernel faulted: {err}")))?;

        let warmup_deadline = Instant::now() + WARMUP;
        while Instant::now() < warmup_deadline {
            // SAFETY: as above; the first launch already completed cleanly.
            unsafe { launch.launch(cfg) }
                .map_err(|err| KernelFailure::Launch(format!("launch rejected: {err}")))?;
            stream
                .synchronize()
                .map_err(|err| KernelFailure::Launch(format!("kernel faulted: {err}")))?;
        }

        let started = Instant::now();
        for _ in 0..TIMED_RUNS {
            // SAFETY: as above.
            unsafe { launch.launch(cfg) }
                .map_err(|err| KernelFailure::Launch(format!("launch rejected: {err}")))?;
        }
        stream
            .synchronize()
            .map_err(|err| KernelFailure::Launch(format!("kernel faulted: {err}")))?;
        let elapsed = started.elapsed();

        // Verification reads back the *warm* result, so a kernel cannot pass
        // by being correct once and racy afterwards.
        let max_abs_err = self.verify()?;

        let seconds = elapsed.as_secs_f64() / TIMED_RUNS as f64;
        let gb_per_s = self.traffic_bytes() as f64 / seconds / 1e9;
        Ok(Measurement {
            micros: seconds * 1e6,
            gb_per_s,
            pct_of_roofline: 100.0 * gb_per_s / self.roofline_gb_per_s,
            max_abs_err,
        })
    }

    /// Reject launch geometry the driver would refuse, with a message the
    /// agent can act on. Cheaper than a round trip to the device, and it keeps
    /// trivially malformed plans out of the fault-recovery path.
    fn check_geometry(&self, plan: &KernelPlan) -> Result<(), KernelFailure> {
        let threads = plan.threads_per_block();
        let max_threads = u64::try_from(self.info.max_threads_per_block).unwrap_or(1024);
        if threads == 0 || plan.blocks() == 0 {
            return Err(KernelFailure::Geometry(format!(
                "grid {:?} x block {:?} launches no threads",
                plan.grid, plan.block
            )));
        }
        if threads > max_threads {
            return Err(KernelFailure::Geometry(format!(
                "block {:?} is {threads} threads, the device allows at most {max_threads}",
                plan.block
            )));
        }
        let rows = self.rows as u64;
        if plan.blocks() * threads < rows {
            return Err(KernelFailure::Geometry(format!(
                "grid {:?} x block {:?} is {} threads for {rows} rows: some rows would go \
                 unprocessed",
                plan.grid,
                plan.block,
                plan.blocks() * threads
            )));
        }
        Ok(())
    }

    /// Compare the device output against the CPU oracle.
    fn verify(&self) -> Result<f32, KernelFailure> {
        let got = self.stream.clone_dtoh(&self.output).map_err(|err| {
            KernelFailure::Launch(format!("could not read the output back: {err}"))
        })?;

        let mut max_abs_err = 0.0_f32;
        let mut first_bad: Option<(usize, f32, f32)> = None;
        for (idx, (&got, &want)) in got.iter().zip(self.reference.iter()).enumerate() {
            let err = (got - want).abs();
            if got.is_finite() && err <= max_abs_err {
                continue;
            }
            if got.is_finite() {
                max_abs_err = err;
            }
            if (!got.is_finite() || err > TOLERANCE) && first_bad.is_none() {
                first_bad = Some((idx, got, want));
            }
        }

        if let Some((idx, got, want)) = first_bad {
            let (row, col) = (idx / self.cols, idx % self.cols);
            return Err(KernelFailure::Wrong {
                max_abs_err,
                detail: format!(
                    "output[{row}][{col}] = {got:e}, expected {want:e} (tolerance {TOLERANCE:e})"
                ),
            });
        }
        Ok(max_abs_err)
    }

    /// Time a kernel that only copies the benchmark buffers.
    ///
    /// A copy moves exactly the traffic a perfect softmax must move — one read
    /// and one write — so its throughput is the honest ceiling to report
    /// candidates against, and it self-calibrates on whatever card is present
    /// instead of trusting a spec sheet.
    ///
    /// It is deliberately a *kernel* and not `cuMemcpyDtoD`: the driver copy is
    /// blocking, so at these sizes its per-call overhead would dominate and
    /// produce a ceiling that candidates comfortably exceed. Measuring the
    /// ceiling through the same launch path as the candidates keeps the
    /// comparison apples to apples.
    fn measure_copy_roofline(&mut self) -> Result<f64, GpuError> {
        let internal = |what: &str, err: &dyn std::fmt::Display| {
            GpuError::Internal(format!("the built-in copy kernel {what}: {err}"))
        };
        let ptx = compile_ptx_with_opts(
            format!("{NVRTC_PRELUDE}{COPY_KERNEL}"),
            CompileOptions {
                arch: self.arch,
                ..Default::default()
            },
        )
        .map_err(|err| internal("did not compile", &compile_log(&err)))?;
        let module = self
            .ctx
            .load_module(ptx)
            .map_err(|err| internal("did not load", &err))?;
        let func = module
            .load_function("copy_kernel")
            .map_err(|err| internal("has no entry point", &err))?;

        let elements = i32::try_from(self.rows * self.cols / 4).expect("float4 count fits in i32");
        let cfg = LaunchConfig::for_num_elems(u32::try_from(elements).expect("positive"));
        let Self {
            stream,
            input,
            output,
            ..
        } = self;
        let mut launch = stream.launch_builder(&func);
        launch.arg(&*input).arg(&mut *output).arg(&elements);

        let warmup_deadline = Instant::now() + WARMUP;
        while Instant::now() < warmup_deadline {
            // SAFETY: the copy kernel is host-written, its arguments match the
            // signature above, and `for_num_elems` covers every element once.
            unsafe { launch.launch(cfg) }.map_err(|err| internal("failed to launch", &err))?;
            stream
                .synchronize()
                .map_err(|err| internal("faulted", &err))?;
        }
        let started = Instant::now();
        for _ in 0..TIMED_RUNS {
            // SAFETY: as above.
            unsafe { launch.launch(cfg) }.map_err(|err| internal("failed to launch", &err))?;
        }
        stream
            .synchronize()
            .map_err(|err| internal("faulted", &err))?;
        let seconds = started.elapsed().as_secs_f64() / TIMED_RUNS as f64;
        Ok(self.traffic_bytes() as f64 / seconds / 1e9)
    }
}

/// The bandwidth ceiling, expressed as a kernel: read the matrix once as
/// `float4`, write it once, do nothing else.
const COPY_KERNEL: &str = r#"
extern "C" __global__ void copy_kernel(const float4* __restrict__ input,
                                       float4* __restrict__ output,
                                       int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) output[i] = input[i];
}
"#;

/// The copy ceiling handed down by a parent process, if any.
fn inherited_roofline() -> Option<f64> {
    std::env::var(crate::ROOFLINE_ENV)
        .ok()?
        .parse::<f64>()
        .ok()
        .filter(|gb_per_s| *gb_per_s > 0.0)
}

/// Whether `libcuda` can be loaded at all.
fn driver_present() -> bool {
    // SAFETY: this only attempts a `dlopen` of the driver library and reports
    // whether it succeeded. It initializes no CUDA state.
    unsafe { cudarc::driver::sys::is_culib_present() }
}

/// NVRTC's `--gpu-architecture` flag for a compute capability.
///
/// `CompileOptions::arch` is a `&'static str`, so the string is leaked — once
/// per process in practice, since the device does not change under us. Falling
/// back to `None` for an unknown capability is safe: NVRTC then emits PTX for
/// its default architecture and the driver JITs it for the real device.
fn arch_flag(compute_capability: (i32, i32)) -> Option<&'static str> {
    let (major, minor) = compute_capability;
    if major < 5 {
        return None;
    }
    Some(Box::leak(
        format!("compute_{major}{minor}").into_boxed_str(),
    ))
}

/// Query the device properties worth putting in a prompt.
fn device_info(ctx: &Arc<CudaContext>) -> Result<DeviceInfo, DriverError> {
    Ok(DeviceInfo {
        name: ctx.name()?,
        compute_capability: ctx.compute_capability()?,
        multiprocessors: ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
        max_threads_per_block: ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?,
        max_shared_memory_per_block: ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?,
        warp_size: ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE)?,
    })
}

/// Pull the compiler log out of an NVRTC error.
///
/// [`CompileError`]'s own `Display` is its `Debug`, which buries the log in
/// escaped `CString` output next to the full option list. The agent needs the
/// diagnostics and nothing else.
fn compile_log(err: &CompileError) -> String {
    match err {
        CompileError::CompileError { log, .. } => {
            let log = log.to_string_lossy().trim().to_string();
            if log.is_empty() {
                "nvrtc rejected the source without a diagnostic".to_string()
            } else {
                log
            }
        }
        other => format!("{other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        COLS,
        NAIVE_KERNEL,
    };

    /// Rows used by the tests. Small enough that the `f64` CPU oracle is cheap
    /// in a debug build, large enough to fill a modern GPU.
    const TEST_ROWS: usize = 512;

    /// A kernel a competent optimizer would write: one block per row, the row
    /// held in registers as one `float4` per thread, and two block-wide
    /// reductions over warp shuffles. Test-only on purpose — it is the answer,
    /// and the host crate's rustdoc is fed to the agent.
    const OPTIMIZED_KERNEL: &str = r#"
extern "C" __global__ void softmax(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int n4 = cols >> 2;

    const float4* in4 = (const float4*)(input + (size_t)row * cols);
    float4* out4 = (float4*)(output + (size_t)row * cols);

    float4 v = make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
    if (threadIdx.x < n4) v = in4[threadIdx.x];

    float m = fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
    for (int off = 16; off > 0; off >>= 1) {
        m = fmaxf(m, __shfl_down_sync(0xffffffff, m, off));
    }
    __shared__ float warp_max[32];
    __shared__ float warp_sum[32];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int warps = (blockDim.x + 31) >> 5;
    if (lane == 0) warp_max[warp] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
        float acc = warp_max[0];
        for (int i = 1; i < warps; ++i) acc = fmaxf(acc, warp_max[i]);
        warp_max[0] = acc;
    }
    __syncthreads();
    float row_max = warp_max[0];

    float ex = __expf(v.x - row_max), ey = __expf(v.y - row_max);
    float ez = __expf(v.z - row_max), ew = __expf(v.w - row_max);
    float s = ex + ey + ez + ew;
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_down_sync(0xffffffff, s, off);
    }
    if (lane == 0) warp_sum[warp] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float acc = 0.0f;
        for (int i = 0; i < warps; ++i) acc += warp_sum[i];
        warp_sum[0] = acc;
    }
    __syncthreads();
    float inv = 1.0f / warp_sum[0];

    if (threadIdx.x < n4) {
        out4[threadIdx.x] = make_float4(ex * inv, ey * inv, ez * inv, ew * inv);
    }
}
"#;

    fn naive_plan(rows: usize) -> KernelPlan {
        KernelPlan {
            source: NAIVE_KERNEL.to_string(),
            grid: (u32::try_from(rows.div_ceil(256)).expect("fits"), 1, 1),
            block: (256, 1, 1),
            shared_bytes: 0,
        }
    }

    fn optimized_plan(rows: usize) -> KernelPlan {
        KernelPlan {
            source: OPTIMIZED_KERNEL.to_string(),
            grid: (u32::try_from(rows).expect("fits"), 1, 1),
            block: (256, 1, 1),
            shared_bytes: 0,
        }
    }

    /// Every test needs a device; without one they report and pass, so the
    /// suite stays green on machines that cannot run it.
    fn gpu() -> Option<Gpu> {
        match Gpu::new(TEST_ROWS, COLS) {
            Ok(gpu) => Some(gpu),
            Err(err) => {
                eprintln!("skipping: {err}");
                None
            }
        }
    }

    #[test]
    fn the_naive_kernel_is_correct() {
        let Some(mut gpu) = gpu() else { return };
        let measurement = gpu
            .evaluate(&naive_plan(TEST_ROWS))
            .expect("the naive kernel is correct");
        assert!(
            measurement.max_abs_err <= TOLERANCE,
            "max abs err {}",
            measurement.max_abs_err
        );
        assert!(measurement.gb_per_s > 0.0);
        eprintln!(
            "naive: {:.1} us, {:.0} GB/s ({:.1}% of ceiling)",
            measurement.micros, measurement.gb_per_s, measurement.pct_of_roofline
        );
    }

    #[test]
    fn an_optimized_kernel_measures_faster_than_the_naive_one() {
        let Some(mut gpu) = gpu() else { return };
        let naive = gpu
            .evaluate(&naive_plan(TEST_ROWS))
            .expect("the naive kernel is correct");
        let optimized = gpu
            .evaluate(&optimized_plan(TEST_ROWS))
            .expect("the optimized kernel is correct");
        eprintln!(
            "naive {:.1} us -> optimized {:.1} us ({:.0} GB/s, {:.1}% of ceiling)",
            naive.micros, optimized.micros, optimized.gb_per_s, optimized.pct_of_roofline
        );
        // The point of the metric is that it separates these two by a lot.
        assert!(
            optimized.micros * 3.0 < naive.micros,
            "expected a large speedup, got {:.1} us vs {:.1} us",
            optimized.micros,
            naive.micros
        );
    }

    #[test]
    fn a_kernel_that_skips_the_max_subtraction_is_rejected() {
        let Some(mut gpu) = gpu() else { return };
        let plan = KernelPlan {
            source: r#"
extern "C" __global__ void softmax(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* in = input + (size_t)row * cols;
    float* out = output + (size_t)row * cols;
    float sum = 0.0f;
    for (int i = 0; i < cols; ++i) sum += expf(in[i]);
    for (int i = 0; i < cols; ++i) out[i] = expf(in[i]) / sum;
}
"#
            .to_string(),
            ..naive_plan(TEST_ROWS)
        };
        let failure = gpu.evaluate(&plan).expect_err("inf/inf is not a softmax");
        assert!(
            matches!(failure, KernelFailure::Wrong { .. }),
            "{failure:?}"
        );
        assert!(!failure.poisons_context());
    }

    #[test]
    fn nvrtc_diagnostics_come_back_as_feedback() {
        let Some(mut gpu) = gpu() else { return };
        let plan = KernelPlan {
            source: "extern \"C\" __global__ void softmax(int a) { this is not c++ }".to_string(),
            ..naive_plan(TEST_ROWS)
        };
        let failure = gpu.evaluate(&plan).expect_err("that does not compile");
        let KernelFailure::Compile(log) = &failure else {
            panic!("{failure:?}")
        };
        assert!(log.contains("error"), "log without diagnostics: {log}");
    }

    #[test]
    fn unlaunchable_geometry_is_rejected_before_the_driver_sees_it() {
        let Some(mut gpu) = gpu() else { return };
        let too_many_threads = KernelPlan {
            block: (2048, 1, 1),
            ..naive_plan(TEST_ROWS)
        };
        assert!(matches!(
            gpu.evaluate(&too_many_threads),
            Err(KernelFailure::Geometry(_))
        ));

        let too_few_threads = KernelPlan {
            grid: (1, 1, 1),
            block: (32, 1, 1),
            ..naive_plan(TEST_ROWS)
        };
        assert!(matches!(
            gpu.evaluate(&too_few_threads),
            Err(KernelFailure::Geometry(_))
        ));
    }

    #[test]
    fn an_out_of_bounds_kernel_is_reported_as_poisoning_the_context() {
        let Some(mut gpu) = gpu() else { return };
        let plan = KernelPlan {
            source: r#"
extern "C" __global__ void softmax(const float* input, float* output, int rows, int cols) {
    output[(size_t)(blockIdx.x + 1) * 1000000000ull] = 1.0f;
}
"#
            .to_string(),
            ..naive_plan(TEST_ROWS)
        };
        let failure = gpu.evaluate(&plan).expect_err("that writes out of bounds");
        assert!(
            failure.poisons_context(),
            "an illegal access is sticky: {failure:?}"
        );

        // The device is unusable from here on. The naive kernel passed a
        // moment ago in `the_naive_kernel_is_correct`; now the very same plan
        // cannot run, and neither retaining the primary context nor creating
        // an independent one recovers it. That is why candidates are evaluated
        // in a child process instead of being caught in-place.
        assert!(
            gpu.evaluate(&naive_plan(TEST_ROWS)).is_err(),
            "if this ever starts passing, in-process recovery became possible"
        );
    }
}

// SPDX-License-Identifier: MPL-2.0
//! Evolving a CUDA kernel: the agent writes GPU code, the host measures it.
//!
//! The evolvable function returns a [`KernelPlan`] — CUDA C source plus the
//! grid/block geometry to launch it with. Each round the host compiles every
//! candidate with NVRTC, checks it against a CPU softmax oracle, times it, and
//! reports the achieved memory throughput as a percentage of the device's
//! measured copy bandwidth. The best kernel is activated; the rest are kept as
//! revisions.
//!
//! ## Why a GPU kernel is a good subject for an evolution loop
//!
//! The fitness function is not a matter of taste. A row-wise softmax has to
//! move `2 * rows * cols * 4` bytes and nothing else, so there is a hard
//! ceiling — measured here as a device-to-device copy of the same buffers —
//! and every candidate can be scored as a fraction of it. The naive kernel the
//! run starts from sits at a few percent of that ceiling, and closing the gap
//! is a matter of *strategy* (coalescing, block-per-row reductions, warp
//! shuffles, vectorized loads, fast-math intrinsics), not of micro-edits. That
//! is exactly the shape of problem [`Runtime::evolve_batch`] exists for: many
//! genuinely different candidates per round, each scored, only the winner
//! activated.
//!
//! ## What runs where
//!
//! Agent code stays inside symbiont's policy: no `unsafe`, no statics, no FFI.
//! It emits *text* and typed launch parameters. Everything that needs a device
//! pointer lives in the host crate behind `Gpu`.
//!
//! Candidate kernels are then launched in a **child process**, one per
//! candidate. Symbiont contains a misbehaving CPU implementation in-process by
//! wrapping the dylib's functions in `catch_unwind`; a GPU kernel cannot be
//! contained that way, because an illegal memory access is a sticky CUDA error
//! that leaves the whole process unable to create *any* context afterwards.
//! The process boundary is the GPU's `catch_unwind`. See
//! `cuda_softmax_example::Isolated`.
//!
//! ## Running
//!
//! ```bash
//! MODEL=... cargo run -p cuda-softmax-example --release
//! ```
//!
//! `ROUNDS` (default 3) and `LANES` (default 4) size the search. Without a
//! CUDA device the example reports that and exits successfully, unless
//! `STRICT=1` is set.

#![allow(
    unused_crate_dependencies,
    reason = "cudarc is used by this package's library target."
)]

use std::fmt::Write;

use cuda_softmax_example::{
    COLS,
    DeviceInfo,
    Gpu,
    Isolated,
    KERNEL_SIGNATURE,
    Measurement,
    ROWS,
    prelude::*,
};
use symbiont::{
    DylibConfig,
    Revision,
    Runtime,
};
use tracing::{
    info,
    warn,
};

symbiont::evolvable! {
    /// Produce the CUDA kernel that computes a row-wise softmax, and the
    /// geometry to launch it with.
    ///
    /// The host compiles `source` with NVRTC, looks up the `softmax` symbol,
    /// and launches it once per measurement with
    /// `(input, output, rows, cols)`. `input` and `output` are row-major
    /// `rows * cols` `f32` matrices in device memory; `output[r][c]` must be
    /// `exp(input[r][c] - max(row r)) / sum(exp(input[r][*] - max(row r)))`.
    ///
    /// Correctness is checked against a CPU oracle before any timing is
    /// reported, and the input contains per-row biases up to ~90, so the
    /// max-subtraction is not optional: `expf(90.0f)` is `inf` in single
    /// precision.
    fn plan(rows: usize, cols: usize) -> KernelPlan {
        // One thread per row: correct, and about as slow as a softmax gets.
        KernelPlan {
            source: NAIVE_KERNEL.to_string(),
            grid: ((rows as u32).div_ceil(256), 1, 1),
            block: (256, 1, 1),
            shared_bytes: 0,
        }
    }
}

/// Output cap per lane. Kernels plus a short explanation fit comfortably; the
/// wall clock of a batch is its slowest lane, so an unbounded rambler holds up
/// the whole round.
const MAX_OUTPUT_TOKENS: u64 = 3072;

/// One strategy hint per lane, appended last so every lane shares the same
/// (cacheable) prompt prefix. Each one points at a different optimization
/// axis, which is what makes the candidates worth comparing.
const STRATEGIES: &[&str] = &[
    "Assign one block per row and reduce the maximum and the sum across the \
     whole block — warp shuffles first, then combine the per-warp partials \
     through shared memory — so consecutive threads read consecutive columns.",
    "Assign one warp (32 threads) per row and use only warp shuffles — no \
     shared memory, no __syncthreads.",
    "Use vectorized float4 loads and stores; cols is a multiple of 4, so each \
     thread can process four contiguous columns per access.",
    "Stage the row in shared memory on the first pass so the exponentials are \
     computed without re-reading global memory.",
    "Use the online (single-pass) softmax formulation: keep a running maximum \
     and a running sum, rescaling the sum by exp(old_max - new_max) whenever \
     the maximum grows.",
    "Use the fast-math intrinsics __expf and __fdividef, and multiply by a \
     precomputed reciprocal instead of dividing per element.",
    "Pick the block size from the row length and maximize occupancy; make \
     sure every global load is coalesced across the warp.",
    "Combine float4 vectorized loads with a warp-shuffle reduction and a \
     single fused write pass.",
];

/// The best candidate so far, and what it achieved.
struct Champion {
    revision: Revision,
    plan: KernelPlan,
    measurement: Measurement,
    label: String,
}

/// How one lane of a round turned out.
struct LaneOutcome {
    lane: usize,
    revision: Option<Revision>,
    plan: Option<KernelPlan>,
    measurement: Option<Measurement>,
    /// Rejection reason, phrased for the next round's prompt.
    failure: Option<String>,
    kind: &'static str,
}

impl LaneOutcome {
    fn rejected(lane: usize, revision: Option<Revision>, kind: &'static str, why: String) -> Self {
        Self {
            lane,
            revision,
            plan: None,
            measurement: None,
            failure: Some(why),
            kind,
        }
    }
}

/// Call the candidate's `plan`, then compile, verify and time its kernel in a
/// child process.
///
/// The `plan` call goes through a [`symbiont::RevisionFn`] handle, which pins
/// its own revision: candidates are measured without ever becoming the active
/// implementation, so a round can be evaluated in any order and nothing is
/// hot-swapped between measurements.
fn evaluate_lane(isolated: &Isolated, lane: usize, revision: Revision) -> LaneOutcome {
    let Some(handle) = plan_fn(revision) else {
        return LaneOutcome::rejected(lane, None, "unregistered", "revision not registered".into());
    };

    let plan = handle.get()(ROWS, COLS);
    if let Some(panic) = handle.take_panic() {
        return LaneOutcome::rejected(
            lane,
            Some(revision),
            "panicked",
            format!("`plan` panicked: {panic}"),
        );
    }

    match isolated.evaluate(&plan) {
        Ok(measurement) => LaneOutcome {
            lane,
            revision: Some(revision),
            plan: Some(plan),
            measurement: Some(measurement),
            failure: None,
            kind: "ok",
        },
        Err(failure) => LaneOutcome {
            lane,
            revision: Some(revision),
            plan: Some(plan),
            measurement: None,
            failure: Some(failure.to_string()),
            kind: failure.kind(),
        },
    }
}

/// The part of the prompt that never changes: task, contract, device.
///
/// Kept first and byte-identical across every lane and every round so a server
/// with prefix caching prefills it once for the whole run.
fn task_preamble(signature: &str, info: &DeviceInfo, roofline_gb_per_s: f64) -> String {
    format!(
        "You are optimizing a CUDA kernel. Implement this Rust function:\n\
         ```rust\n{signature}\n```\n\
         It returns a `KernelPlan {{ source, grid, block, shared_bytes }}`: the CUDA C source \
         of the kernel, and the geometry the host launches it with.\n\n\
         The kernel must have exactly this signature:\n\
         ```cuda\n{KERNEL_SIGNATURE}\n```\n\
         `input` and `output` are row-major {ROWS}x{COLS} f32 matrices in device memory. For \
         every row r and column c:\n\
         `output[r][c] = expf(input[r][c] - rowmax) / sum_c expf(input[r][c] - rowmax)`, where \
         `rowmax` is the maximum of row r.\n\n\
         Rules:\n\
         - The output is compared against a CPU reference before any timing is reported. Rows \
           contain per-row biases up to ~90, so subtracting the row maximum is mandatory: \
           `expf(90.0f)` is `inf` in f32.\n\
         - Every one of the {ROWS} rows must be written. `cols` is {COLS}, a multiple of 4, so \
           float4 loads are legal and aligned.\n\
         - The kernel must be `extern \"C\"` or the host cannot find the symbol.\n\
         - Out-of-bounds accesses are detected and cost the whole candidate; guard your indices.\n\
         - The comparison is strict (1e-6 absolute). The usual cause of a wrong answer is an \
           incomplete reduction: __shfl_down_sync only reduces within one warp, so a block of \
           more than 32 threads must combine the per-warp partials through shared memory before \
           the maximum or the sum is valid for the whole row.\n\
         - Every thread of a block must reach every __syncthreads(). Returning early from \
           out-of-range threads before a barrier deadlocks the block.\n\
         - Return the launch geometry that matches your kernel. A block-per-row kernel wants \
           `grid = (rows, 1, 1)`; a thread-per-row kernel wants `grid = (rows / block, 1, 1)`.\n\
         - Set `shared_bytes` only for `extern __shared__` arrays; statically sized \
           `__shared__` declarations need no dynamic allocation.\n\n\
         Target device: {} (compute capability {}.{}, {} SMs, up to {} threads and {} bytes of \
         shared memory per block, warp size {}).\n\
         This kernel is memory bound: it must read {ROWS}x{COLS} floats and write as many. A \
         device-to-device copy of that same traffic runs at {roofline_gb_per_s:.0} GB/s on this \
         card, which is the practical ceiling.\n",
        info.name,
        info.compute_capability.0,
        info.compute_capability.1,
        info.multiprocessors,
        info.max_threads_per_block,
        info.max_shared_memory_per_block,
        info.warp_size,
    )
}

/// The part of the prompt that changes between rounds: the incumbent and what
/// went wrong last time.
fn round_state(champion: &Champion, feedback: &str) -> String {
    let m = &champion.measurement;
    format!(
        "\nCurrent best kernel ({}): {:.0} GB/s, {:.1}% of the copy ceiling, {:.1} us per \
         launch, launched with grid {:?} block {:?} shared {} bytes:\n```cuda\n{}\n```\n\
         Beat it. A correct kernel that is slower than this one is not an improvement.\n{feedback}",
        champion.label,
        m.gb_per_s,
        m.pct_of_roofline,
        m.micros,
        champion.plan.grid,
        champion.plan.block,
        champion.plan.shared_bytes,
        champion.plan.source.trim(),
    )
}

/// Summarize the previous round's rejections so the next one does not repeat
/// them. Truncated per lane: NVRTC logs can run to dozens of lines.
fn feedback_from(outcomes: &[LaneOutcome]) -> String {
    let mut text = String::new();
    for outcome in outcomes.iter().filter(|o| o.failure.is_some()) {
        let why = outcome.failure.as_deref().unwrap_or_default();
        let why: String = why.lines().take(6).collect::<Vec<_>>().join(" | ");
        let _ = writeln!(
            text,
            "A previous attempt was rejected ({}): {}",
            outcome.kind,
            why.chars().take(400).collect::<String>()
        );
    }
    if text.is_empty() {
        String::new()
    } else {
        format!("\nRejected attempts from the last round — do not repeat these:\n{text}")
    }
}

/// Print one round's leaderboard.
fn print_report(outcomes: &[LaneOutcome], baseline: &Measurement) {
    println!("\n| Lane | Rev | Status         |     us |   GB/s | % ceiling | Speedup");
    println!("|------|-----|----------------|--------|--------|-----------|--------");
    for outcome in outcomes {
        let rev = outcome
            .revision
            .map_or_else(|| "-".to_string(), |r| r.to_string());
        match &outcome.measurement {
            Some(m) => println!(
                "| {:<4} | {rev:<3} | {:<14} | {:>6.1} | {:>6.0} | {:>8.1}% | {:>6.1}x",
                outcome.lane,
                "ok",
                m.micros,
                m.gb_per_s,
                m.pct_of_roofline,
                baseline.micros / m.micros,
            ),
            None => println!(
                "| {:<4} | {rev:<3} | {:<14} |      - |      - |         - |      -",
                outcome.lane, outcome.kind,
            ),
        }
    }
    for outcome in outcomes {
        if let Some(failure) = &outcome.failure {
            let first = failure.lines().next().unwrap_or_default();
            println!("  lane {}: {first}", outcome.lane);
        }
    }
}

/// Read a positive `usize` from the environment, or fall back to `default`.
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

/// Bring up the GPU, or explain why the example is skipping.
fn open_gpu() -> Option<Gpu> {
    match Gpu::new(ROWS, COLS) {
        Ok(gpu) => Some(gpu),
        Err(err) => {
            let strict = std::env::var_os("STRICT").is_some();
            assert!(
                !strict,
                "no usable CUDA device: {err} (STRICT was requested)"
            );
            println!(
                "Skipping: {err}\nThis example needs an NVIDIA GPU; everything else about it \
                 (including `cargo check`) works without one."
            );
            None
        }
    }
}

fn main() -> symbiont::Result<()> {
    // A child process staged by `Isolated` evaluates one kernel and exits; it
    // must never fall through into the evolution loop. Checked before anything
    // else, including the tokio runtime it would have no use for.
    if let Some(code) = cuda_softmax_example::evaluator_main() {
        std::process::exit(code);
    }
    search()
}

#[expect(
    clippy::too_many_lines,
    reason = "One linear walkthrough of the search — device, baseline, rounds, winner — which is what the example exists to show"
)]
#[tokio::main]
async fn search() -> symbiont::Result<()> {
    symbiont::init_tracing();

    // Probe the device before spending anything on inference or compilation.
    // The parent only ever runs the host's own copy kernel, so its context
    // stays healthy for the whole run.
    let Some(gpu) = open_gpu() else {
        return Ok(());
    };
    let isolated = Isolated::new()
        .expect("can stage evaluation child processes")
        // One ceiling for the whole run, measured here, so every candidate's
        // percentage refers to the same number.
        .with_roofline(gpu.roofline_gb_per_s());
    let info = gpu.info().clone();
    println!(
        "Device: {} (cc {}.{}, {} SMs). Benchmark: {ROWS}x{COLS} f32 softmax, {:.1} MiB of \
         traffic per launch.\nCopy ceiling: {:.0} GB/s.",
        info.name,
        info.compute_capability.0,
        info.compute_capability.1,
        info.multiprocessors,
        gpu.traffic_bytes() as f64 / (1024.0 * 1024.0),
        gpu.roofline_gb_per_s(),
    );

    // Debug: the evolved Rust is a few lines that build a string, so there is
    // nothing for the optimizer to do — all the performance is in the CUDA
    // text, which NVRTC compiles at evaluation time regardless.
    let host_crate = env!("CARGO_PKG_NAME");
    let runtime = Runtime::new(
        SYMBIONT_DECLS,
        SYMBIONT_PRELUDE,
        DylibConfig::host_package(
            symbiont::Profile::Debug,
            host_crate,
            env!("CARGO_MANIFEST_DIR"),
        ),
    )
    .await?;

    let model = std::env::var("MODEL").expect("the MODEL env var names the model slug");
    let agent = symbiont::agent_builder_from_env(Some(host_crate), &model)
        .await?
        .max_tokens(MAX_OUTPUT_TOKENS)
        .build();

    // -- Baseline ---------------------------------------------------------
    let baseline_outcome = evaluate_lane(&isolated, 0, Revision::INITIAL);
    let (Some(baseline), Some(baseline_plan)) =
        (baseline_outcome.measurement, baseline_outcome.plan)
    else {
        panic!(
            "the naive kernel must be correct: {:?}",
            baseline_outcome.failure
        );
    };
    println!(
        "\nBaseline (one thread per row): {:.1} us, {:.0} GB/s, {:.1}% of the copy ceiling.",
        baseline.micros, baseline.gb_per_s, baseline.pct_of_roofline
    );

    let mut champion = Champion {
        revision: Revision::INITIAL,
        plan: baseline_plan,
        measurement: baseline,
        label: "baseline".to_string(),
    };

    let preamble = task_preamble(&runtime.fn_sigs()[0], &info, gpu.roofline_gb_per_s());
    let rounds = env_usize("ROUNDS", 3);
    let lanes = env_usize("LANES", 4);
    let mut feedback = String::new();

    // -- Search -----------------------------------------------------------
    for round in 1..=rounds {
        let state = round_state(&champion, &feedback);
        let prompts = Vec::from_iter(STRATEGIES.iter().cycle().take(lanes).map(|hint| {
            format!("{preamble}{state}\nStrategy for this attempt: {hint}\nCode only.")
        }));

        println!("\n=== Round {round}/{rounds}: {lanes} candidate kernels ===");
        let started = std::time::Instant::now();
        let results = runtime.evolve_batch(&agent, &prompts).await;
        println!(
            "Generated, validated and compiled {} candidates in {:.1}s.",
            results.len(),
            started.elapsed().as_secs_f64()
        );

        let mut outcomes = Vec::with_capacity(results.len());
        for (lane, result) in results.into_iter().enumerate() {
            let outcome = match result {
                Ok(revision) => evaluate_lane(&isolated, lane, revision),
                Err(err) => LaneOutcome::rejected(lane, None, "no code", err.to_string()),
            };
            outcomes.push(outcome);
        }

        print_report(&outcomes, &baseline);
        feedback = feedback_from(&outcomes);

        // -- Commit to the winner, if it is one ---------------------------
        let improvement = outcomes
            .iter()
            .filter_map(|o| Some((o, o.measurement?)))
            .filter(|(_, m)| m.micros < champion.measurement.micros)
            .min_by(|a, b| a.1.micros.total_cmp(&b.1.micros));
        if let Some((outcome, measurement)) = improvement {
            let revision = outcome.revision.expect("a measured lane has a revision");
            runtime.activate_revision(revision)?;
            println!(
                "Activated revision {revision} from lane {}: {:.1} us ({:.1}x the baseline, \
                 {:.1}% of the ceiling).",
                outcome.lane,
                measurement.micros,
                baseline.micros / measurement.micros,
                measurement.pct_of_roofline,
            );
            champion = Champion {
                revision,
                plan: outcome.plan.clone().expect("a measured lane has a plan"),
                measurement,
                label: format!("round {round}, lane {}", outcome.lane),
            };
        } else {
            info!(
                "round {round} produced no improvement; keeping revision {}",
                champion.revision
            );
        }
    }

    // -- Result -----------------------------------------------------------
    println!(
        "\n=== Best kernel: {} ===\n{:.1} us, {:.0} GB/s, {:.1}% of the {:.0} GB/s copy ceiling, \
         {:.1}x faster than the naive baseline.\n```cuda\n{}\n```",
        champion.label,
        champion.measurement.micros,
        champion.measurement.gb_per_s,
        champion.measurement.pct_of_roofline,
        gpu.roofline_gb_per_s(),
        baseline.micros / champion.measurement.micros,
        champion.plan.source.trim(),
    );

    // Dispatch now runs the winner: a plain call, not a pinned handle, so this
    // really is the hot-swapped implementation.
    let active = plan(ROWS, COLS);
    assert_eq!(
        active, champion.plan,
        "the active revision must be the champion"
    );
    match isolated.evaluate(&active) {
        Ok(m) => println!(
            "Re-measured through the active dispatch pointer: {:.1} us.",
            m.micros
        ),
        Err(err) => warn!("the activated kernel failed on re-measurement: {err}"),
    }

    Ok(())
}

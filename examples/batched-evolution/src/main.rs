// SPDX-License-Identifier: MPL-2.0
//! Population search in one inference round: eight prompt variants, evolved
//! concurrently, benchmarked against each other, best one activated.
//!
//! The task is deliberately one where *strategy* dominates: counting primes
//! below `n`. Trial division, a sieve of Eratosthenes, a bit-packed sieve and
//! wheel factorization are all a few dozen lines and all obviously correct to
//! a model, but they span three orders of magnitude in runtime. Eight prompts
//! that differ only in a trailing strategy hint therefore produce eight
//! genuinely different programs rather than eight paraphrases of one — which
//! is what makes the comparison worth running.
//!
//! ## Why a batch instead of eight rounds
//!
//! Against a server that batches (vLLM, SGLang, or `llama-server --parallel n`)
//! the eight lanes are merged into shared forward passes, so the round costs
//! far less than eight sequential evolutions: decode is memory-bandwidth-bound,
//! so a batch of eight reads the model weights once per step instead of eight
//! times. Everything before the strategy hint is byte-identical across lanes
//! and is prefilled once when the server caches prefixes — a bigger effect for
//! hosts that document their crate API, either inline in the system prompt
//! (`DocMode::Inline`) or as the compact prelude index (`DocMode::IndexAndTools`).
//! `symbiont/benches/vllm/` has a concurrency sweep that measures both effects
//! against a containerized vLLM.
//!
//! ## What the batch does *not* do
//!
//! [`symbiont::Runtime::evolve_batch`] registers every candidate but activates
//! none of them. That is the whole point: eight implementations with no
//! fitness signal between them is not a hill to climb. The harness benchmarks
//! each candidate through a [`symbiont::RevisionFn`] handle — which pins its
//! own revision and bypasses the dispatch pointers entirely — and only then
//! commits to a winner with `activate_revision`.
//!
//! Whether a small model actually invents a sieve is a property of the model,
//! not of the harness, so a round where every lane fails is reported rather
//! than fatal. Set `STRICT=1` to require at least one correct implementation.

use std::time::{
    Duration,
    Instant,
};

use symbiont::{
    DocMode,
    Lane,
    Revision,
    Runtime,
};
use tracing::{
    info,
    warn,
};

/// `n` for the timed benchmark. Large enough that algorithmic choice
/// dominates, small enough that the naive baseline still finishes promptly.
const BENCH_N: u32 = 50_000;
/// Timed runs per candidate (odd for a clean median).
const BENCH_RUNS: usize = 5;
/// Inputs every candidate must get right, including the awkward small ones.
const CORRECTNESS_INPUTS: &[u32] = &[0, 1, 2, 3, 4, 10, 100, 1_000, 10_000];
/// Output cap per lane; see where it is applied in `main`.
const MAX_OUTPUT_TOKENS: u64 = 1536;

// Default: trial division by *every* smaller integer. Correct, and about as
// slow as counting primes can reasonably be — roughly O(n^2 / log n).
symbiont::evolvable! {
    fn count_primes_below(n: u32) -> u32 {
        let mut count = 0;
        let mut candidate = 2;
        while candidate < n {
            let mut is_prime = true;
            let mut divisor = 2;
            while divisor < candidate {
                if candidate % divisor == 0 {
                    is_prime = false;
                    break;
                }
                divisor += 1;
            }
            if is_prime {
                count += 1;
            }
            candidate += 1;
        }
        count
    }
}

/// The eight strategy hints. Each one steers a lane toward a different
/// algorithm family; the rest of every prompt is identical.
const STRATEGIES: &[&str] = &[
    "Keep it simple and obviously correct — a direct trial-division loop is fine.",
    "Test candidate divisors only up to the square root of the candidate.",
    "Handle 2 specially, then skip all even candidates and step divisors by 2.",
    "Use a sieve of Eratosthenes over a boolean array.",
    "Use a sieve of Eratosthenes, but pack the flags into bits to shrink the working set.",
    "Use a segmented sieve so the working set stays inside the CPU cache.",
    "Use wheel factorization modulo 6: only candidates of the form 6k-1 and 6k+1 can be prime.",
    "Avoid heap allocation entirely — prefer an allocation-free strategy.",
];

/// Reference counts, computed host-side with a plain sieve rather than
/// hardcoded, so the expected values cannot drift from the inputs.
fn reference_count(n: u32) -> u32 {
    if n < 3 {
        return 0;
    }
    let len = n as usize;
    let mut composite = vec![false; len];
    let mut count = 0;
    for candidate in 2..len {
        if composite[candidate] {
            continue;
        }
        count += 1;
        let mut multiple = candidate * candidate;
        while multiple < len {
            composite[multiple] = true;
            multiple += candidate;
        }
    }
    count
}

/// How one lane of the batch turned out.
struct LaneOutcome {
    lane: usize,
    /// `None` if the lane exhausted its retry budget.
    revision: Option<Revision>,
    /// Set when the lane never produced usable code.
    error: Option<String>,
    correct: bool,
    /// Median of [`BENCH_RUNS`] timed runs; only meaningful when `correct`.
    median: Duration,
    /// Panic message, if the candidate panicked during evaluation.
    panic: Option<String>,
}

impl LaneOutcome {
    fn failed(lane: usize, error: String) -> Self {
        Self {
            lane,
            revision: None,
            error: Some(error),
            correct: false,
            median: Duration::MAX,
            panic: None,
        }
    }
}

/// Build the prompt for one lane.
///
/// The shared part comes first and the per-lane hint last. That ordering is
/// what makes the batch cheap: prefix reuse on the server stops at the first
/// differing token, so a hint spliced into the middle — or a per-lane persona
/// prepended — would throw away the cached prefill of the (large) system
/// preamble and the task description alike.
fn prompt_for(signature: &str, strategy: &str) -> String {
    format!(
        "Implement this function, which returns how many prime numbers are \
         strictly less than `n`:\n\
         ```\n{signature}\n```\n\n\
         Rules:\n\
         - Implement the algorithm from scratch; do not call into any prime-related crate.\n\
         - `count_primes_below(0)`, `(1)` and `(2)` must all return 0.\n\
         - The function must never panic, overflow or index out of bounds.\n\
         - Optimize for speed at n = {BENCH_N}.\n\n\
         Strategy to use for this attempt: {strategy}\n\
         Code only."
    )
}

/// Benchmark one registered candidate through a pinned handle.
///
/// Nothing here touches the active revision, so every candidate is measured
/// against the same baseline without any hot-swapping between measurements.
/// Note that a panic inside a handle call lands in *that revision's* buffer —
/// `handle.take_panic()`, not `runtime.take_panic()`, which reads whichever
/// revision happens to be active.
fn evaluate(lane: usize, revision: Revision) -> LaneOutcome {
    let Some(handle) = count_primes_below_fn(revision) else {
        return LaneOutcome::failed(lane, "revision was not registered".to_string());
    };
    // Hoisted out of the loops: a bare fn pointer, no dispatch overhead.
    let candidate = handle.get();

    let mut outcome = LaneOutcome {
        lane,
        revision: Some(revision),
        error: None,
        correct: true,
        median: Duration::MAX,
        panic: None,
    };

    for &n in CORRECTNESS_INPUTS {
        let got = candidate(n);
        if let Some(msg) = handle.take_panic() {
            outcome.correct = false;
            outcome.panic = Some(msg);
            return outcome;
        }
        let want = reference_count(n);
        if got != want {
            outcome.correct = false;
            outcome.error = Some(format!("count_primes_below({n}) = {got}, expected {want}"));
            return outcome;
        }
    }

    // Hoisted: recomputing the reference sieve inside the loop would be pure
    // waste, and it must not sit between the two `Instant` reads anyway.
    let want = reference_count(BENCH_N);
    let mut times = Vec::with_capacity(BENCH_RUNS);
    for _ in 0..BENCH_RUNS {
        let start = Instant::now();
        let got = std::hint::black_box(candidate(std::hint::black_box(BENCH_N)));
        let elapsed = start.elapsed();
        if let Some(msg) = handle.take_panic() {
            outcome.correct = false;
            outcome.panic = Some(msg);
            return outcome;
        }
        if got != want {
            outcome.correct = false;
            outcome.error = Some(format!(
                "count_primes_below({BENCH_N}) = {got}, expected {want}"
            ));
            return outcome;
        }
        times.push(elapsed);
    }
    times.sort_unstable();
    outcome.median = times[times.len() / 2];
    outcome
}

fn format_duration(d: Duration) -> String {
    if d == Duration::MAX {
        return "-".to_string();
    }
    let us = d.as_micros();
    if us < 1_000 {
        format!("{us} us")
    } else {
        format!("{:.2} ms", d.as_secs_f64() * 1000.0)
    }
}

fn print_report(outcomes: &[LaneOutcome], baseline: Duration) {
    println!("\n| Lane | Revision | Correct | Median      | Speedup | Strategy");
    println!("|------|----------|---------|-------------|---------|---------");
    for o in outcomes {
        let speedup = if o.correct && o.median > Duration::ZERO {
            format!("{:.0}x", baseline.as_secs_f64() / o.median.as_secs_f64())
        } else {
            "-".to_string()
        };
        println!(
            "| {:<4} | {:<8} | {:<7} | {:>11} | {:>7} | {}",
            o.lane,
            o.revision
                .map_or_else(|| "-".to_string(), |r| r.to_string()),
            if o.correct { "yes" } else { "NO" },
            format_duration(o.median),
            speedup,
            STRATEGIES[o.lane],
        );
    }
    for o in outcomes {
        if let Some(panic) = &o.panic {
            println!("  lane {} panicked: {panic}", o.lane);
        } else if let Some(err) = &o.error {
            println!("  lane {} rejected: {err}", o.lane);
        }
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "One linear walkthrough of the batched round — baseline, batch, per-lane report, activation — which is what the example exists to show; splitting it would hide the order"
)]
#[tokio::main]
async fn main() -> symbiont::Result<()> {
    symbiont::init_tracing();

    // Release: the whole comparison is about generated-code speed, and the
    // optimizer is worth orders of magnitude on this workload.
    let runtime =
        Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, symbiont::Profile::Release).await?;
    let signature = &runtime.fn_sigs()[0];

    let model = std::env::var("MODEL").expect("the MODEL env var names the model slug");
    // Bound the output. A batch's wall clock is its *slowest* lane, so one
    // model that starts rambling — chat completions generate until the context
    // runs out unless told otherwise — holds up the whole round while its
    // siblings sit finished. Ample for these implementations.
    let agent = symbiont::agent_builder_from_env(None, DocMode::default(), &model, false)
        .await?
        .max_tokens(MAX_OUTPUT_TOKENS)
        .build();

    // -- Baseline ---------------------------------------------------------
    println!("\n=== Baseline: the default trial-division implementation ===");
    let baseline_outcome = evaluate(0, Revision::INITIAL);
    assert!(
        baseline_outcome.correct,
        "the default implementation must be correct: {:?} / {:?}",
        baseline_outcome.error, baseline_outcome.panic
    );
    let baseline = baseline_outcome.median;
    println!("n = {BENCH_N}: {}", format_duration(baseline));

    // -- One batched round ------------------------------------------------
    let prompts = Vec::from_iter(
        STRATEGIES
            .iter()
            .map(|strategy| prompt_for(signature, strategy)),
    );

    println!(
        "\n=== Evolving {} prompt variants concurrently ===",
        prompts.len()
    );
    let started = Instant::now();
    let results = runtime.evolve_batch(&agent, &prompts).await;
    let batch_time = started.elapsed();
    println!(
        "Batch of {} lanes finished in {:.1}s.",
        results.len(),
        batch_time.as_secs_f64(),
    );

    // Failures are attributed to the lane that produced them, so a host can
    // see which prompt variants are unproductive rather than just that
    // "something failed".
    let failures = runtime.take_evolve_failures();
    if !failures.is_empty() {
        println!("\nSelf-healing retries by lane:");
        for lane in 0..prompts.len() {
            let kinds = Vec::from_iter(
                failures
                    .iter()
                    .filter(|f| f.lane() == Lane::from(lane as u32))
                    .map(symbiont::EvolveFailure::kind),
            );
            if !kinds.is_empty() {
                println!("  lane {lane}: {kinds:?}");
            }
        }
    }

    // -- Evaluate every candidate, activating none of them ----------------
    let outcomes = Vec::from_iter(results.into_iter().enumerate().map(
        |(lane, result)| match result {
            Ok(info) => evaluate(lane, info.revision()),
            Err(e) => LaneOutcome::failed(lane, e.to_string()),
        },
    ));
    print_report(&outcomes, baseline);

    // Lanes that converged on identical source share a revision — a signal
    // that the hints are not diversifying the output as much as intended.
    let mut distinct = Vec::from_iter(outcomes.iter().filter_map(|o| o.revision));
    distinct.sort_unstable();
    distinct.dedup();
    info!(
        "{} lanes produced {} distinct implementations.",
        outcomes.len(),
        distinct.len()
    );

    // -- Commit to the winner ---------------------------------------------
    let Some(best) = outcomes
        .iter()
        .filter(|o| o.correct)
        .min_by_key(|o| o.median)
    else {
        let msg = format!(
            "No lane produced a correct implementation ({} lanes attempted).",
            outcomes.len()
        );
        assert!(
            std::env::var_os("STRICT").is_none(),
            "{msg} (STRICT was requested)"
        );
        warn!("{msg} The batched evolution pipeline itself ran fine.");
        return Ok(());
    };

    let winner = best.revision.expect("a correct lane has a revision");
    println!(
        "\n=== Activating lane {} (revision {winner}): {} ===",
        best.lane, STRATEGIES[best.lane],
    );
    runtime.activate_revision(winner)?;

    // Dispatch now runs the winner — verified through the plain call, not a
    // handle, so this really is the swapped-in implementation.
    let start = Instant::now();
    let count = count_primes_below(BENCH_N);
    let elapsed = start.elapsed();
    assert_eq!(count, reference_count(BENCH_N), "activated code is correct");
    println!(
        "Active implementation: {} primes below {BENCH_N} in {} ({:.0}x faster than the baseline).",
        count,
        format_duration(elapsed),
        baseline.as_secs_f64() / best.median.as_secs_f64(),
    );
    println!(
        "\nWinning implementation:\n```rust\n{}```",
        runtime.current_code()
    );

    Ok(())
}

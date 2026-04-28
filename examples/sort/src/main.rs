// SPDX-License-Identifier: MPL-2.0
//! Performance-driven sort evolution across multiple data distributions.
//!
//! The LLM must implement a sorting algorithm **from scratch** (no std
//! sort methods). The harness benchmarks the evolved implementation on
//! six data distributions and feeds per-distribution timings back to the
//! LLM each round.
//!
//! The default implementation is intentionally bubble sort — O(n²) and
//! painfully slow on the benchmark arrays. A naive quicksort written in
//! round 1 will be fast on random data but degrade to O(n²) on sorted
//! or reverse-sorted input (bad pivot selection). The per-distribution
//! breakdown guides the LLM toward better pivot strategies, hybrid
//! approaches (insertion sort for small partitions), and three-way
//! partitioning for inputs with many duplicates.
//!
//! This showcases symbiont's strength in **iterative performance
//! optimization**: the LLM writes real compiled code, the harness
//! benchmarks it at native speed, and concrete timing data drives each
//! subsequent evolution.

use std::time::{
    Duration,
    Instant,
};

use romu::Rng;
use symbiont::Runtime;
use tracing::{
    info,
    warn,
};
use tracing_subscriber::EnvFilter;

/// Number of elements in each benchmark array.
const ARRAY_LEN: usize = 10_000;
/// Number of runs per distribution (odd for clean median).
const BENCH_RUNS: usize = 21;

// Default: bubble sort — intentionally O(n²).
// The LLM must evolve this into something competitive.
symbiont::evolvable! {
    fn sort(data: &mut [f64], len: usize) {
        for i in 0..len {
            for j in 0..len.saturating_sub(1 + i) {
                if data[j] > data[j + 1] {
                    data.swap(j, j + 1);
                }
            }
        }
    }
}

// -- Data distributions --------------------------------------------------

const DISTRIBUTIONS: &[&str] = &[
    "random",
    "sorted",
    "reverse",
    "nearly_sorted",
    "few_unique",
    "pipe_organ",
];

fn generate(dist: &str, len: usize, rng: &Rng) -> Vec<f64> {
    match dist {
        "random" => (0..len).map(|_| rng.f64()).collect(),
        "sorted" => (0..len).map(|i| i as f64).collect(),
        "reverse" => (0..len).rev().map(|i| i as f64).collect(),
        "nearly_sorted" => {
            let mut v: Vec<f64> = (0..len).map(|i| i as f64).collect();
            // Swap ~5 % of adjacent pairs to perturb sorted order.
            for _ in 0..len / 20 {
                let i = rng.mod_usize(len - 1);
                v.swap(i, i + 1);
            }
            v
        }
        "few_unique" => (0..len).map(|_| rng.mod_usize(10) as f64).collect(),
        "pipe_organ" => {
            // 0, 1, 2, …, n/2, …, 2, 1, 0
            let half = len / 2;
            (0..half)
                .chain((0..len - half).rev())
                .map(|i| i as f64)
                .collect()
        }
        _ => unreachable!(),
    }
}

// -- Benchmarking --------------------------------------------------------

/// Pre-generated template + reference sort for one distribution.
struct DistBench {
    name: &'static str,
    template: Vec<f64>,
    reference: Vec<f64>,
}

/// Timing + correctness for one distribution.
struct BenchResult {
    name: &'static str,
    median: Duration,
    correct: bool,
    /// If the function panicked, the panic message.
    panic: Option<String>,
}

/// Generate fixed test data for each distribution (same data every round).
fn prepare_benchmarks(rng: &Rng) -> Vec<DistBench> {
    Vec::from_iter(DISTRIBUTIONS.iter().map(|&name| {
        let template = generate(name, ARRAY_LEN, rng);
        let mut reference = template.clone();
        reference.sort_unstable_by(|a, b| a.partial_cmp(b).expect("no NaN in test data"));
        DistBench {
            name,
            template,
            reference,
        }
    }))
}

/// Benchmark the current `sort` implementation on all distributions.
///
/// Panics in the evolvable function are caught inside the dylib and
/// retrieved via [`symbiont::catch_panic`] so a bad evolution round
/// doesn't crash the harness.
fn run_benchmarks(runtime: &Runtime, benches: &[DistBench]) -> Vec<BenchResult> {
    Vec::from_iter(benches.iter().map(|b| {
        let mut times = Vec::with_capacity(BENCH_RUNS);
        let mut correct = true;
        let mut panic_msg: Option<String> = None;

        for _ in 0..BENCH_RUNS {
            let mut data = b.template.clone();
            let start = Instant::now();
            sort(&mut data, ARRAY_LEN);
            if let Some(msg) = runtime.take_panic() {
                correct = false;
                panic_msg = Some(msg);
                break;
            } else {
                let elapsed = start.elapsed();
                times.push(elapsed);
                if data != b.reference {
                    correct = false;
                }
            }
        }

        times.sort();
        BenchResult {
            name: b.name,
            median: times.get(times.len() / 2).copied().unwrap_or_default(),
            correct,
            panic: panic_msg,
        }
    }))
}

// -- Reporting -----------------------------------------------------------

fn format_duration(d: Duration) -> String {
    let us = d.as_micros();
    if us < 1_000 {
        format!("{us} us")
    } else {
        format!("{:.2} ms", d.as_secs_f64() * 1000.0)
    }
}

fn format_report(results: &[BenchResult]) -> String {
    let mut report = String::from(
        "| Distribution   | Median Time   | Correct |\n\
         |----------------|---------------|---------|\n",
    );

    let mut total = Duration::ZERO;
    for r in results {
        total += r.median;
        report.push_str(&format!(
            "| {:<14} | {:>13} | {:<7} |\n",
            r.name,
            format_duration(r.median),
            if r.correct { "yes" } else { "NO" },
        ));
    }
    report.push_str(&format!(
        "| {:<14} | {:>13} |         |\n",
        "TOTAL",
        format_duration(total),
    ));

    if results.iter().any(|r| !r.correct) {
        report.push_str(
            "\nWARNING: Incorrect output on some distributions! Fix correctness first.\n",
        );
    }

    for r in results
        .iter()
        .filter_map(|r| r.panic.as_ref().map(|p| (r.name, p)))
    {
        report.push_str(&format!("\nPANIC on '{}': {}\n", r.0, r.1));
    }

    report
}

fn all_correct(results: &[BenchResult]) -> bool {
    results.iter().all(|r| r.correct)
}

fn collect_panics(results: &[BenchResult]) -> Vec<(&str, &str)> {
    Vec::from_iter(
        results
            .iter()
            .filter_map(|r| r.panic.as_deref().map(|msg| (r.name, msg))),
    )
}

fn total_time(results: &[BenchResult]) -> Duration {
    results.iter().map(|r| r.median).sum()
}

// -- Main ----------------------------------------------------------------

#[tokio::main]
async fn main() -> symbiont::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_line_number(true)
        .init();

    let runtime = Runtime::init(SYMBIONT_DECLS, symbiont::Profile::Release).await?;
    let fn_sigs = runtime.fn_sigs();
    info!("fn_sigs: {fn_sigs:?}");

    let agent = symbiont::inference::init_agent()?;

    // Fixed test data — identical across rounds for fair comparison.
    let rng = Rng::from_seed_with_64bit(42);
    let benches = prepare_benchmarks(&rng);

    // -- Round 0: benchmark the default bubble sort ----------------------
    println!("\n=== Round 0: default implementation (bubble sort) ===");
    let mut results = run_benchmarks(runtime, &benches);
    let mut report = format_report(&results);
    println!("{report}");

    // -- Evolution loop ---------------------------------------------------
    let max_rounds = 5;
    let orig_total = total_time(&results);
    let mut prev_code = String::new();
    let mut best_total = Duration::MAX;
    let mut best_code = String::new();

    for round in 1..=max_rounds {
        println!("\n=== Round {round}: evolving via LLM ===");

        // Core prompt: signature + rules + previous code (if any).
        // Kept lean so that backpressure retries (compilation errors)
        // don't carry stale benchmark noise.
        let prev_impl_section = if prev_code.is_empty() {
            String::new()
        } else {
            format!("Your previous implementation:\n```rust\n{prev_code}```\n\n")
        };
        let core_prompt = format!(
            "Implement this function that sorts the first `len` elements of `data` \
             in ascending order:\n\
             ```\n{sig}\n```\n\n\
             Rules:\n\
             - Do NOT use standard library sort methods \
             (.sort(), .sort_unstable(), .sort_by(), etc.)\n\
             - Implement the sorting algorithm from scratch\n\
             - Must produce correct ascending order for ALL distributions\n\n\
             {prev_impl_section}",
            sig = fn_sigs[0],
        );

        let panics = collect_panics(&results);
        let prompt = if !panics.is_empty() {
            // Panic-focused prompt: no benchmarks, just the crash info.
            let mut panic_report = String::new();
            for (dist, msg) in &panics {
                panic_report.push_str(&format!("- Distribution '{dist}': {msg}\n"));
            }
            format!(
                "{core_prompt}\
                 Runtime panics:\n{panic_report}\n\
                 Fix the panic. The function must not crash for any input. Code only.",
            )
        } else {
            // Normal optimization prompt with benchmark results.
            format!(
                "{core_prompt}\
                 Benchmark results ({ARRAY_LEN} elements, median of {BENCH_RUNS} runs):\n\
                 {report}\n\
                 Minimize the total time across all distributions. \
                 Focus on the slowest distributions first. Code only.",
            )
        };

        runtime
            .evolve(&agent, &prompt)
            .await
            .expect("evolution should succeed");

        prev_code = runtime
            .read_clean_code()
            .expect("failed to read generated code");

        results = run_benchmarks(runtime, &benches);
        let new_report = format_report(&results);
        println!("{new_report}");

        let new_total = total_time(&results);
        if all_correct(&results) {
            let speedup = orig_total.as_secs_f64() / new_total.as_secs_f64();
            info!(
                "Total: {} -> {} ({:.1}x {})",
                format_duration(orig_total),
                format_duration(new_total),
                if speedup >= 1.0 {
                    speedup
                } else {
                    1.0 / speedup
                },
                if speedup >= 1.0 { "faster" } else { "slower" },
            );

            if new_total < best_total {
                best_total = new_total;
                best_code = prev_code.clone();
                info!("New best: {}", format_duration(best_total));
            }
        } else {
            warn!("Incorrect output — LLM must fix correctness next round.");
        }

        report = new_report;
    }

    println!("\nEvolution complete after {max_rounds} rounds.");
    if best_code.is_empty() {
        println!("No correct implementation was found.");
    } else {
        println!(
            "Best implementation found ({max_rounds} iterations, original time: {}, new time: {}):\n```rust\n{best_code}```",
            format_duration(orig_total),
            format_duration(best_total),
        );
    }

    Ok(())
}

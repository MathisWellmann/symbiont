// SPDX-License-Identifier: MPL-2.0
//! Symbolic regression: discover the Rastrigin function from sample data.
//!
//! The ground-truth function is the 2-D Rastrigin:
//!
//!   f(x, y) = 20 + x² + y² − 10·cos(2πx) − 10·cos(2πy)
//!
//! The LLM never sees the name or formula. It receives a table of
//! `(x, y) → f(x, y)` sample points and must reverse-engineer the exact
//! symbolic expression through iterative evolution.
//!
//! Each round:
//!   1. Evaluate the current (hot-swapped) implementation on test points.
//!   2. Compute per-point absolute error and overall mean squared error.
//!   3. Feed the worst errors back to the LLM as context.
//!   4. Evolve — the harness handles compilation failures automatically.
//!
//! The example exits once MSE drops below a tight threshold (1e-10),
//! meaning the LLM found the exact symbolic formula, not just a numerical
//! approximation.

use std::f64::consts::PI;

use romu::Rng;
use symbiont::Runtime;
use tracing::{
    info,
    warn,
};
use tracing_subscriber::EnvFilter;

// The evolvable function starts with a trivial default (returns 0.0).
// The LLM must discover the Rastrigin formula purely from sample data.
symbiont::evolvable! {
    fn surface(x: f64, y: f64) -> f64 {
        let _ = (x, y);
        0.0
    }
}

/// Ground-truth 2-D Rastrigin function (never shown to the LLM).
#[inline(always)]
fn ground_truth(x: f64, y: f64) -> f64 {
    20.0 + x * x + y * y - 10.0 * (2.0 * PI * x).cos() - 10.0 * (2.0 * PI * y).cos()
}

/// A test point with its expected output.
struct Sample {
    x: f64,
    y: f64,
    expected: f64,
}

const SAMPLES_PER_AXIS: usize = 100;
const DOMAIN_MIN: f64 = -5.0;
const DOMAIN_MAX: f64 = 5.0;

/// Sweep x and y across the domain with `SAMPLES_PER_AXIS` evenly spaced
/// values per axis (100 x 100 = 10 000 points).
fn build_samples() -> Vec<Sample> {
    let mut samples = Vec::with_capacity(SAMPLES_PER_AXIS * SAMPLES_PER_AXIS);
    for xi in 0..SAMPLES_PER_AXIS {
        let x = DOMAIN_MIN + (DOMAIN_MAX - DOMAIN_MIN) * xi as f64 / (SAMPLES_PER_AXIS - 1) as f64;
        for yi in 0..SAMPLES_PER_AXIS {
            let y =
                DOMAIN_MIN + (DOMAIN_MAX - DOMAIN_MIN) * yi as f64 / (SAMPLES_PER_AXIS - 1) as f64;
            samples.push(Sample {
                x,
                y,
                expected: ground_truth(x, y),
            });
        }
    }
    samples
}

/// Evaluate the current implementation against the sample set.
/// Returns (mse, report_string).
fn evaluate(samples: &[Sample]) -> (f64, String) {
    let mut sum_sq_err = 0.0;
    let mut errors = Vec::<(usize, f64, f64, f64, f64)>::with_capacity(samples.len()); // (idx, x, y, expected, got)

    for (i, s) in samples.iter().enumerate() {
        let got = surface(s.x, s.y);
        let err = (got - s.expected).abs();
        sum_sq_err += (got - s.expected) * (got - s.expected);
        if err > 1e-10 {
            errors.push((i, s.x, s.y, s.expected, got));
        }
    }

    let mse = sum_sq_err / samples.len() as f64;

    let mut report = format!(
        "MSE: {mse:.6e}  ({} / {} points within tolerance)\n",
        samples.len() - errors.len(),
        samples.len(),
    );

    if errors.is_empty() {
        report.push_str("All points match the target function!");
    } else {
        // Sort by descending absolute error so the worst misses are first.
        errors.sort_by(|a, b| {
            (b.4 - b.3)
                .abs()
                .partial_cmp(&(a.4 - a.3).abs())
                .expect("no NaN errors")
        });
        report.push_str("Worst mismatches (x, y, expected, got):\n");
        for &(_, x, y, expected, got) in errors.iter().take(10) {
            report.push_str(&format!(
                "  f({x:>6.2}, {y:>6.2}) = {expected:>12.6}  got {got:>12.6}  (err {:.2e})\n",
                (got - expected).abs(),
            ));
        }
        if errors.len() > 10 {
            report.push_str(&format!("  ... and {} more\n", errors.len() - 10));
        }
    }

    (mse, report)
}

/// Number of randomly sampled points shown to the LLM each round.
const PROMPT_SAMPLES: usize = 50;

/// Pick a fresh random subset of the evaluation grid and format it as a
/// table the LLM can reason about. Re-sampling every round ensures the
/// LLM sees different data each iteration.
fn sample_table(samples: &[Sample]) -> String {
    use std::collections::HashSet;

    let rng = Rng::new();
    let mut chosen = HashSet::with_capacity(PROMPT_SAMPLES);
    while chosen.len() < PROMPT_SAMPLES {
        chosen.insert(rng.mod_usize(samples.len()));
    }

    let mut indices = Vec::<usize>::from_iter(chosen);
    indices.sort_unstable();

    let mut table = String::from("| x | y | f(x,y) |\n|---|---|---|\n");
    for i in indices {
        let s = &samples[i];
        table.push_str(&format!(
            "| {:.4} | {:.4} | {:.4} |\n",
            s.x, s.y, s.expected
        ));
    }
    table
}

#[tokio::main]
async fn main() -> symbiont::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_line_number(true)
        .init();

    let runtime = Runtime::init(SYMBIONT_DECLS, symbiont::Profile::Debug).await?;
    let fn_sigs = runtime.fn_sigs();
    info!("fn_sigs: {fn_sigs:?}");

    let agent = symbiont::inference::init_agent()?;
    let samples = build_samples();

    // Convergence threshold: MSE < 1e-10 means the formula is exact
    // (not just numerically close).
    let mse_threshold = 1e-10;

    // -- Round 0: evaluate the default (trivially wrong) implementation -
    println!("\n=== Round 0: default implementation ===");
    let (mse, report) = evaluate(&samples);
    println!("{report}");

    if mse < mse_threshold {
        println!("Default implementation already matches — nothing to evolve.");
        return Ok(());
    }

    // -- Evolution loop ---------------------------------------------------
    let max_rounds = 10;
    let mut prev_report = report;

    for round in 1..=max_rounds {
        println!("\n=== Round {round}: evolving via LLM ===");

        // Fresh random subset each round so the LLM sees new evidence.
        let data_table = sample_table(&samples);

        let prompt = format!(
            "Implement this function so that it matches the sample data exactly:\n\
             ```\n{sig}\n```\n\n\
             Sample data:\n{data_table}\n\
             Previous evaluation:\n{prev_report}\n\
             Find the exact symbolic formula. Do NOT use lookup tables or interpolation. \
             Code only.",
            sig = fn_sigs[0],
        );

        info!("Evolution prompt:\n{prompt}");

        runtime
            .evolve_with_backpressure(&agent, &prompt)
            .await
            .expect("evolution should succeed");

        let (new_mse, new_report) = evaluate(&samples);
        println!("{new_report}");

        if new_mse < mse_threshold {
            println!("Exact formula found after {round} evolution round(s)!\n");
            let code = runtime
                .read_clean_code()
                .expect("failed to read generated code");
            println!("Generated code:\n```rust\n{code}```");
            return Ok(());
        }

        warn!("MSE {new_mse:.6e} after round {round} — refining.");
        prev_report = new_report;
    }

    println!("\nDid not converge after {max_rounds} rounds.");
    Ok(())
}

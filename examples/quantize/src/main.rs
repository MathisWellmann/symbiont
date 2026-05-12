// SPDX-License-Identifier: MPL-2.0
//! Optimal quantization: discover the best compression–fidelity trade-off.
//!
//! The LLM must implement a function that quantizes f64 values into fewer
//! distinct levels, minimizing reconstruction error (MSE) while using as
//! few distinct output values as possible.
//!
//! The default implementation copies the input unchanged — perfect fidelity
//! but zero compression. The LLM must discover quantization schemes that
//! reduce the number of distinct output values while keeping MSE low.
//!
//! Each round:
//!   1. Call the evolvable `quantize` function on data from a specific distribution.
//!   2. Measure MSE (reconstruction error) and count distinct output values.
//!   3. Update the Pareto frontier of (num_levels, MSE) trade-offs.
//!   4. Feed the frontier and per-distribution results back to the LLM.
//!
//! The Pareto frontier tracks non-dominated (num_levels, MSE) pairs across
//! all rounds, showing the LLM which trade-offs have been achieved and
//! challenging it to push the frontier further.

use std::{
    collections::HashSet,
    fmt::Display,
};

use colorgrad::Gradient;
use plotters::prelude::*;
use romu::Rng;
use symbiont::Runtime;
use tracing::{
    info,
    warn,
};

/// Number of values per distribution.
const SAMPLE_LEN: usize = 10_000;

// Default: identity copy — zero error, zero compression.
// The LLM must evolve this into an actual quantization scheme.
symbiont::evolvable! {
    fn quantize(input: &[f64], len: usize, output: &mut [f64]) {
        for i in 0..len {
            output[i] = input[i];
        }
    }
}

// -- Data distributions ------------------------------------------------------

/// The specific input distribution to create a quantization for.
#[derive(Debug, Clone, Copy, derive_more::Display)]
#[expect(
    dead_code,
    reason = "Leave this in to manually play with input distributions."
)]
enum Distribution {
    Uniform,
    Gaussian,
    Bimodal,
    Laplacian,
    LogNormal,
}

impl Distribution {
    fn generate(&self, len: usize, rng: &Rng) -> Vec<f64> {
        use Distribution::*;
        match self {
            Uniform => Vec::from_iter((0..len).map(|_| rng.f64() * 2.0 - 1.0)),
            Gaussian => Vec::from_iter((0..len).map(|_| {
                let u1 = rng.f64().max(1e-15);
                let u2 = rng.f64();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            })),
            Bimodal => Vec::from_iter((0..len).map(|_| {
                let u1 = rng.f64().max(1e-15);
                let u2 = rng.f64();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                if rng.f64() < 0.5 {
                    -2.0 + 0.5 * z
                } else {
                    2.0 + 0.5 * z
                }
            })),
            Laplacian => Vec::from_iter((0..len).map(|_| {
                let u = rng.f64() - 0.5;
                -u.signum() * (1.0 - 2.0 * u.abs()).max(1e-15).ln()
            })),
            LogNormal => Vec::from_iter((0..len).map(|_| {
                let u1 = rng.f64().max(1e-15);
                let u2 = rng.f64();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                z.exp()
            })),
        }
    }
}

// -- Evaluation --------------------------------------------------------------

/// Result of evaluating the quantize function on one distribution.
struct EvalResult {
    distr: Distribution,
    mse: f64,
    num_distinct: usize,
    bits_per_value: f64,
    panic: Option<String>,
}

fn count_distinct(data: &[f64]) -> usize {
    let mut seen = HashSet::with_capacity(data.len());
    for &v in data {
        seen.insert(v.to_bits());
    }
    seen.len()
}

fn evaluate(runtime: &Runtime, input: &[f64], distr: Distribution) -> EvalResult {
    let mut output = vec![0.0f64; input.len()];
    quantize(input, input.len(), &mut output);
    match runtime.take_panic() {
        None => {
            let mse = input
                .iter()
                .zip(output.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                / input.len() as f64;
            let num_distinct = count_distinct(&output);
            let bits_per_value = required_bits(num_distinct);
            EvalResult {
                distr,
                mse,
                num_distinct,
                bits_per_value,
                panic: None,
            }
        }
        Some(msg) => EvalResult {
            distr,
            mse: f64::INFINITY,
            num_distinct: 0,
            bits_per_value: 0.0,
            panic: Some(msg),
        },
    }
}

// -- Pareto frontier ---------------------------------------------------------

/// A point on the Pareto frontier: (num_distinct, mse) from a given round,
/// together with the source code that produced it.
#[derive(Clone)]
struct ParetoPoint {
    round: usize,
    num_distinct: usize,
    mse: f64,
    code: String,
}

#[inline]
fn required_bits(num_distinct: usize) -> f64 {
    if num_distinct <= 1 {
        0.0
    } else {
        (num_distinct as f64).log2().ceil()
    }
}

/// Maintains the Pareto frontier of (num_distinct, mse) trade-offs.
///
/// A point is non-dominated if no other point has both fewer (or equal)
/// distinct values AND lower (or equal) MSE.
struct ParetoFrontier {
    points: Vec<ParetoPoint>,
}

impl ParetoFrontier {
    fn new() -> Self {
        Self { points: Vec::new() }
    }

    /// Add a candidate point. Returns true if it was added to the frontier
    /// (i.e. it is non-dominated).
    fn add(&mut self, point: ParetoPoint) -> bool {
        // Check if dominated by any existing point.
        let dominated = self.points.iter().any(|p| {
            p.num_distinct <= point.num_distinct
                && p.mse <= point.mse
                && (p.num_distinct < point.num_distinct || p.mse < point.mse)
        });
        if dominated {
            return false;
        }

        // Remove points dominated by the new one.
        self.points.retain(|p| {
            !(point.num_distinct <= p.num_distinct
                && point.mse <= p.mse
                && (point.num_distinct < p.num_distinct || point.mse < p.mse))
        });

        self.points.push(point);
        self.points.sort_by_key(|p| p.num_distinct);
        true
    }

    /// Snapshot the current frontier as (num_distinct, mse) pairs.
    fn snapshot(&self) -> Vec<(f64, f64)> {
        Vec::from_iter(self.points.iter().map(|p| (p.num_distinct as f64, p.mse)))
    }

    fn format_table(&self) -> String {
        if self.points.is_empty() {
            return String::from("(no valid points yet)\n");
        }
        let mut table = String::from(
            "| Distinct levels | Bits/value | MSE        | Round |\n\
             |-----------------|------------|------------|-------|\n",
        );
        for p in &self.points {
            let bits = if p.num_distinct <= 1 {
                0.0
            } else {
                (p.num_distinct as f64).log2().ceil()
            };
            table.push_str(&format!(
                "| {:>15} | {:>10.1} | {:>10.4e} | {:>5} |\n",
                p.num_distinct, bits, p.mse, p.round,
            ));
        }
        table
    }

    /// Format each frontier entry with its source code so the LLM can see
    /// *how* each trade-off was achieved.
    fn format_with_code(&self) -> String {
        if self.points.is_empty() {
            return String::from("(no valid points yet)\n");
        }
        let mut s = String::new();
        for p in &self.points {
            let bits = required_bits(p.num_distinct);
            s.push_str(&format!(
                "### {} distinct | {:.1} bits/value | MSE {:.4e} (round {})\n```rust\n{}\n```\n\n",
                p.num_distinct, bits, p.mse, p.round, p.code,
            ));
        }
        s
    }
}

// -- Reporting ---------------------------------------------------------------

impl Display for EvalResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "| Distribution | Distinct | Bits/val | MSE        |\n\
             |--------------|----------|----------|------------|\n",
        )?;

        if self.panic.is_some() {
            writeln!(
                f,
                "| {:<12} | PANIC    |          |            |",
                self.distr.to_string()
            )?;
        } else {
            writeln!(
                f,
                "| {:<12} | {:>8} | {:>8.1} | {:>10.4e} |",
                self.distr.to_string(),
                self.num_distinct,
                self.bits_per_value,
                self.mse,
            )?;
        }

        Ok(())
    }
}

// -- Plotting ----------------------------------------------------------------

/// Render the Pareto frontier progression to a PNG and display it with `viuer`.
fn plot_frontier_progression(
    history: &[(usize, Vec<(f64, f64)>)],
) -> Result<(), Box<dyn std::error::Error>> {
    // Collect all points to determine axis ranges.
    let all_points =
        Vec::<(f64, f64)>::from_iter(history.iter().flat_map(|(_, pts)| pts.iter().copied()));
    if all_points.is_empty() {
        println!("No Pareto frontier data to plot.");
        return Ok(());
    }

    let x_min = 16_f64;
    let x_max = 10_000_f64;
    let y_min = 0_f64;
    let y_max = 0.035;

    // Add some padding to the ranges.
    let x_lo = (x_min / 1.5).max(1.0);
    let x_hi = x_max * 1.5;
    let y_pad = (y_max - y_min).max(1e-10) * 0.08;

    let path = std::env::temp_dir().join("quantize_pareto.png");
    {
        let root = BitMapBackend::new(&path, (2048, 2048)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(
                "Pareto Frontier Progression",
                ("sans-serif", 22).into_font().with_color(BLACK),
            )
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(
                (x_lo..x_hi).log_scale().base(2.0),
                (y_min - y_pad)..(y_max + y_pad),
            )?;

        chart
            .configure_mesh()
            .x_desc("Distinct levels")
            .y_desc("MSE")
            .draw()?;

        let grad = colorgrad::preset::turbo();
        for (round, pts) in history {
            let color = grad.at(*round as f32 / history.len() as f32).to_rgba8();
            let color = RGBColor(color[0], color[1], color[2]);
            let label = format!("Round {round}");

            // Draw the frontier line + points for this generation.
            let mut sorted = pts.clone();
            sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite"));

            chart
                .draw_series(LineSeries::new(
                    sorted.iter().copied(),
                    color.stroke_width(2),
                ))?
                .label(&label)
                .legend(move |(x, y)| {
                    Rectangle::new([(x, y - 4), (x + 14, y + 4)], color.filled())
                });

            chart.draw_series(
                sorted
                    .iter()
                    .map(|&(x, y)| Circle::new((x, y), 4, color.filled())),
            )?;
        }

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .position(SeriesLabelPosition::UpperRight)
            .draw()?;

        root.present()?;
    }
    println!("Plot saved to: {}", path.display());

    // Display in terminal via viuer.
    let conf = viuer::Config {
        width: Some(80),
        absolute_offset: false,
        ..Default::default()
    };
    viuer::print_from_file(&path, &conf)?;

    Ok(())
}

// -- Main --------------------------------------------------------------------

#[tokio::main]
async fn main() -> symbiont::Result<()> {
    symbiont::init_tracing();

    let runtime =
        Runtime::init(SYMBIONT_DECLS, SYMBIONT_PRELUDE, symbiont::Profile::Release).await?;
    let fn_sigs = runtime.fn_sigs();
    info!("fn_sigs: {fn_sigs:?}");

    let agent = symbiont::inference::init_agent()?;

    // Fixed test data — identical across rounds for fair comparison.
    let rng = Rng::from_seed_with_64bit(42);
    let distr = Distribution::Laplacian;
    let dist_data = distr.generate(SAMPLE_LEN, &rng);

    let mut frontier = ParetoFrontier::new();

    // Frontier snapshots per round for the final progression plot.
    let mut frontier_history: Vec<(usize, Vec<(f64, f64)>)> = Vec::new();

    // -- Round 0: evaluate the default (identity copy) -----------------------
    println!("\n=== Round 0: default implementation (identity copy) ===");
    let mut result: EvalResult = evaluate(runtime, &dist_data, distr);
    println!("{result}");

    // Seed frontiers with round 0 data.
    if result.panic.is_none() {
        frontier.add(ParetoPoint {
            round: 0,
            num_distinct: result.num_distinct,
            mse: result.mse,
            code: "for i in 0..len {\n    output[i] = input[i];\n}".into(),
        });
        frontier_history.push((0, frontier.snapshot()));
    }

    // -- Evolution loop ------------------------------------------------------
    let max_rounds = 20;
    let mut prev_code = String::new();

    for round in 1..=max_rounds {
        println!("\n=== Round {round}: evolving via LLM ===");

        let sig = &fn_sigs[0];

        let prompt = if let Some(ref panic_msg) = result.panic {
            let last_attempt = if prev_code.is_empty() {
                String::new()
            } else {
                format!("## Last Attempt\n```rust\n{prev_code}\n```\n\n")
            };
            format!(
                "## Task\n\
                 Implement this function that quantizes `len` input values, writing \
                 the reconstructed (quantized) values into `output`:\n\
                 ```\n{sig}\n```\n\n\
                 ## Constraints\n\
                 - No external crates — only `std` and built-in operations.\n\
                 - `input` and `output` are the same length; `len` gives the count.\n\
                 - The function must not panic or crash for any input.\n\n\
                 {last_attempt}\
                 ## Runtime Panic\n\
                 - {panic_msg}\n\n\
                 Fix the panic. Rust code only.",
            )
        } else {
            let last_attempt = if prev_code.is_empty() {
                String::new()
            } else {
                format!(
                    "## Last Attempt\n\
                     ```rust\n{prev_code}\n```\n\
                     Result: {result}\n\n",
                )
            };
            format!(
                "## Task\n\
                 Implement this function that quantizes `len` input values, writing \
                 the reconstructed (quantized) values into `output`:\n\
                 ```\n{sig}\n```\n\n\
                 ## Constraints\n\
                 - No external crates — only `std` and built-in operations.\n\
                 - `input` and `output` are the same length; `len` gives the count.\n\n\
                 ## Goal\n\
                 Minimize the number of distinct values in `output` (compression) while \
                 minimizing reconstruction error (MSE).\n\
                 - Fewer distinct output values = better compression (fewer bits per value)\n\
                 - Lower MSE = better reconstruction quality\n\
                 - The ideal solution adaptively chooses quantization bin boundaries \
                 based on the input data distribution\n\n\
                 ## Current Frontier ({distr}, {SAMPLE_LEN} samples)\n\
                 {frontier}\n\
                 {last_attempt}\
                 ## Direction\n\
                 Push the Pareto frontier: find solutions with fewer distinct levels at \
                 the same or lower MSE, or lower MSE at the same number of levels. \
                 Rust code only.",
                frontier = frontier.format_with_code(),
            )
        };

        runtime
            .evolve(&agent, &prompt)
            .await
            .expect("evolution should succeed");

        prev_code = runtime.current_code();

        result = evaluate(runtime, &dist_data, distr);
        println!("{result}");

        // Update frontier.
        if result.panic.is_none() {
            frontier.add(ParetoPoint {
                round,
                num_distinct: result.num_distinct,
                mse: result.mse,
                code: prev_code.clone(),
            });
        }

        // Always snapshot the frontier state after each round.
        frontier_history.push((round, frontier.snapshot()));
    }

    // -- Summary -------------------------------------------------------------
    println!("\nEvolution complete after {max_rounds} rounds.");
    println!("Final aggregate Pareto frontier:");
    println!("{distr} frontier:");
    println!("{}", frontier.format_table());

    if !prev_code.is_empty() {
        println!("Last implementation:\n```rust\n{prev_code}```",);
    }

    // Plot the Pareto frontier progression across generations.
    if let Err(e) = plot_frontier_progression(&frontier_history) {
        warn!("Failed to render Pareto frontier plot: {e}");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plot_frontier() {
        let history = vec![(0, vec![(1.0, 2.0)])];
        plot_frontier_progression(&history).expect("Can plot")
    }
}

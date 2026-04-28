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
//!   1. Call the evolvable `quantize` function on data from multiple distributions.
//!   2. Measure MSE (reconstruction error) and count distinct output values.
//!   3. Update the Pareto frontier of (num_levels, MSE) trade-offs.
//!   4. Feed the frontier and per-distribution results back to the LLM.
//!
//! The Pareto frontier tracks non-dominated (num_levels, MSE) pairs across
//! all rounds, showing the LLM which trade-offs have been achieved and
//! challenging it to push the frontier further.

use std::collections::HashSet;

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
            let bits_per_value = if num_distinct <= 1 {
                0.0
            } else {
                (num_distinct as f64).log2().ceil()
            };
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

/// A point on the Pareto frontier: (num_distinct, mse) from a given round.
#[derive(Clone)]
struct ParetoPoint {
    round: usize,
    num_distinct: usize,
    mse: f64,
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
}

// -- Reporting ---------------------------------------------------------------

fn format_results(r: &EvalResult) -> String {
    let mut report = String::from(
        "| Distribution | Distinct | Bits/val | MSE        |\n\
         |--------------|----------|----------|------------|\n",
    );

    if r.panic.is_some() {
        report.push_str(&format!(
            "| {:<12} | PANIC    |          |            |\n",
            r.distr.to_string()
        ));
    } else {
        report.push_str(&format!(
            "| {:<12} | {:>8} | {:>8.1} | {:>10.4e} |\n",
            r.distr.to_string(),
            r.num_distinct,
            r.bits_per_value,
            r.mse,
        ));
    }

    if r.panic.is_some() {
        report.push_str(&format!(
            "\nPANIC on '{}': {}\n",
            r.distr.to_string(),
            r.panic.as_deref().expect("filtered for panic"),
        ));
    }

    report
}

// -- Plotting ----------------------------------------------------------------

/// Palette of colours for successive generations.
const PALETTE: &[RGBColor] = &[
    RGBColor(31, 119, 180),
    RGBColor(255, 127, 14),
    RGBColor(44, 160, 44),
    RGBColor(214, 39, 40),
    RGBColor(148, 103, 189),
    RGBColor(140, 86, 75),
    RGBColor(227, 119, 194),
    RGBColor(127, 127, 127),
    RGBColor(188, 189, 34),
    RGBColor(23, 190, 207),
    RGBColor(174, 199, 232),
];

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

    let x_min = all_points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let x_max = all_points.iter().map(|p| p.0).fold(0.0f64, f64::max);
    let y_min = all_points
        .iter()
        .map(|p| p.1)
        .filter(|y| y.is_finite())
        .fold(f64::INFINITY, f64::min);
    let y_max = all_points
        .iter()
        .map(|p| p.1)
        .filter(|y| y.is_finite())
        .fold(0.0f64, f64::max);

    // Add some padding to the ranges.
    let x_pad = (x_max - x_min).max(1.0) * 0.08;
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
                (x_min - x_pad)..(x_max + x_pad),
                (y_min - y_pad)..(y_max + y_pad),
            )?;

        chart
            .configure_mesh()
            .x_desc("Distinct levels")
            .y_desc("MSE")
            .draw()?;

        for (round, pts) in history {
            let color = PALETTE[*round % PALETTE.len()];
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

    let runtime = Runtime::init(SYMBIONT_DECLS, symbiont::Profile::Release).await?;
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
    let mut report = format_results(&result);
    println!("{report}");

    // Seed frontiers with round 0 data.
    if result.panic.is_none() {
        frontier.add(ParetoPoint {
            round: 0,
            num_distinct: result.num_distinct,
            mse: result.mse,
        });
        frontier_history.push((0, frontier.snapshot()));
    }

    // -- Evolution loop ------------------------------------------------------
    let max_rounds = 10;
    let mut prev_code = String::new();

    for round in 1..=max_rounds {
        println!("\n=== Round {round}: evolving via LLM ===");

        let prev_impl_section = if prev_code.is_empty() {
            String::new()
        } else {
            format!("Your previous implementation:\n```rust\n{prev_code}```\n\n")
        };

        let prompt = if let Some(panic_msg) = result.panic {
            let mut panic_report = String::new();
            panic_report.push_str(&format!("- {panic_msg}\n"));
            format!(
                "Implement this function that quantizes `len` input values, writing \
                 the reconstructed (quantized) values into `output`:\n\
                 ```\n{sig}\n```\n\n\
                 {prev_impl_section}\
                 Runtime panics:\n{panic_report}\n\
                 Fix the panic. The function must not crash for any input. Code only.",
                sig = fn_sigs[0],
            )
        } else {
            let frontier_section = {
                let mut s = String::from(
                    "Pareto frontier of best (distinct_levels, MSE) trade-offs across all rounds:\n",
                );
                s.push_str("\nPer-distribution frontiers:\n");
                s.push_str(&format!("\n{distr}:\n"));
                s.push_str(&frontier.format_table());
                s
            };

            format!(
                "Implement this function that quantizes `len` input values, writing \
                 the reconstructed (quantized) values into `output`:\n\
                 ```\n{sig}\n```\n\n\
                 Goal: minimize the number of distinct values in `output` (compression) \
                 while minimizing reconstruction error (MSE between input and output).\n\
                 - Fewer distinct output values = better compression (fewer bits per value)\n\
                 - Lower MSE = better reconstruction quality\n\
                 - The ideal solution adaptively chooses quantization bin boundaries \
                 based on the input data distribution\n\n\
                 {prev_impl_section}\
                 Current evaluation ({SAMPLE_LEN} values per distribution):\n\
                 {report}\n\
                 {frontier_section}\n\
                 Push the Pareto frontier: find solutions with fewer distinct levels at \
                 the same or lower MSE, or lower MSE at the same number of levels. Code only.",
                sig = fn_sigs[0],
            )
        };

        runtime
            .evolve(&agent, &prompt)
            .await
            .expect("evolution should succeed");

        prev_code = runtime
            .read_clean_code()
            .expect("failed to read generated code");

        result = evaluate(runtime, &dist_data, distr);
        report = format_results(&result);
        println!("{report}");

        // Update frontier
        if result.panic.is_none() {
            frontier.add(ParetoPoint {
                round,
                num_distinct: result.num_distinct,
                mse: result.mse,
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

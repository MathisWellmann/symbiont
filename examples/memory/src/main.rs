// SPDX-License-Identifier: MPL-2.0
//! Population search that remembers across rounds, with `symbiont-memory`.
//!
//! The runtime forgets every `evolve` call on purpose. This example keeps
//! what the population learned in a [`Memory`] and renders it into the next
//! round's prompts:
//!
//! - the **archive** of every evaluated attempt (the best ones with source,
//!   the newest ones with their report),
//! - **known pitfalls**, counted from the lanes' evolution traces and the
//!   panics caught during evaluation,
//! - an **ACE playbook** of lessons, curated by one reflection call per
//!   island and round, with helpful/harmful counters,
//! - two **islands**, so the lanes do not all chase the same program, with
//!   ring migration every other round, and a **novelty** check that skips
//!   near copies before evaluating them.
//!
//! The task is symbolic regression of a hidden 1-D function from samples:
//! cheap to evaluate, with a graded score, so every round produces a
//! meaningful ranking for the memory to learn from.
//!
//! Prompts put the memory's shared part first, then the island part, then
//! the lane part, so a server with a prefix cache reuses as much as it can.

use std::{
    env::var,
    fmt::Write as _,
    hash::{
        DefaultHasher,
        Hash,
        Hasher,
    },
};

use symbiont::{
    DocMode,
    EvolutionAgent,
    EvolveError,
    EvolveInfo,
    Revision,
    Runtime,
};
use symbiont_memory::{
    Evaluated,
    IslandConfig,
    IslandId,
    Memory,
    MemoryConfig,
    REFLECTION_SYSTEM_PROMPT,
    ReflectionConfig,
    Rendered,
    Score,
};
use tracing::{
    info,
    warn,
};

symbiont::evolvable! {
    fn model(x: f64) -> f64 {
        let _ = x;
        0.0
    }
}

/// The function to rediscover. The agent only ever sees samples of it.
fn hidden(x: f64) -> f64 {
    x * (1.5 * x).sin() + 0.25 * x * x
}

const LANES: usize = 4;
const ISLANDS: u16 = 2;
const ROUNDS: usize = 6;
const MIGRATE_EVERY: usize = 2;
const SAMPLES: usize = 64;
/// An MSE below this means the formula is exact up to float noise.
const SOLVED_MSE: f64 = 1e-12;
/// One hint per lane, last in the prompt so the prefix before it is shared.
const HINTS: [&str; LANES] = [
    "Try a combination of polynomial terms.",
    "Try trigonometric terms, possibly multiplied by powers of x.",
    "Start from the best attempt so far and fix its largest errors.",
    "Look at where the errors grow with |x| and model that growth.",
];

/// One evaluated candidate.
struct Outcome {
    mse: f64,
    report: String,
    panic: Option<String>,
}

fn samples() -> Vec<(f64, f64)> {
    Vec::from_iter((0..SAMPLES).map(|i| {
        let x = -4.0 + 8.0 * (i as f64) / ((SAMPLES - 1) as f64);
        (x, hidden(x))
    }))
}

/// Run a registered candidate on every sample through its pinned handle.
fn evaluate(revision: Revision, samples: &[(f64, f64)]) -> Outcome {
    let handle = model_fn(revision).expect("evolve_batch registered the revision");
    let candidate = handle.get();
    let mut errors = Vec::with_capacity(samples.len());
    for &(x, want) in samples {
        let got = candidate(x);
        if let Some(panic) = handle.take_panic() {
            let report = format!("Panicked at x = {x}: {panic}");
            return Outcome {
                mse: f64::INFINITY,
                report,
                panic: Some(panic),
            };
        }
        errors.push((x, want, got));
    }
    let mse = errors.iter().map(|(_, w, g)| (w - g).powi(2)).sum::<f64>() / errors.len() as f64;
    errors.sort_by(|a, b| (b.1 - b.2).abs().total_cmp(&(a.1 - a.2).abs()));
    let mut report = format!("MSE {mse:.3e}. Largest errors:\n");
    for (x, want, got) in errors.iter().take(4) {
        writeln!(report, "- x = {x:.3}: want {want:.4}, got {got:.4}").expect("String write");
    }
    Outcome {
        mse,
        report,
        panic: None,
    }
}

/// Higher is better and finite: digits of precision, capped by float noise.
/// A panic or a non-finite MSE scores like a constant-zero guess.
fn score_of(mse: f64) -> Score {
    let mse = if mse.is_finite() { mse } else { 1e6 };
    Score::new(-(mse + 1e-15).log10()).expect("finite by construction")
}

/// The host-side id of a candidate: a hash of its source. Revision ids are
/// local to one process; a content hash is what dedups across rounds.
fn source_id(source: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    hasher.finish()
}

fn island_of(lane: usize) -> IslandId {
    IslandId::new(u16::try_from(lane).expect("few lanes") % ISLANDS)
}

fn prompt(task: &str, memory: &Rendered, hint: &str) -> String {
    format!(
        "{task}\n{}{}{}## This attempt\n{hint} Respond with Rust code only.",
        memory.shared(),
        memory.island(),
        memory.lane()
    )
}

/// Commit one lane's trace and, if it produced a novel candidate, its
/// evaluation. Returns the candidate's MSE and revision if it was evaluated.
fn absorb(
    memory: &mut Memory<u64>,
    runtime: &Runtime,
    result: &Result<EvolveInfo, EvolveError>,
    (lane, shown): (usize, &Rendered),
    samples: &[(f64, f64)],
) -> Option<(f64, Revision)> {
    let island = island_of(lane);
    let info = match result {
        Ok(info) => info,
        Err(e) => {
            memory.commit_trace(e.trace());
            warn!("Lane {lane} produced no candidate: {e}");
            return None;
        }
    };
    memory.commit_trace(info.trace());
    let source = runtime
        .revision_code(info.revision())
        .expect("just registered");
    if let Some(nearest) = memory.novelty(island, &source)
        && !nearest.novel
    {
        info!(
            "Lane {lane}: {}% like {}, skipped as a near copy.",
            nearest.percent, nearest.seq
        );
        return None;
    }
    let outcome = evaluate(info.revision(), samples);
    if let Some(panic) = &outcome.panic {
        memory.commit_panic(panic);
    }
    println!(
        "Lane {lane} ({island}): {}",
        outcome.report.lines().next().unwrap_or_default()
    );
    let _ = memory.commit(
        Evaluated::builder()
            .id(source_id(&source))
            .source(source)
            .score(score_of(outcome.mse))
            .report(outcome.report)
            .island(island)
            .bullets_shown(shown.bullets().to_vec())
            .build(),
    );
    Some((outcome.mse, info.revision()))
}

/// One reflection call for `island`, if enough attempts await review.
async fn reflect(
    memory: &mut Memory<u64>,
    reflector: &symbiont::Agent,
    island: IslandId,
    task: &str,
) {
    if !memory.reflection_due(island) {
        return;
    }
    let Some(request) = memory.reflection_request(island, task) else {
        return;
    };
    let answer = match reflector
        .run_without_tools(request.user_prompt(), Vec::new())
        .await
    {
        Ok(run) => run.output,
        Err(e) => {
            warn!("Reflection on {island} failed: {e}; its attempts wait for the next round.");
            memory.abandon_reflection(request);
            return;
        }
    };
    match memory.apply_reflection(&request, &answer) {
        Ok(report) => println!(
            "Reflection on {island}: +{} lessons, {} rejected, tags {}/{}/{} (helpful/harmful/neutral), {} pruned.",
            report.added().len(),
            report.rejected().len(),
            report.helpful(),
            report.harmful(),
            report.neutral(),
            report.pruned().len(),
        ),
        Err(e) => {
            warn!("Reflection answer on {island} unusable: {e}");
            memory.abandon_reflection(request);
        }
    }
}

async fn agents() -> symbiont::Result<(symbiont::Agent, symbiont::Agent)> {
    let base_url = var("BASE_URL").unwrap_or_default();
    let api_key = var("API_KEY").unwrap_or_default();
    let model = var("MODEL").expect("the MODEL env var names the model slug");
    let builder = async || {
        symbiont::agent_builder(None, DocMode::Inline, &base_url, &api_key, &model, false).await
    };
    let coder = symbiont::Agent::new(builder().await?.max_tokens(4096).build(), &base_url, &model);
    // The same endpoint with the reflection prompt as its preamble.
    let reflector = symbiont::Agent::new(
        builder()
            .await?
            .preamble(REFLECTION_SYSTEM_PROMPT)
            .max_tokens(4096)
            .build(),
        &base_url,
        &model,
    );
    Ok((coder, reflector))
}

#[tokio::main]
async fn main() -> symbiont::Result<()> {
    symbiont::init_tracing();
    let runtime = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, symbiont::Profile::Debug).await?;
    runtime.set_max_in_flight(u16::try_from(LANES).expect("few lanes"));
    let (coder, reflector) = agents().await?;
    let samples = samples();
    let mut table = String::new();
    for (x, y) in samples.iter().step_by(4) {
        writeln!(table, "f({x:.3}) = {y:.6}").expect("String write");
    }
    let task = format!(
        "## Task\nImplement this function so that it reproduces a hidden function of x \
         on [-4, 4], ideally as its exact formula:\n```rust\n{}\n```\nSamples:\n{table}",
        runtime.fn_sigs()[0]
    );

    let mut memory = Memory::<u64>::new(MemoryConfig {
        islands: IslandConfig::new(ISLANDS, 1, 2, 95).expect("valid"),
        reflection: ReflectionConfig::new(LANES / usize::from(ISLANDS), 16, 3_000, 2),
        ..MemoryConfig::default()
    });
    let mut best: Option<(f64, Revision)> = None;

    for round in 1..=ROUNDS {
        println!("\n=== Round {round} ===");
        let rendered = Vec::from_iter((0..LANES).map(|lane| memory.render(island_of(lane), &[])));
        let prompts =
            Vec::from_iter((0..LANES).map(|lane| prompt(&task, &rendered[lane], HINTS[lane])));
        let results = runtime.evolve_batch(&coder, &prompts).await;
        for (lane, result) in results.iter().enumerate() {
            let evaluated = absorb(
                &mut memory,
                runtime,
                result,
                (lane, &rendered[lane]),
                &samples,
            );
            if let Some((mse, rev)) = evaluated
                && best.is_none_or(|(b, _)| mse < b)
            {
                best = Some((mse, rev));
            }
        }
        for island in memory.config().islands.ids() {
            reflect(&mut memory, &reflector, island, &task).await;
        }
        if round % MIGRATE_EVERY == 0 {
            let migration = memory.migrate();
            println!(
                "Migration: {} attempts moved, {} lessons promoted.",
                migration.migrated.len(),
                migration.promoted.len()
            );
        }
        if best.is_some_and(|(mse, _)| mse < SOLVED_MSE) {
            println!("Solved in round {round}.");
            break;
        }
    }

    let Some((mse, revision)) = best else {
        warn!("No lane produced an evaluable candidate; the memory pipeline itself ran.");
        return Ok(());
    };
    runtime.activate_revision(revision)?;
    println!(
        "\nBest candidate (MSE {mse:.3e}):\n```rust\n{}```",
        runtime.current_code()
    );
    println!(
        "\nFinal memory as the next lane on island 0 would see it:\n{}",
        memory.render(IslandId::new(0), &[]).text()
    );
    Ok(())
}

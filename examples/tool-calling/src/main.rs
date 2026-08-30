// SPDX-License-Identifier: MPL-2.0
//! Tool calling during evolution: the LLM must discover a hidden rule.
//!
//! Declares an evolvable `transform` function whose specification is
//! deliberately **withheld** from the prompt. The only way for the agent to
//! learn the expected behaviour is to call the registered `probe` tool, which
//! grants black-box access to the hidden reference function.
//!
//! This showcases rig's tool-calling loop inside `Runtime::evolve`:
//! the agent is built with `.tool(Probe)` and `.default_max_turns(..)`, rig
//! dispatches the tool calls internally, and symbiont only consumes the final
//! code block. Watch the `Tool call: probe(..)` log lines to see the agent
//! experimenting before it commits to an implementation.
//!
//! # Reading the trajectory afterwards
//!
//! Every round writes its [`symbiont::EvolutionTrace`] into the DeepSeek
//! Harness session store, so the whole exchange — system prompt, each `probe`
//! call with its arguments and result, every corrective nudge, and what the
//! harness decided — can be replayed in `dsh` instead of scraped out of the
//! log. A round always exports, including the round that failed.
//!
//! The store is `$SYMBIONT_DSH_SESSIONS`, else `$DSH_HOME/sessions`, else
//! `~/.dsh/sessions`. Each round owns a fixed session id, so a rerun replaces
//! its previous export rather than piling up.

use std::path::{
    Path,
    PathBuf,
};

use rig_core::tool::PortableTool;
use symbiont::{
    DocMode,
    DshSession,
    EvolutionTrace,
    Runtime,
};
use tracing::{
    info,
    warn,
};

// The default body is intentionally wrong — always returns 0.
// The LLM must evolve it to match the hidden rule, which it can only
// discover through the `probe` tool.
symbiont::evolvable! {
    fn transform(n: i64) -> i64 {
        let _ = n;
        0
    }
}

/// The hidden ground-truth rule. It is never shown to the agent; it is only
/// reachable through the [`Probe`] tool.
fn hidden_rule(n: i64) -> i64 {
    3_i64.saturating_mul(n).saturating_add(7)
}

/// Arguments for the [`Probe`] tool, deserialized from the model's JSON.
#[derive(serde::Deserialize)]
struct ProbeArgs {
    /// The input to evaluate the hidden reference function at.
    n: i64,
}

/// A tool granting the agent black-box access to [`hidden_rule`].
struct Probe;

impl PortableTool for Probe {
    const NAME: &'static str = "probe";

    type Error = std::convert::Infallible;
    type Args = ProbeArgs;
    type Output = i64;

    fn description(&self) -> String {
        "Evaluate the hidden reference function at any integer input `n` and \
          return its output. Call this with a few different inputs to discover \
          the underlying rule before implementing `transform`."
            .to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "The input to evaluate the hidden function at."
                }
            },
            "required": ["n"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let out = hidden_rule(args.n);
        info!("Tool call: probe(n = {}) -> {out}", args.n);
        Ok(out)
    }
}

/// Write one lane's trajectory into the DeepSeek Harness session store.
///
/// The trace holds every message of the round but not the system prompt, the
/// provider or the model — see [`symbiont::DshSession`] for why — so those
/// come from the caller. Exporting is best-effort: a failure here must not
/// take down a run whose evolution succeeded.
fn export_trace(trace: &EvolutionTrace, model: &str, round: u32) {
    let Some(root) = sessions_root() else {
        warn!("no session store to export the round-{round} trajectory to");
        return;
    };

    let cwd = std::env::current_dir().unwrap_or_default();
    let cwd = cwd.to_string_lossy();

    let session = DshSession::builder()
        .provider("local")
        .model(model)
        .cwd(&cwd)
        .session_id(format!("session-tool-calling-round{round}"))
        .build();

    match symbiont::export_dsh_session(trace, &session, &root) {
        Ok(path) => println!("Round {round} trajectory: {}", path.display()),
        Err(error) => warn!("could not export the round-{round} trajectory: {error}"),
    }
}

/// Where the DeepSeek Harness keeps its sessions.
///
/// `$SYMBIONT_DSH_SESSIONS` wins, so the export can be pointed at a scratch
/// directory instead of the real store. Otherwise it is `$DSH_HOME/sessions`,
/// and `~/.dsh/sessions` when `DSH_HOME` is unset.
fn sessions_root() -> Option<PathBuf> {
    if let Ok(root) = std::env::var("SYMBIONT_DSH_SESSIONS") {
        return Some(PathBuf::from(root));
    }
    if let Ok(home) = std::env::var("DSH_HOME") {
        return Some(Path::new(&home).join("sessions"));
    }
    std::env::var("HOME")
        .ok()
        .map(|home| Path::new(&home).join(".dsh").join("sessions"))
}

/// Run the test suite against the hidden rule and return (passed, total).
fn run_tests() -> (usize, usize) {
    let inputs = -10..=10_i64;
    let total = inputs.clone().count();
    let passed = inputs.filter(|&n| transform(n) == hidden_rule(n)).count();
    (passed, total)
}

#[tokio::main]
async fn main() -> symbiont::Result<()> {
    symbiont::init_tracing();

    let runtime = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, symbiont::Profile::Debug).await?;
    let fn_sigs = runtime.fn_sigs();
    info!("fn_sigs: {fn_sigs:?}");

    // Register the `probe` tool on the pre-configured builder.
    // `default_max_turns` must be >= 1, otherwise rig aborts the run with
    // `MaxTurnsError` as soon as the model chains tool calls.
    let model = std::env::var("MODEL").expect("the MODEL env var names the model slug");
    let agent = symbiont::agent_builder_from_env(None, DocMode::default(), &model, false)
        .await?
        .tool(Probe)
        .default_max_turns(10)
        .build();

    // -- Round 0: run the default (wrong) implementation ----------------
    println!("\n=== Round 0: default implementation ===");
    let (mut passed, mut total) = run_tests();
    println!("{passed}/{total} tests passed.");

    // The specification is deliberately absent from this prompt: the agent
    // has to call the `probe` tool to figure out what `transform` must do.
    let prompt = format!(
        "Implement this function:\n\
         ```\n{sig}\n```\n\n\
         It must reproduce a hidden reference function exactly. The rule is NOT \
         given here. Use the `probe` tool to query the hidden function with a \
         few inputs of your choosing, deduce the rule, then implement it.\n\n\
         Code only.",
        sig = fn_sigs[0],
    );

    // -- Evolution loop --------------------------------------------------
    let max_rounds = 5;
    for round in 1..=max_rounds {
        println!("\n=== Round {round}: evolving via LLM (tool calls enabled) ===");

        let trace = match runtime.evolve(&agent, &prompt).await {
            Ok(info) => info.into_trace(),
            Err(error) => {
                // The failed lane is the one worth reading, so it is exported
                // before the panic rather than lost with it.
                let (error, trace) = error.into_parts();
                export_trace(&trace, &model, round);
                panic!("evolution failed in round {round}: {error}");
            }
        };
        export_trace(&trace, &model, round);

        // Re-run tests with the newly hot-swapped implementation.
        (passed, total) = run_tests();
        println!("{passed}/{total} tests passed.");

        if passed == total {
            println!("Agent discovered the hidden rule after {round} round(s)!");
            return Ok(());
        }

        warn!("{passed}/{total} correct after round {round} — retrying.");
    }

    panic!("Did not converge after {max_rounds} rounds: {passed}/{total} tests passed.")
}

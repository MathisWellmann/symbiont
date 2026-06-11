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

use rig_core::{
    completion::ToolDefinition,
    tool::Tool,
};
use symbiont::Runtime;
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
    3 * n + 7
}

/// Arguments for the [`Probe`] tool, deserialized from the model's JSON.
#[derive(serde::Deserialize)]
struct ProbeArgs {
    /// The input to evaluate the hidden reference function at.
    n: i64,
}

/// A tool granting the agent black-box access to [`hidden_rule`].
struct Probe;

impl Tool for Probe {
    const NAME: &'static str = "probe";

    type Error = std::convert::Infallible;
    type Args = ProbeArgs;
    type Output = i64;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Evaluate the hidden reference function at any integer input `n` and \
                          return its output. Call this with a few different inputs to discover \
                          the underlying rule before implementing `transform`."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "The input to evaluate the hidden function at."
                    }
                },
                "required": ["n"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let out = hidden_rule(args.n);
        info!("Tool call: probe(n = {}) -> {out}", args.n);
        Ok(out)
    }
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
    let agent = symbiont::init_agent_builder(None)
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

        runtime
            .evolve(&agent, &prompt)
            .await
            .expect("evolution should succeed");

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

// SPDX-License-Identifier: MPL-2.0
//! Backpressure integration test: transient provider failures (connection
//! resets, rate limits) are retried with an *unmodified* prompt and do not
//! count against the self-healing budget.
//!
//! One test per binary: [`symbiont::Runtime`] is a process-wide singleton.
#![expect(
    unused_crate_dependencies,
    reason = "Integration tests don't use them all"
)]

mod common;

use common::{
    ScriptedAgent,
    Turn,
};
use rig_core::completion::{
    CompletionError,
    PromptError,
};
use symbiont::{
    Profile,
    Runtime,
};

const BASE_PROMPT: &str = "Implement the function. Code only.";

#[tokio::test]
#[tracing_test::traced_test]
async fn transient_http_error_is_retried_with_unmodified_prompt() {
    symbiont::evolvable! {
        fn bp_transient_step(counter: &mut usize) {
            *counter += 1;
        }
    };
    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init runtime");

    let agent = ScriptedAgent::new([
        // Attempt 1: connection-level failure -> transient, retried with backoff.
        Turn::Fail(PromptError::CompletionError(CompletionError::HttpError(
            rig_core::http_client::Error::Instance(Box::new(std::io::Error::other(
                "simulated connection reset",
            ))),
        ))),
        // Attempt 2: valid code -> success.
        Turn::reply(
            "```rust\npub fn bp_transient_step(counter: &mut usize) { *counter += 23; }\n```",
        ),
    ]);

    rt.evolve(&agent, BASE_PROMPT)
        .await
        .expect("evolution should succeed after one transient retry");

    assert_eq!(agent.calls(), 2, "exactly one transient retry expected");

    // Transient failures are not the LLM's fault: the prompt must be retried
    // verbatim, without any self-healing nudge appended.
    assert_eq!(agent.prompt(0), BASE_PROMPT);
    assert_eq!(
        agent.prompt(1),
        BASE_PROMPT,
        "transient retries must not modify the prompt"
    );

    // The hot-swapped implementation is live.
    let mut counter = 0;
    bp_transient_step(&mut counter);
    assert_eq!(counter, 23, "evolved implementation should be hot-swapped");
}

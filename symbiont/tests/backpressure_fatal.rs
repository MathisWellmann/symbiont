// SPDX-License-Identifier: MPL-2.0
//! Backpressure integration test: non-transient, non-recoverable agent errors
//! propagate immediately without burning the self-healing retry budget.
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
    Error,
    Profile,
    Runtime,
};

const BASE_PROMPT: &str = "Implement the function. Code only.";

#[tokio::test]
#[tracing_test::traced_test]
async fn fatal_agent_error_propagates_without_retry() {
    symbiont::evolvable! {
        fn bp_fatal_step(counter: &mut usize) {
            *counter += 1;
        }
    };
    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init runtime");

    // A provider error is neither transient nor self-healable by prompt
    // feedback, so evolve must give up immediately.
    let agent = ScriptedAgent::new([Turn::Fail(PromptError::CompletionError(
        CompletionError::ProviderError("simulated provider failure".to_string()),
    ))]);

    let err = rt
        .evolve(&agent, BASE_PROMPT)
        .await
        .expect_err("a fatal provider error must propagate");

    assert!(
        matches!(err, Error::RigPrompt(_)),
        "expected the rig error to propagate unchanged, got: {err}"
    );
    assert_eq!(
        agent.calls(),
        1,
        "fatal errors must not be retried with prompt feedback"
    );

    // The original implementation is untouched.
    let mut counter = 0;
    bp_fatal_step(&mut counter);
    assert_eq!(counter, 1, "failed evolution must not change the function");
}

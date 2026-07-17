// SPDX-License-Identifier: MPL-2.0
//! Backpressure integration test: a generated function with a mismatching
//! signature is rejected, the expected signature is fed back, and the agent
//! recovers.
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
use symbiont::{
    Profile,
    Runtime,
};

const BASE_PROMPT: &str = "Implement the function. Code only.";

#[tokio::test]
#[tracing_test::traced_test]
async fn signature_mismatch_is_fed_back_and_recovered_from() {
    symbiont::evolvable! {
        fn bp_sig_step(counter: &mut usize) {
            *counter += 1;
        }
    };
    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init runtime");

    let agent = ScriptedAgent::new([
        // Attempt 1: wrong parameter type -> `SignatureMismatch`.
        Turn::reply("```rust\npub fn bp_sig_step(counter: &mut u64) { *counter += 1; }\n```"),
        // Attempt 2: correct types with a renamed argument -> success.
        Turn::reply("```rust\npub fn bp_sig_step(_counter: &mut usize) { *_counter = 7; }\n```"),
    ]);

    rt.evolve(&agent, BASE_PROMPT)
        .await
        .expect("evolution should succeed after one self-healing retry");

    assert_eq!(agent.calls(), 2, "exactly one retry expected");

    let retry_prompt = agent.prompt(1);
    assert!(
        retry_prompt.starts_with(BASE_PROMPT),
        "retry prompt must start with the base prompt, got: {retry_prompt}"
    );
    assert!(
        retry_prompt.contains("Signature mismatch in"),
        "retry prompt must contain the signature-mismatch nudge, got: {retry_prompt}"
    );
    assert!(
        retry_prompt.contains("fn bp_sig_step(counter: &mut usize)"),
        "retry prompt must contain the expected signature, got: {retry_prompt}"
    );
    assert!(
        retry_prompt.contains("fn bp_sig_step(&mut u64)"),
        "retry prompt must name the generated signature, got: {retry_prompt}"
    );
    assert!(
        retry_prompt.contains("&mut u64"),
        "retry prompt must echo the offending generated code, got: {retry_prompt}"
    );

    // The hot-swapped implementation is live.
    let mut counter = 0;
    bp_sig_step(&mut counter);
    assert_eq!(counter, 7, "evolved implementation should be hot-swapped");
}

// SPDX-License-Identifier: MPL-2.0
//! Backpressure integration test: generated code containing an `unsafe`
//! block is rejected at validation time (before compiling), the forbidden
//! construct is fed back, and the agent recovers with safe code.
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
#[cfg_attr(
    miri,
    ignore = "compiles and dlopens dylibs, which Miri does not support"
)]
#[tracing_test::traced_test]
async fn unsafe_code_is_rejected_and_recovered_from() {
    symbiont::evolvable! {
        fn bp_unsafe_step(counter: &mut usize) {
            *counter += 1;
        }
    };
    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init runtime");

    let agent = ScriptedAgent::new([
        // Attempt 1: an unsafe block -> rejected at validation, no compile.
        Turn::reply(
            "```rust\npub fn bp_unsafe_step(counter: &mut usize) { unsafe { *(counter as *mut usize) += 1; } }\n```",
        ),
        // Attempt 2: safe code -> success.
        Turn::reply("```rust\npub fn bp_unsafe_step(counter: &mut usize) { *counter = 9; }\n```"),
    ]);

    rt.evolve(&agent, BASE_PROMPT)
        .await
        .expect("evolution should succeed after one self-healing retry");

    assert_eq!(agent.calls(), 2, "exactly one retry expected");

    let retry_prompt = agent.prompt(1);
    assert!(
        !retry_prompt.contains(BASE_PROMPT),
        "retry prompt must contain only the correction, got: {retry_prompt}"
    );
    assert!(
        retry_prompt.contains("unsafe code is forbidden"),
        "retry prompt must contain the unsafe nudge, got: {retry_prompt}"
    );
    assert!(
        retry_prompt.contains("an `unsafe` block"),
        "retry prompt must name the offending construct, got: {retry_prompt}"
    );

    // The failure record is drained with the `unsafe` kind.
    let failures = rt.take_evolve_failures();
    assert_eq!(failures.len(), 1);
    assert_eq!(failures[0].kind(), "unsafe");
    assert!(failures[0].generated_code().contains("unsafe"));

    // The hot-swapped safe implementation is live.
    let mut counter = 0;
    bp_unsafe_step(&mut counter);
    assert_eq!(counter, 9, "evolved implementation should be hot-swapped");
}

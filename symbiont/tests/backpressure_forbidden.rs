// SPDX-License-Identifier: MPL-2.0
//! Backpressure integration test: generated code containing a forbidden
//! construct (a `static` item) is rejected at validation time (before
//! compiling), the reason is fed back, and the agent recovers.
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
async fn forbidden_construct_is_rejected_and_recovered_from() {
    symbiont::evolvable! {
        fn bp_forbidden_step(counter: &mut usize) {
            *counter += 1;
        }
    };
    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init runtime");

    let agent = ScriptedAgent::new([
        // Attempt 1: dylib-local static state -> rejected at validation.
        Turn::reply(
            "```rust\nstatic CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);\npub fn bp_forbidden_step(counter: &mut usize) { CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed); *counter += 1; }\n```",
        ),
        // Attempt 2: host-owned state only -> success.
        Turn::reply(
            "```rust\npub fn bp_forbidden_step(counter: &mut usize) { *counter = 11; }\n```",
        ),
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
        retry_prompt.contains("which is forbidden in evolvable code"),
        "retry prompt must contain the forbidden nudge, got: {retry_prompt}"
    );
    assert!(
        retry_prompt.contains("a `static` item"),
        "retry prompt must name the offending construct, got: {retry_prompt}"
    );
    assert!(
        retry_prompt.contains("host-owned"),
        "retry prompt must explain the reason, got: {retry_prompt}"
    );

    // The failure record is drained with the `forbidden` kind.
    let failures = rt.take_evolve_failures();
    assert_eq!(failures.len(), 1);
    assert_eq!(failures[0].kind(), "forbidden");
    assert!(failures[0].generated_code().contains("static CALLS"));

    // The hot-swapped implementation is live.
    let mut counter = 0;
    bp_forbidden_step(&mut counter);
    assert_eq!(counter, 11, "evolved implementation should be hot-swapped");
}

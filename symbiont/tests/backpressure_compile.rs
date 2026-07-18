// SPDX-License-Identifier: MPL-2.0
//! Backpressure integration test: code that parses and matches the signature
//! but fails to compile gets the compiler diagnostics fed back, and the agent
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
async fn compile_failure_feeds_compiler_diagnostics_back() {
    symbiont::evolvable! {
        fn bp_compile_step(counter: &mut usize) {
            *counter += 1;
        }
    };
    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init runtime");

    let agent = ScriptedAgent::new([
        // Attempt 1: valid syntax + correct signature, but a type error
        // (E0308) plus an unused variable that would emit a warning.
        Turn::reply(
            "```rust\npub fn bp_compile_step(counter: &mut usize) {\n    \
             let unused_noise = 42;\n    \
             let wrong: usize = \"definitely not a usize\";\n    \
             *counter += wrong;\n}\n```",
        ),
        // Attempt 2: compiles -> success.
        Turn::reply("```rust\npub fn bp_compile_step(counter: &mut usize) { *counter += 9; }\n```"),
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
        retry_prompt.contains("failed to compile"),
        "retry prompt must contain the compile-failure nudge, got: {retry_prompt}"
    );
    assert!(
        retry_prompt.contains("mismatched types"),
        "retry prompt must contain the actual rustc diagnostics, got: {retry_prompt}"
    );
    assert!(
        retry_prompt.contains("definitely not a usize"),
        "retry prompt must echo the offending generated code, got: {retry_prompt}"
    );
    assert!(
        !retry_prompt.contains("warning"),
        "the generated crate allows all warnings so compiler feedback \
         surfaces only errors, got: {retry_prompt}"
    );

    // The hot-swapped implementation is live.
    let mut counter = 0;
    bp_compile_step(&mut counter);
    assert_eq!(counter, 9, "evolved implementation should be hot-swapped");
}

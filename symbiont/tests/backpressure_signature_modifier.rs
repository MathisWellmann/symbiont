// SPDX-License-Identifier: MPL-2.0
//! Backpressure integration test: ABI-incompatible signature modifiers are
//! named in feedback so the agent can remove them.
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
async fn signature_modifier_is_named_and_recovered_from() {
    symbiont::evolvable! {
        fn bp_modifier_step(counter: &mut usize) {
            *counter += 1;
        }
    };
    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init runtime");

    let agent = ScriptedAgent::new([
        Turn::reply(
            "```rust\npub async fn bp_modifier_step(counter: &mut usize) { *counter += 1; }\n```",
        ),
        Turn::reply(
            "```rust\npub fn bp_modifier_step(counter: &mut usize) { *counter = 13; }\n```",
        ),
    ]);

    rt.evolve(&agent, BASE_PROMPT)
        .await
        .expect("evolution should recover after removing `async`");

    let retry_prompt = agent.prompt(1);
    assert!(!retry_prompt.contains(BASE_PROMPT));
    assert!(retry_prompt.contains("async fn bp_modifier_step"));
    assert!(retry_prompt.contains("fn bp_modifier_step(counter: &mut usize)"));
    assert_eq!(agent.history_len(1), 2);

    let mut counter = 0;
    bp_modifier_step(&mut counter);
    assert_eq!(counter, 13);
}

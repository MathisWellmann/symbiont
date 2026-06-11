// SPDX-License-Identifier: MPL-2.0
//! Backpressure integration test: panics in evolved code are caught inside
//! the dylib and the panic message is retrievable via
//! [`symbiont::Runtime::take_panic`] for prompt feedback.
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
async fn panicking_evolved_code_is_caught_and_message_retrievable() {
    symbiont::evolvable! {
        fn bp_panic_step(counter: &mut usize) {
            *counter += 1;
        }
    };
    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init runtime");

    // A normal call leaves no panic behind.
    let mut counter = 0;
    bp_panic_step(&mut counter);
    assert_eq!(counter, 1);
    assert_eq!(rt.take_panic(), None, "no panic should be stored initially");

    // Evolve into an implementation that panics at runtime — code that is
    // valid, compiles, and only misbehaves when executed.
    let agent = ScriptedAgent::new([Turn::reply(
        "```rust\npub fn bp_panic_step(counter: &mut usize) {\n    \
         *counter += 1;\n    \
         panic!(\"intentional panic for backpressure test\");\n}\n```",
    )]);
    rt.evolve(&agent, BASE_PROMPT)
        .await
        .expect("evolution should succeed — the code is valid, it only panics at runtime");

    // The call must return normally: the panic is caught inside the dylib
    // and never unwinds across the `dlopen` boundary.
    let mut counter = 0;
    bp_panic_step(&mut counter);
    assert_eq!(counter, 1, "side effects before the panic are preserved");

    // The panic message is available exactly once for prompt feedback.
    let msg = rt
        .take_panic()
        .expect("the panic must have been recorded inside the dylib");
    assert!(
        msg.contains("intentional panic for backpressure test"),
        "panic message must be preserved, got: {msg}"
    );
    assert_eq!(
        rt.take_panic(),
        None,
        "the stored panic message must be cleared on read"
    );
}

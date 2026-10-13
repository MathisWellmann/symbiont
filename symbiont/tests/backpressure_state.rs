// SPDX-License-Identifier: MPL-2.0
//! Backpressure integration test: with a declared `type State`, a candidate
//! that does not define `State` the way the harness needs is rejected before
//! compiling, the reason is fed back, and the agent recovers.
//!
//! One test per binary: [`symbiont::Runtime`] is a process-wide singleton.

mod common;

use common::{
    ScriptedAgent,
    Turn,
};
use symbiont::{
    LadderEvent,
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
async fn invalid_state_is_rejected_and_recovered_from() {
    symbiont::evolvable! {
        type State: Default + Send + 'static;

        fn bp_state_step(state: &mut State, counter: &mut usize) {
            let _ = state;
            *counter += 1;
        }
    };
    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init runtime");
    // Without a `struct State` in the block, the initial one is a unit struct.
    assert_eq!(
        rt.state_source().map(|s| s.to_string()).as_deref(),
        Some("#[derive(Default)]\nstruct State;\n")
    );

    let agent = ScriptedAgent::new([
        // Attempt 1: no `State` at all.
        Turn::reply(
            "```rust\npub fn bp_state_step(state: &mut State, counter: &mut usize) { *counter += 2; }\n```",
        ),
        // Attempt 2: a `State` without `Default`.
        Turn::reply(
            "```rust\nstruct State { calls: usize }\npub fn bp_state_step(state: &mut State, counter: &mut usize) { state.calls += 1; *counter += state.calls; }\n```",
        ),
        // Attempt 3: valid.
        Turn::reply(
            "```rust\n#[derive(Default)]\nstruct State { calls: usize }\npub fn bp_state_step(state: &mut State, counter: &mut usize) { state.calls += 1; *counter += state.calls; }\n```",
        ),
    ]);

    let info = rt
        .evolve(&agent, BASE_PROMPT)
        .await
        .expect("evolution should succeed after two self-healing retries");
    assert_eq!(agent.calls(), 3, "exactly two retries expected");

    let missing = agent.prompt(1);
    assert!(
        missing.contains("does not define `State`"),
        "the first nudge names the missing definition, got: {missing}"
    );
    let no_default = agent.prompt(2);
    assert!(
        no_default.contains("neither derives nor implements `Default`"),
        "the second nudge names the missing `Default`, got: {no_default}"
    );

    for rejected in &info.trace().attempts()[..2] {
        assert!(
            matches!(rejected.ladder(), LadderEvent::SelfHeal { kind, .. } if kind == "state"),
            "got: {:?}",
            rejected.ladder()
        );
    }

    let mut state = new_state(info.revision()).expect("state of the evolved revision");
    let mut counter = 0;
    bp_state_step(&mut state, &mut counter);
    bp_state_step(&mut state, &mut counter);
    assert_eq!(counter, 3, "the evolved state counts the calls");
}

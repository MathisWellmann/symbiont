// SPDX-License-Identifier: MPL-2.0
//! Backpressure integration test: a response without parseable Rust code is
//! fed back to the agent, which then recovers.
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
async fn parse_failure_is_fed_back_and_recovered_from() {
    symbiont::evolvable! {
        fn bp_parse_step(counter: &mut usize) {
            *counter += 1;
        }
    };
    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init runtime");

    let agent = ScriptedAgent::new([
        // Attempt 1: a fenced code block that is NOT valid Rust — `as u8 << 16`
        // fails to parse because `<<` after a cast type is interpreted as the
        // start of generic arguments (`u8<...`), not a shift.
        Turn::reply(
            "```rust\npub fn bp_parse_step(counter: &mut usize) { *counter = (*counter as u8 << 1) as usize; }\n```",
        ),
        // Attempt 2: valid code -> success.
        Turn::reply("```rust\npub fn bp_parse_step(counter: &mut usize) { *counter += 41; }\n```"),
    ]);

    rt.evolve(&agent, BASE_PROMPT)
        .await
        .expect("evolution should succeed after one self-healing retry");

    assert_eq!(agent.calls(), 2, "exactly one retry expected");

    // Attempt 1 receives the unmodified base prompt and an empty history.
    assert_eq!(agent.prompt(0), BASE_PROMPT);
    assert_eq!(agent.history_len(0), 0);

    // Attempt 2 receives base prompt + the parse-failure nudge appended,
    // including the offending code and syn's located diagnostic.
    let retry_prompt = agent.prompt(1);
    assert!(
        retry_prompt.starts_with(BASE_PROMPT),
        "retry prompt must start with the base prompt, got: {retry_prompt}"
    );
    assert!(
        retry_prompt.contains("is not valid Rust"),
        "retry prompt must contain the parse-failure nudge, got: {retry_prompt}"
    );
    assert!(
        retry_prompt.contains("as u8 << 1"),
        "retry prompt must echo the offending code, got: {retry_prompt}"
    );
    assert!(
        retry_prompt.contains("line "),
        "retry prompt must carry the parse error location, got: {retry_prompt}"
    );

    // The failed attempt is still part of the chat history (user + assistant).
    assert_eq!(agent.history_len(1), 2);

    // The hot-swapped implementation is live.
    let mut counter = 0;
    bp_parse_step(&mut counter);
    assert_eq!(counter, 41, "evolved implementation should be hot-swapped");
}

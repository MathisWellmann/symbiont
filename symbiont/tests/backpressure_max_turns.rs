// SPDX-License-Identifier: MPL-2.0
//! Backpressure integration test: a rig `MaxTurnsError` (tool-call turn
//! budget exhausted) gets a concise turn-budget correction, and the agent
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
use rig_core::{
    completion::PromptError,
    message::Message,
};
use symbiont::{
    Profile,
    Runtime,
};

const BASE_PROMPT: &str = "Implement the function. Code only.";

#[tokio::test]
#[tracing_test::traced_test]
async fn max_turns_error_is_nudged_and_recovered_from() {
    symbiont::evolvable! {
        fn bp_turns_step(counter: &mut usize) {
            *counter += 1;
        }
    };
    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init runtime");

    let agent = ScriptedAgent::new([
        // Attempt 1: rig aborts the run because the model chained more tool
        // calls than `default_max_turns` allows.
        Turn::Fail(PromptError::MaxTurnsError {
            max_turns: 3,
            chat_history: Box::new(Vec::new()),
            prompt: Box::new(Message::user(BASE_PROMPT)),
        }),
        // Attempt 2: final code without further tool calls -> success.
        Turn::reply("```rust\npub fn bp_turns_step(counter: &mut usize) { *counter += 11; }\n```"),
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
        retry_prompt.contains("exhausted the tool-call turn budget"),
        "retry prompt must contain the turn-budget nudge, got: {retry_prompt}"
    );

    // A failed run produces no messages, so the history stays empty.
    assert_eq!(agent.history_len(0), 0);
    assert_eq!(agent.history_len(1), 0);

    // The hot-swapped implementation is live.
    let mut counter = 0;
    bp_turns_step(&mut counter);
    assert_eq!(counter, 11, "evolved implementation should be hot-swapped");
}

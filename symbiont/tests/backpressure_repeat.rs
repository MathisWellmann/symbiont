// SPDX-License-Identifier: MPL-2.0
//! Backpressure integration test: an agent that echoes the exact same
//! rejected code on consecutive attempts triggers a history reset — the
//! next request starts fresh from the base prompt with an explicit
//! do-not-repeat instruction that does not quote the broken code again.
//!
//! This pins down the recovery path for weak models that copy their own
//! broken answer out of the chat history instead of applying the
//! correction (observed with small local models in CI).
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

/// Invalid Rust (C-style for loop) that a weak model keeps repeating.
const BROKEN: &str = "```rust\npub fn bp_repeat_step(counter: &mut usize) { for (i = 0; i < 1; i++) { *counter += 1; } }\n```";

#[tokio::test]
#[cfg_attr(
    miri,
    ignore = "compiles and dlopens dylibs, which Miri does not support"
)]
#[tracing_test::traced_test]
async fn repeated_rejected_code_resets_history_and_restarts_from_base_prompt() {
    symbiont::evolvable! {
        fn bp_repeat_step(counter: &mut usize) {
            *counter += 1;
        }
    };
    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init runtime");

    let agent = ScriptedAgent::new([
        // Attempt 1: invalid code -> parse error fed back.
        Turn::reply(BROKEN),
        // Attempt 2: the agent echoes the exact same invalid code.
        Turn::reply(BROKEN),
        // Attempt 3: fresh start; the agent recovers.
        Turn::reply("```rust\npub fn bp_repeat_step(counter: &mut usize) { *counter += 41; }\n```"),
    ]);

    rt.evolve(&agent, BASE_PROMPT)
        .await
        .expect("evolution should succeed after the repeat reset");

    assert_eq!(agent.calls(), 3, "exactly two retries expected");

    // Attempt 2 receives the normal parse correction with history intact.
    let retry_prompt = agent.prompt(1);
    assert!(
        retry_prompt.contains("is not valid Rust"),
        "first retry must carry the parse-failure nudge, got: {retry_prompt}"
    );
    assert_eq!(agent.history_len(1), 2);

    // Attempt 3 detects the verbatim repeat: history is discarded and the
    // prompt restarts from the base prompt with a do-not-repeat instruction
    // that does NOT quote the rejected code again.
    let reset_prompt = agent.prompt(2);
    assert_eq!(
        agent.history_len(2),
        0,
        "history must be discarded after a verbatim repeat"
    );
    assert!(
        reset_prompt.starts_with(BASE_PROMPT),
        "reset prompt must restart from the base prompt, got: {reset_prompt}"
    );
    assert!(
        reset_prompt.contains("do NOT repeat"),
        "reset prompt must carry the do-not-repeat instruction, got: {reset_prompt}"
    );
    assert!(
        !reset_prompt.contains("i++"),
        "reset prompt must not quote the rejected code, got: {reset_prompt}"
    );

    // The hot-swapped implementation is live.
    let mut counter = 0;
    bp_repeat_step(&mut counter);
    assert_eq!(counter, 41, "evolved implementation should be hot-swapped");
}

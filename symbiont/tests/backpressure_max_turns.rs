// SPDX-License-Identifier: MPL-2.0
//! Backpressure integration test: a rig `MaxTurnsError` (tool-call turn
//! budget exhausted) gets a concise turn-budget correction, the agent's tools
//! are withdrawn for the rest of the lane, and the agent recovers.
//!
//! Nudging alone did not work in production: a model that spent fifty turns
//! on documentation lookups resumed them after the nudge, ten times over.
//! Every request after the first exhaustion therefore goes out through
//! `run_without_tools`, and a tool call that arrives anyway
//! (`UnknownToolCall`) is one more nudge, not a terminal failure.
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
use rig_agent::completion::PromptError;
use rig_core::message::Message;
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
        // Attempt 2, without tools: the model still emits a tool call, which
        // rig refuses to dispatch under `tool_choice: none`.
        Turn::Fail(PromptError::UnknownToolCall {
            tool_name: "api_doc".to_string(),
            available_tools: vec!["api_doc".to_string()],
            allowed_tools: Vec::new(),
            chat_history: Box::new(Vec::new()),
        }),
        // Attempt 3: final code without further tool calls -> success.
        Turn::reply("```rust\npub fn bp_turns_step(counter: &mut usize) { *counter += 11; }\n```"),
    ]);

    rt.evolve(&agent, BASE_PROMPT)
        .await
        .expect("evolution should succeed after two self-healing retries");

    assert_eq!(agent.calls(), 3, "exactly two retries expected");

    let retry_prompt = agent.prompt(1);
    assert!(
        !retry_prompt.contains(BASE_PROMPT),
        "retry prompt must contain only the correction, got: {retry_prompt}"
    );
    assert!(
        retry_prompt.contains("spent all 3 tool-call turns") && retry_prompt.contains("withdrawn"),
        "retry prompt must contain the turn-budget nudge, got: {retry_prompt}"
    );
    let stray_tool_prompt = agent.prompt(2);
    assert!(
        stray_tool_prompt.contains("`api_doc`")
            && stray_tool_prompt.contains("Do not call any tool"),
        "a tool call after withdrawal is nudged, got: {stray_tool_prompt}"
    );

    // The first request had tools; every request after the exhaustion does
    // not, including the one after the stray tool call.
    assert!(agent.tools_allowed(0));
    assert!(!agent.tools_allowed(1));
    assert!(!agent.tools_allowed(2));

    // A run that aborts without producing any messages (rig reported an
    // empty transcript in the error) recovers nothing, so the history
    // stays empty.
    assert_eq!(agent.history_len(0), 0);
    assert_eq!(agent.history_len(1), 0);
    assert_eq!(agent.history_len(2), 0);

    // The hot-swapped implementation is live.
    let mut counter = 0;
    bp_turns_step(&mut counter);
    assert_eq!(counter, 11, "evolved implementation should be hot-swapped");
}

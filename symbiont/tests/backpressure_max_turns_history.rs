// SPDX-License-Identifier: MPL-2.0
//! Backpressure integration test: a run that aborts inside the tool loop
//! (rig `MaxTurnsError`) still produced messages before the abort, and the
//! runtime recovers them into the retry history.
//!
//! Without the recovery, the retry request is byte-identical to the one that
//! just exhausted the budget — the model cannot remember its tool exchanges,
//! so it re-issues them and exhausts the budget again. With the recovery, the
//! retry extends the aborted conversation and the turn-budget nudge ("respond
//! with the final Rust code block now") is actionable.
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

/// The transcript rig reports in the error of an aborted tool-loop run:
/// the input history it was given, plus the messages the run itself
/// produced. The runtime treats `Message`s opaquely, so plain text turns
/// stand in for tool-call/tool-result exchanges here.
fn aborted_transcript(input: &[Message], run: &[Message]) -> Vec<Message> {
    input.iter().chain(run).cloned().collect()
}

#[tokio::test]
#[cfg_attr(
    miri,
    ignore = "compiles and dlopens dylibs, which Miri does not support"
)]
#[tracing_test::traced_test]
async fn aborted_tool_run_messages_reach_the_retry() {
    symbiont::evolvable! {
        fn bp_turns_hist_step(counter: &mut usize) {
            *counter += 1;
        }
    };
    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init runtime");

    // --- Phase 1: abort on the first attempt, empty input history. ---
    let input: Vec<Message> = Vec::new();
    let run = vec![
        Message::user(BASE_PROMPT),
        Message::assistant("calling api_doc for the signature"),
        Message::user("tool result: fn bp_turns_hist_step(counter: &mut usize)"),
    ];
    let transcript = aborted_transcript(&input, &run);

    let agent = ScriptedAgent::new([
        // Attempt 1: rig aborts the run after its tool exchanges and reports
        // the transcript it reached.
        Turn::Fail(PromptError::MaxTurnsError {
            max_turns: 3,
            chat_history: Box::new(transcript.clone()),
            prompt: Box::new(Message::user(BASE_PROMPT)),
        }),
        // Attempt 2: the retry sees the recovered tool results, so the model
        // can answer with the final code without further tool calls.
        Turn::reply(
            "```rust\npub fn bp_turns_hist_step(counter: &mut usize) { *counter += 21; }\n```",
        ),
    ]);

    rt.evolve(&agent, BASE_PROMPT)
        .await
        .expect("evolution should succeed after one self-healing retry");

    assert_eq!(agent.calls(), 2, "exactly one retry expected");

    // The first attempt started with an empty history.
    assert!(agent.history(0).is_empty());

    // The retry request carries exactly the aborted run's transcript: no
    // lost tool exchanges, and no duplicated input.
    assert_eq!(agent.history(1), transcript);
    let base_prompt_copies = agent
        .history(1)
        .iter()
        .filter(|m| format!("{m:?}").contains(BASE_PROMPT))
        .count();
    assert_eq!(
        base_prompt_copies, 1,
        "the base prompt must appear exactly once in the retry history"
    );
    assert!(
        agent
            .prompt(1)
            .contains("exhausted the tool-call turn budget"),
        "retry prompt must contain the turn-budget nudge, got: {}",
        agent.prompt(1)
    );

    let mut counter = 0;
    bp_turns_hist_step(&mut counter);
    assert_eq!(counter, 21, "evolved implementation should be hot-swapped");

    // --- Phase 2: abort mid-lane, on an attempt whose input history is
    // non-empty. The recovery must append only the run's own messages, not
    // replay the input it was given. ---
    let prior_exchange = vec![
        Message::user(BASE_PROMPT),
        Message::assistant("oops, no code block here"),
    ];
    let run_mid = vec![
        Message::user(
            "nudge: Your response did not contain a rust code block. \
            Please try again and make sure its wrapped like this: ```CODE```",
        ),
        Message::assistant("calling api_doc again"),
        Message::user("tool result: fn bp_turns_hist_step(counter: &mut usize)"),
    ];
    // Attempt 2's visible history was `prior_exchange`, and rig reports
    // input ++ run messages in the error.
    let transcript_mid = aborted_transcript(&prior_exchange, &run_mid);

    let agent_mid = ScriptedAgent::new([
        // Attempt 1: a response without a code block. The run itself
        // succeeds, so the exchange lands in the history and the ladder
        // nudges with NoRustCode.
        Turn::reply("oops, no code block here"),
        // Attempt 2: rig aborts the tool loop; the error transcript is the
        // two-message input plus the run's three messages.
        Turn::Fail(PromptError::MaxTurnsError {
            max_turns: 3,
            chat_history: Box::new(transcript_mid.clone()),
            prompt: Box::new(Message::user(
                "nudge: You exhausted the tool-call turn budget before producing code.",
            )),
        }),
        // Attempt 3: final code, recovered.
        Turn::reply(
            "```rust\npub fn bp_turns_hist_step(counter: &mut usize) { *counter += 7; }\n```",
        ),
    ]);

    rt.evolve(&agent_mid, BASE_PROMPT)
        .await
        .expect("evolution should succeed after two self-healing retries");

    assert_eq!(agent_mid.calls(), 3);
    assert_eq!(agent_mid.history(0).len(), 0);
    // After the failed (but successful-run) attempt 1, the history carries
    // the prompt and the reply.
    assert_eq!(agent_mid.history(1), prior_exchange);
    // The abort recovery appended only the run's three messages — the
    // two-message input it was given is not replayed.
    assert_eq!(agent_mid.history(2), transcript_mid);
    let base_prompt_copies = agent_mid
        .history(2)
        .iter()
        .filter(|m| format!("{m:?}").contains(BASE_PROMPT))
        .count();
    assert_eq!(
        base_prompt_copies, 1,
        "the base prompt must still appear exactly once"
    );

    let mut counter = 0;
    bp_turns_hist_step(&mut counter);
    assert_eq!(counter, 7, "final evolution should be hot-swapped");
}

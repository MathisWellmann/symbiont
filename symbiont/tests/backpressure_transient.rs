// SPDX-License-Identifier: MPL-2.0
//! Backpressure integration test: transient provider failures (connection
//! resets, rate limits) are retried with an *unmodified* prompt and do not
//! count against the self-healing budget.
//!
//! Second scenario: a run that fails on a later tool turn keeps what it
//! produced. The answered turns stay in the history, their tokens are in the
//! trace, and the retry asks the model to continue rather than to start over.
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
use rig_core::{
    completion::{
        CompletionError,
        Usage,
    },
    message::Message,
};
use symbiont::{
    CompletionCall,
    PartialRun,
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
async fn transient_http_error_is_retried_with_unmodified_prompt() {
    symbiont::evolvable! {
        fn bp_transient_step(counter: &mut usize) {
            *counter += 1;
        }
    };
    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init runtime");

    let agent = ScriptedAgent::new([
        // Attempt 1: connection-level failure -> transient, retried with backoff.
        Turn::Fail(connection_reset()),
        // Attempt 2: valid code -> success.
        Turn::reply(
            "```rust\npub fn bp_transient_step(counter: &mut usize) { *counter += 23; }\n```",
        ),
    ]);

    rt.evolve(&agent, BASE_PROMPT)
        .await
        .expect("evolution should succeed after one transient retry");

    assert_eq!(agent.calls(), 2, "exactly one transient retry expected");
    // A run that never got an answer is a run that did not happen.
    assert!(agent.history(1).is_empty());

    // Transient failures are not the LLM's fault: the prompt must be retried
    // verbatim, without any self-healing nudge appended.
    assert_eq!(agent.prompt(0), BASE_PROMPT);
    assert_eq!(
        agent.prompt(1),
        BASE_PROMPT,
        "transient retries must not modify the prompt"
    );

    // The hot-swapped implementation is live.
    let mut counter = 0;
    bp_transient_step(&mut counter);
    assert_eq!(counter, 23, "evolved implementation should be hot-swapped");

    partial_run_is_kept(rt).await;
    let mut counter = 0;
    bp_transient_step(&mut counter);
    assert_eq!(counter, 29, "the continued run's implementation is live");
}

fn connection_reset() -> PromptError {
    PromptError::CompletionError(CompletionError::HttpError(
        rig_core::http_client::Error::Instance(Box::new(std::io::Error::other(
            "simulated connection reset",
        ))),
    ))
}

/// The failed run had one answered turn (a tool call and its result) before
/// the endpoint went away.
async fn partial_run_is_kept(rt: &Runtime) {
    let mut usage = Usage::new();
    usage.input_tokens = 1200;
    usage.output_tokens = 40;
    let produced = vec![
        Message::user(BASE_PROMPT),
        Message::assistant("let me look that up"),
        Message::user("api_doc: pub struct Thing;"),
    ];
    let agent = ScriptedAgent::new([
        Turn::FailAfter(
            connection_reset(),
            PartialRun {
                new_messages: produced.clone(),
                usage,
                completion_calls: vec![CompletionCall::new(0, usage)],
            },
        ),
        Turn::reply(
            "```rust\npub fn bp_transient_step(counter: &mut usize) { *counter += 29; }\n```",
        ),
    ]);

    let info = rt
        .evolve(&agent, BASE_PROMPT)
        .await
        .expect("evolution should succeed after one transient retry");
    assert_eq!(agent.calls(), 2);

    // The retry sees the answered turn and is asked to carry on from it.
    assert_eq!(agent.history_len(0), 0);
    assert_eq!(
        agent.history(1),
        produced,
        "the failed run's messages are kept"
    );
    let retry = agent.prompt(1);
    assert!(
        retry.starts_with("nudge:") && retry.contains("Continue from there"),
        "a run with progress is continued, not restarted, got: {retry}"
    );

    // The trace has the failed attempt's tokens and messages.
    let trace = info.into_trace();
    let failed = trace.attempts()[0]
        .run()
        .as_ref()
        .expect("a run that answered once is recorded");
    assert_eq!(*failed.usage(), usage);
    assert_eq!(failed.completion_calls().len(), 1);
    assert_eq!(*failed.produced(), 0..3);
    assert!(failed.response().is_empty());
    assert_eq!(trace.usage().input_tokens, 1200);
}

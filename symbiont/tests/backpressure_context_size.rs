// SPDX-License-Identifier: MPL-2.0
//! Backpressure integration test: a request that exceeds the model's context
//! window is recovered by discarding the accumulated retry history and
//! restarting from the base prompt, instead of failing the evolution.
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
use rig_core::completion::{
    CompletionError,
    PromptError,
};
use symbiont::{
    Profile,
    Runtime,
};

const BASE_PROMPT: &str = "Implement the function. Code only.";

/// The error body llama.cpp returns when the request overflows `n_ctx`.
const CONTEXT_SIZE_BODY: &str = r#"{"error":{"code":400,"message":"request (258963 tokens) exceeds the available context size (256000 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":258963,"n_ctx":256000}}"#;

fn context_size_error() -> PromptError {
    PromptError::CompletionError(CompletionError::HttpError(
        rig_core::http_client::Error::InvalidStatusCodeWithMessage(
            http::StatusCode::BAD_REQUEST,
            CONTEXT_SIZE_BODY.to_string(),
        ),
    ))
}

#[tokio::test]
#[cfg_attr(
    miri,
    ignore = "compiles and dlopens dylibs, which Miri does not support"
)]
#[tracing_test::traced_test]
async fn context_size_overflow_restarts_from_the_base_prompt() {
    symbiont::evolvable! {
        fn bp_ctx_step(counter: &mut usize) {
            *counter += 1;
        }
    };
    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init runtime");

    let agent = ScriptedAgent::new([
        // Attempt 1: invalid Rust -> parse error, correction turn queued and
        // the exchange lands in the retry history.
        Turn::reply("```rust\nthis is not rust\n```"),
        // Attempt 2: the grown request overflows the context window.
        Turn::Fail(context_size_error()),
        // Attempt 3: restarted from the base prompt -> success.
        Turn::reply("```rust\npub fn bp_ctx_step(counter: &mut usize) { *counter += 31; }\n```"),
    ]);

    rt.evolve(&agent, BASE_PROMPT)
        .await
        .expect("evolution should succeed after the context-size restart");

    assert_eq!(agent.calls(), 3);

    // Attempt 2 was a self-healing retry: correction prompt, history carries
    // the failed exchange.
    assert!(
        agent.prompt(1).contains("not valid Rust"),
        "got: {}",
        agent.prompt(1)
    );
    assert_eq!(agent.history_len(1), 2);

    // Attempt 3 restarted fresh: base prompt, no history.
    assert_eq!(
        agent.prompt(2),
        BASE_PROMPT,
        "the retry after a context-size overflow must restart from the base prompt"
    );
    assert_eq!(
        agent.history_len(2),
        0,
        "the accumulated history must be discarded on a context-size overflow"
    );

    // The hot-swapped implementation is live.
    let mut counter = 0;
    bp_ctx_step(&mut counter);
    assert_eq!(counter, 31, "evolved implementation should be hot-swapped");

    // -- Oversized base prompt ---------------------------------------------------

    // When even a fresh request overflows (no history to discard), the error
    // is surfaced to the caller: only the host can slim down the base prompt.
    let agent = ScriptedAgent::new([Turn::Fail(context_size_error())]);
    let err = rt
        .evolve(&agent, BASE_PROMPT)
        .await
        .expect_err("a base prompt exceeding the context window cannot be recovered");
    assert!(
        err.to_string().contains("exceed_context_size"),
        "got: {err}"
    );
    assert_eq!(
        agent.calls(),
        1,
        "no retry can help an oversized base prompt"
    );
}

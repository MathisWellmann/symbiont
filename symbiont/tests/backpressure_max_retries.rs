// SPDX-License-Identifier: MPL-2.0
//! Backpressure integration test: a persistently misbehaving agent exhausts
//! the self-healing budget and `evolve` returns `MaxRetriesExceeded` instead
//! of looping forever. Also pins down that each retry sends only the latest
//! correction and corrections never accumulate.
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
    Error,
    Profile,
    Runtime,
};

const BASE_PROMPT: &str = "Implement the function. Code only.";

#[tokio::test]
#[tracing_test::traced_test]
async fn retry_budget_is_bounded_and_nudges_do_not_accumulate() {
    symbiont::evolvable! {
        fn bp_retries_step(counter: &mut usize) {
            *counter += 1;
        }
    };
    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init runtime");

    // The agent never produces code, on any attempt.
    let agent = ScriptedAgent::new(
        (0..Runtime::MAX_EVOLVE_ATTEMPTS).map(|_| Turn::reply("I am unable to help with that.")),
    );

    let err = rt
        .evolve(&agent, BASE_PROMPT)
        .await
        .expect_err("evolution must give up after the retry budget is spent");

    match err {
        Error::MaxRetriesExceeded {
            attempts,
            last_error,
        } => {
            assert_eq!(attempts, Runtime::MAX_EVOLVE_ATTEMPTS);
            assert!(
                matches!(*last_error, Error::NoRustCode),
                "last error should be the missing-code-block failure, got: {last_error}"
            );
        }
        other => panic!("expected MaxRetriesExceeded, got: {other}"),
    }

    assert_eq!(
        agent.calls(),
        Runtime::MAX_EVOLVE_ATTEMPTS,
        "agent must be called exactly once per attempt"
    );

    // Each retry contains only the latest correction — it never stacks.
    const NUDGE: &str = "did not contain a rust code block";
    let last = agent.prompt(Runtime::MAX_EVOLVE_ATTEMPTS - 1);
    assert_eq!(
        last,
        agent.prompt(1),
        "retry corrections must be identical across attempts"
    );
    assert!(
        !last.contains(BASE_PROMPT),
        "retry must not repeat the base prompt, got: {last}"
    );
    assert_eq!(
        last.matches(NUDGE).count(),
        1,
        "the nudge must appear exactly once, got: {last}"
    );

    // Every attempt whose agent run returned text — even unparseable text —
    // is recorded in the chat history (prompt + assistant reply per attempt).
    assert_eq!(
        agent.history_len(Runtime::MAX_EVOLVE_ATTEMPTS - 1),
        (Runtime::MAX_EVOLVE_ATTEMPTS - 1) * 2,
        "history must grow by two messages per failed attempt"
    );

    // The original implementation is untouched.
    let mut counter = 0;
    bp_retries_step(&mut counter);
    assert_eq!(counter, 1, "failed evolution must not change the function");
}

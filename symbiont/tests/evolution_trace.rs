// SPDX-License-Identifier: MPL-2.0
//! Integration test for the evolution trace.
//!
//! An evolution that needs one self-healing retry must carry a trace of the
//! whole lane. The trace holds both attempts, the ladder decision between
//! them, the transcript they produced, and the stage timings that show how far
//! each attempt got.
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
    BuildRecord,
    LadderEvent,
    Lane,
    Profile,
    Runtime,
    TraceOutcome,
};

const BASE_PROMPT: &str = "Implement the function. Code only.";

#[tokio::test]
#[cfg_attr(
    miri,
    ignore = "compiles and dlopens dylibs, which Miri does not support"
)]
#[tracing_test::traced_test]
async fn trace_records_the_whole_lane() {
    symbiont::evolvable! {
        fn trace_step(counter: &mut usize) {
            *counter += 1;
        }
    };
    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init runtime");

    let agent = ScriptedAgent::new([
        // Attempt 1: no code block at all -> `no_rust_code`, nudged.
        Turn::reply("I would be happy to help you with that!"),
        // Attempt 2: valid -> registered.
        Turn::reply("```rust\npub fn trace_step(counter: &mut usize) { *counter += 9; }\n```"),
    ]);

    let info = rt
        .evolve(&agent, BASE_PROMPT)
        .await
        .expect("evolution should succeed after one self-healing retry");
    let trace = info.trace();

    assert_eq!(
        trace.lane(),
        Lane::from(0),
        "single-prompt evolve is lane 0"
    );
    assert_eq!(trace.base_prompt(), BASE_PROMPT);
    assert_eq!(trace.attempts().len(), 2, "one rejection plus one success");

    // `seq` indexes the timeline. `attempt` is the self-healing counter. The
    // two agree here, because neither attempt was a transient retry.
    let seqs: Vec<usize> = trace.attempts().iter().map(|a| a.seq()).collect();
    let attempts: Vec<usize> = trace.attempts().iter().map(|a| a.attempt()).collect();
    assert_eq!(seqs, vec![0, 1]);
    assert_eq!(attempts, vec![1, 2]);

    // The runtime rejected the first attempt before any code existed. That
    // attempt got to the model, but never to the build stage.
    let first = &trace.attempts()[0];
    assert_eq!(
        first.prompt(),
        BASE_PROMPT,
        "attempt 1 sends the base prompt"
    );
    let first_run = first.run().as_ref().expect("attempt 1 reached the model");
    assert!(!first_run.response().is_empty());
    assert_eq!(
        first_run.completion_calls().len(),
        1,
        "one HTTP request per scripted turn"
    );
    assert!(first.stages().llm().is_some(), "the model was called");
    assert!(
        first.stages().build().is_none(),
        "a response without a code block never reaches the build stage"
    );
    match first.ladder() {
        LadderEvent::SelfHeal {
            kind,
            diagnostics,
            api_hints,
        } => {
            assert_eq!(kind, "no_rust_code");
            assert!(
                diagnostics.contains("rust code block"),
                "the nudge is the diagnostics fed back, got: {diagnostics}"
            );
            assert!(
                api_hints.is_empty(),
                "only compile failures about a host type attach documentation"
            );
        }
        other => panic!("expected a self-heal after a missing code block, got: {other:?}"),
    }

    // The second attempt built and registered.
    let second = &trace.attempts()[1];
    assert_ne!(
        second.prompt(),
        BASE_PROMPT,
        "attempt 2 sends the corrective nudge, not the base prompt"
    );
    assert!(second.run().is_some());
    assert!(second.stages().parse_validate().is_some());
    assert!(
        matches!(second.stages().build(), Some(BuildRecord::Built { .. })),
        "a fresh candidate is compiled, got: {:?}",
        second.stages().build()
    );
    assert!(matches!(second.ladder(), LadderEvent::Registered { .. }));

    // The outcome agrees with the returned revision.
    match trace.outcome() {
        TraceOutcome::Registered { revision } => assert_eq!(*revision, info.revision()),
        other => panic!("expected a registered outcome, got: {other:?}"),
    }

    assert_transcript_is_tiled(trace);

    // The usage of the lane is the sum of its attempts, and it includes the
    // rejected one. A rejected attempt is still a run that costs tokens.
    assert_eq!(
        trace.usage().total_tokens,
        info.usage().total_tokens,
        "the trace agrees with the reported usage"
    );
    assert_eq!(trace.completion_calls(), 2);
}

/// The trace owns the transcript one time. The `produced` range of each
/// attempt indexes into it, and the ranges of two attempts in sequence touch.
/// This layout is the reason one copy of the transcript is sufficient.
fn assert_transcript_is_tiled(trace: &symbiont::EvolutionTrace) {
    assert!(
        !trace.history().is_empty(),
        "the lane exchanged messages, so the transcript is not empty"
    );
    let mut expected_start = 0;
    for attempt in trace.attempts() {
        let run = attempt
            .run()
            .as_ref()
            .expect("both attempts reached the model");
        assert_eq!(
            run.produced().start,
            expected_start,
            "attempt {} must continue where the previous one ended",
            attempt.seq(),
        );
        assert!(
            trace.history().get(run.produced().clone()).is_some(),
            "attempt {} range {:?} escapes a transcript of {}",
            attempt.seq(),
            run.produced(),
            trace.history().len(),
        );
        expected_start = run.produced().end;
    }
}

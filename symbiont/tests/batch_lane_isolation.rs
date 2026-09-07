// SPDX-License-Identifier: MPL-2.0
//! Batch integration test: a lane that exhausts its retry budget fails on its
//! own without disturbing its siblings, and every recorded failure is
//! attributed to the lane that produced it.
//!
//! One test per binary: [`symbiont::Runtime`] is a process-wide singleton.
#![expect(
    unused_crate_dependencies,
    reason = "Integration tests don't use them all"
)]

mod common;

use common::{
    ANY_PROMPT,
    RoutedAgent,
};
use symbiont::{
    Error,
    LadderEvent,
    Lane,
    Profile,
    Runtime,
};

/// Source for a lane that returns `value`.
fn implementation(value: usize) -> String {
    format!("```rust\npub fn batch_iso_step(counter: &mut usize) {{ *counter += {value}; }}\n```")
}

#[tokio::test]
#[cfg_attr(
    miri,
    ignore = "compiles and dlopens dylibs, which Miri does not support"
)]
#[tracing_test::traced_test]
async fn a_failing_lane_does_not_disturb_its_siblings() {
    symbiont::evolvable! {
        fn batch_iso_step(counter: &mut usize) {
            *counter += 1;
        }
    };
    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init runtime");

    let prompts = [
        "Implement the function. Code only. Hint: healthy alpha",
        "Implement the function. Code only. Hint: doomed beta",
        "Implement the function. Code only. Hint: healthy gamma",
    ];

    // Lane 1 answers with prose forever: it burns its whole budget and fails.
    // Its retries arrive as bare correction prompts without the "doomed beta"
    // hint, so the prose route has to be the catch-all. The healthy lanes
    // succeed on their first attempt and never reach it.
    let doomed = Vec::from_iter(std::iter::repeat_n(
        "I am afraid I cannot help with that.".to_string(),
        Runtime::MAX_EVOLVE_ATTEMPTS,
    ));
    let agent = RoutedAgent::new([
        ("healthy alpha", vec![implementation(11)]),
        ("healthy gamma", vec![implementation(33)]),
        (ANY_PROMPT, doomed),
    ]);

    let results = rt.evolve_batch(&agent, &prompts).await;
    assert_eq!(results.len(), 3, "one result per prompt");

    // Siblings are unaffected and still positionally aligned.
    let alpha = results[0]
        .as_ref()
        .expect("lane 0 was given compiling code and must succeed");
    let gamma = results[2]
        .as_ref()
        .expect("lane 2 was given compiling code and must succeed");
    assert_ne!(
        alpha.revision(),
        gamma.revision(),
        "successful lanes get distinct revisions"
    );

    // The doomed lane fails in place, with the budget-exhaustion error.
    let doomed = results[1]
        .as_ref()
        .expect_err("lane 1 never produced Rust and must fail");
    match doomed.error() {
        Error::MaxRetriesExceeded { attempts, .. } => assert_eq!(
            *attempts,
            Runtime::MAX_EVOLVE_ATTEMPTS,
            "the doomed lane should spend exactly its own budget"
        ),
        other => panic!("expected MaxRetriesExceeded, got: {other}"),
    }

    // Only the two healthy lanes registered anything.
    assert_eq!(
        rt.revision_count(),
        3,
        "the initial revision plus the two lanes that produced valid code"
    );

    // Every failure belongs to the lane that produced it, and the retry budget
    // is per lane rather than shared.
    let trace = doomed.trace();
    assert_eq!(trace.lane(), Lane::from(1));
    assert_eq!(
        trace.attempts().len(),
        Runtime::MAX_EVOLVE_ATTEMPTS,
        "one record per failed attempt of the doomed lane"
    );
    let (last, healed) = trace
        .attempts()
        .split_last()
        .expect("the lane made attempts");
    assert!(
        healed.iter().all(|attempt| matches!(
            attempt.ladder(),
            LadderEvent::SelfHeal { kind, .. } if kind == "no_rust_code"
        )),
        "the doomed lane only ever answered with prose, got: {:?}",
        Vec::from_iter(healed.iter().map(|attempt| attempt.ladder()))
    );
    assert!(
        matches!(last.ladder(), LadderEvent::Terminal { .. }),
        "the attempt that exhausts the budget ends the lane, got: {:?}",
        last.ladder()
    );
    assert_eq!(
        Vec::from_iter(trace.attempts().iter().map(|attempt| attempt.attempt())),
        Vec::from_iter(1..=Runtime::MAX_EVOLVE_ATTEMPTS),
        "attempt numbering is per lane and starts at 1"
    );
    for healthy in [&results[0], &results[2]] {
        let trace = healthy.as_ref().expect("checked above").trace();
        assert_eq!(
            trace.attempts().len(),
            1,
            "healthy lanes must not record failures"
        );
    }
}

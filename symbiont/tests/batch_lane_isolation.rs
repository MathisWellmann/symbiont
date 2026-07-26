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
    assert_ne!(alpha, gamma, "successful lanes get distinct revisions");

    // The doomed lane fails in place, with the budget-exhaustion error.
    match results[1]
        .as_ref()
        .expect_err("lane 1 never produced Rust and must fail")
    {
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
    let failures = rt.take_evolve_failures();
    assert_eq!(
        failures.len(),
        Runtime::MAX_EVOLVE_ATTEMPTS,
        "one record per failed attempt of the doomed lane"
    );
    assert!(
        failures.iter().all(|f| f.lane() == 1),
        "healthy lanes must not contribute failure records, got lanes: {:?}",
        Vec::from_iter(failures.iter().map(symbiont::EvolveFailure::lane))
    );
    assert!(
        failures.iter().all(|f| f.kind() == "no_rust_code"),
        "the doomed lane only ever answered with prose"
    );
    let mut attempts = Vec::from_iter(failures.iter().map(symbiont::EvolveFailure::attempt));
    attempts.sort_unstable();
    assert_eq!(
        attempts,
        Vec::from_iter(1..=Runtime::MAX_EVOLVE_ATTEMPTS),
        "attempt numbering is per lane and starts at 1"
    );
}

// SPDX-License-Identifier: MPL-2.0
//! Batch integration test: every lane of `evolve_batch` registers its own
//! revision, results stay positionally aligned with the prompts, and the
//! active revision is deliberately left untouched.
//!
//! One test per binary: [`symbiont::Runtime`] is a process-wide singleton.
#![expect(
    unused_crate_dependencies,
    reason = "Integration tests don't use them all"
)]

mod common;

use common::RoutedAgent;
use symbiont::{
    Profile,
    Revision,
    Runtime,
};

/// Source for a lane that returns `value`.
fn implementation(value: usize) -> String {
    format!("```rust\npub fn batch_all_step(counter: &mut usize) {{ *counter += {value}; }}\n```")
}

#[tokio::test]
#[cfg_attr(
    miri,
    ignore = "compiles and dlopens dylibs, which Miri does not support"
)]
#[tracing_test::traced_test]
async fn every_lane_registers_a_revision_without_activating_it() {
    symbiont::evolvable! {
        fn batch_all_step(counter: &mut usize) {
            *counter += 1;
        }
    };
    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init runtime");

    // Three prompts differing only in a trailing hint — the shape a real batch
    // takes, so the shared prefix stays reusable.
    let prompts = [
        "Implement the function. Code only. Hint: strategy alpha",
        "Implement the function. Code only. Hint: strategy beta",
        "Implement the function. Code only. Hint: strategy gamma",
    ];
    let agent = RoutedAgent::new([
        ("strategy alpha", [implementation(10)]),
        ("strategy beta", [implementation(20)]),
        ("strategy gamma", [implementation(30)]),
    ]);

    let results = rt.evolve_batch(&agent, &prompts).await;

    assert_eq!(
        results.len(),
        prompts.len(),
        "one result per prompt, positionally aligned"
    );
    let revisions =
        Vec::from_iter(results.into_iter().map(|r| {
            r.expect("every lane was given a compiling implementation and should succeed")
        }));
    assert_eq!(agent.calls(), 3, "one agent run per lane, no retries");

    // Initial build plus one revision per lane, all distinct.
    assert_eq!(
        rt.revision_count(),
        4,
        "the initial revision plus one per lane"
    );
    let mut sorted = revisions.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), 3, "lanes must not share a revision id");
    assert!(
        !revisions.contains(&Revision::INITIAL),
        "no lane may reuse the initial revision"
    );

    // The whole point: registering is not activating.
    assert_eq!(
        rt.active_revision(),
        Revision::INITIAL,
        "evolve_batch must leave the active revision alone"
    );
    let mut counter = 0;
    batch_all_step(&mut counter);
    assert_eq!(
        counter, 1,
        "dispatch still runs the initial implementation, not any candidate"
    );

    // Each candidate is nevertheless callable through its own pinned handle,
    // and the source recorded for it is the one that lane generated.
    for (lane, revision) in revisions.iter().enumerate() {
        let expected = (lane + 1) * 10;
        let handle = batch_all_step_fn(*revision).expect("registered revisions are retained");
        let mut counter = 0;
        (handle.get())(&mut counter);
        assert_eq!(
            counter, expected,
            "lane {lane} should have registered the implementation adding {expected}"
        );
        assert!(
            rt.revision_code(*revision)
                .expect("registered revisions expose their source")
                .contains(&format!("+= {expected}")),
            "revision source must match what lane {lane} generated"
        );
    }

    // Committing to a winner is an explicit, separate step.
    rt.activate_revision(revisions[2]).expect("revision exists");
    let mut counter = 0;
    batch_all_step(&mut counter);
    assert_eq!(counter, 30, "the activated candidate is now live");

    assert!(
        rt.take_evolve_failures().is_empty(),
        "no lane failed, so nothing should be recorded"
    );
}

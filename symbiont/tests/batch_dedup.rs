// SPDX-License-Identifier: MPL-2.0
//! Batch integration test: lanes that converge on byte-identical source share
//! one revision instead of each spending a `cargo build` on the same artifact.
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
    format!("```rust\npub fn batch_dedup_step(counter: &mut usize) {{ *counter += {value}; }}\n```")
}

#[tokio::test]
#[cfg_attr(
    miri,
    ignore = "compiles and dlopens dylibs, which Miri does not support"
)]
#[tracing_test::traced_test]
async fn colliding_lanes_share_one_revision() {
    symbiont::evolvable! {
        fn batch_dedup_step(counter: &mut usize) {
            *counter += 1;
        }
    };
    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init runtime");

    // Four prompt variants, but only two distinct implementations between
    // them: lanes 0 and 2 collide, and so do lanes 1 and 3. This is what low
    // sampling temperature on near-identical prompts actually produces.
    let prompts = [
        "Implement the function. Code only. Hint: variant one",
        "Implement the function. Code only. Hint: variant two",
        "Implement the function. Code only. Hint: variant three",
        "Implement the function. Code only. Hint: variant four",
    ];
    let agent = RoutedAgent::new([
        ("variant one", [implementation(42)]),
        ("variant two", [implementation(7)]),
        ("variant three", [implementation(42)]),
        ("variant four", [implementation(7)]),
    ]);

    let results = rt.evolve_batch(&agent, &prompts).await;
    let revisions = Vec::from_iter(
        results
            .into_iter()
            .map(|r| r.expect("every lane produced compiling code")),
    );

    assert_eq!(agent.calls(), 4, "every lane still runs its own inference");

    // Colliding lanes report the same revision...
    assert_eq!(
        revisions[0], revisions[2],
        "lanes generating identical source must share a revision"
    );
    assert_eq!(
        revisions[1], revisions[3],
        "lanes generating identical source must share a revision"
    );
    assert_ne!(
        revisions[0], revisions[1],
        "lanes generating different source must not"
    );

    // ...and only the two distinct artifacts were actually built.
    assert_eq!(
        rt.revision_count(),
        3,
        "the initial revision plus one per *distinct* implementation, \
         not one per lane"
    );

    // Dedup must not disturb the rest of the contract: the shared revisions
    // are real, callable, and carry the source their lanes generated.
    assert_eq!(
        rt.active_revision(),
        Revision::INITIAL,
        "dedup must not activate anything either"
    );
    for (lane, (revision, expected)) in revisions.iter().zip([42, 7, 42, 7]).enumerate() {
        let handle = batch_dedup_step_fn(*revision).expect("registered revisions are retained");
        let mut counter = 0;
        (handle.get())(&mut counter);
        assert_eq!(
            counter, expected,
            "lane {lane} must resolve to the implementation it generated"
        );
    }
}

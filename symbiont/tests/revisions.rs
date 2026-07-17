#![expect(
    unused_crate_dependencies,
    missing_docs,
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
    Revision,
    Runtime,
};

/// The full retain-and-rollback lifecycle: every successful evolution is
/// registered as a revision, its dylib stays loaded, and any revision can be
/// re-activated later without parsing or compiling anything.
#[tokio::test]
#[tracing_test::traced_test]
#[expect(
    clippy::too_many_lines,
    reason = "The Runtime singleton allows one runtime per process, so the whole lifecycle lives in one sequential test"
)]
async fn revisions_are_retained_and_activatable() {
    symbiont::evolvable! {
        fn rev_step(counter: &mut usize) {
            *counter += 1;
        }
    };
    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init");
    assert_eq!(rt.active_revision(), Revision::INITIAL);
    assert_eq!(rt.revision_count(), 1);

    let mut counter = 0;
    rev_step(&mut counter);
    assert_eq!(counter, 1, "the initial implementation increments by 1");

    let agent = ScriptedAgent::new([
        Turn::reply("```rust\npub fn rev_step(counter: &mut usize) { *counter += 5; }\n```"),
        Turn::reply("```rust\npub fn rev_step(counter: &mut usize) { *counter += 7; }\n```"),
        Turn::reply("```rust\npub fn rev_step(counter: &mut usize) { *counter += 9; }\n```"),
        Turn::reply(
            "```rust\npub fn rev_step(counter: &mut usize) { let _ = counter; panic!(\"boom\") }\n```",
        ),
    ]);

    let rev_plus_5 = rt
        .evolve(&agent, "increment by 5")
        .await
        .expect("Can evolve");
    assert_eq!(rev_plus_5, Revision::new(1));
    assert_eq!(rt.active_revision(), rev_plus_5);
    counter = 0;
    rev_step(&mut counter);
    assert_eq!(counter, 5);

    let rev_plus_7 = rt
        .evolve(&agent, "increment by 7")
        .await
        .expect("Can evolve");
    assert_eq!(rev_plus_7, Revision::new(2));
    assert_eq!(rt.revision_count(), 3);
    counter = 0;
    rev_step(&mut counter);
    assert_eq!(counter, 7);

    // Roll back to the middle revision: a pointer swap into the still-loaded
    // dylib, no parsing or compilation involved.
    rt.activate_revision(rev_plus_5).expect("Can activate");
    assert_eq!(rt.active_revision(), rev_plus_5);
    counter = 0;
    rev_step(&mut counter);
    assert_eq!(counter, 5, "the re-activated revision dispatches again");
    assert_eq!(
        &rt.current_code(),
        "#[unsafe(no_mangle)]\npub fn rev_step(counter: &mut usize) {\n    *counter += 5;\n}\n",
        "current_code follows the activated revision"
    );

    // Sources of inactive revisions stay queryable.
    assert_eq!(
        rt.revision_code(rev_plus_7)
            .expect("revision 2 is registered"),
        "#[unsafe(no_mangle)]\npub fn rev_step(counter: &mut usize) {\n    *counter += 7;\n}\n"
    );

    // The initial build is a revision like any other.
    rt.activate_revision(Revision::INITIAL)
        .expect("Can activate");
    counter = 0;
    rev_step(&mut counter);
    assert_eq!(counter, 1);

    // Unknown revisions are rejected and leave the active revision untouched.
    let err = rt
        .activate_revision(Revision::new(99))
        .expect_err("unknown revision must be rejected");
    assert!(matches!(
        err,
        Error::UnknownRevision { requested, latest }
            if requested == Revision::new(99) && latest == rev_plus_7
    ));
    assert_eq!(rt.active_revision(), Revision::INITIAL);
    assert!(rt.revision_code(Revision::new(99)).is_none());

    // Evolving after a rollback appends on top of the registry; earlier
    // revisions are never overwritten.
    let rev_plus_9 = rt
        .evolve(&agent, "increment by 9")
        .await
        .expect("Can evolve");
    assert_eq!(rev_plus_9, Revision::new(3));
    assert_eq!(rt.active_revision(), rev_plus_9);
    assert_eq!(rt.revision_count(), 4);
    counter = 0;
    rev_step(&mut counter);
    assert_eq!(counter, 9);

    // -- Typed per-revision handles (RevisionFn) -------------------------------

    // Handles execute their own revision, independent of the active one.
    let plus_5 = rev_step_fn(rev_plus_5).expect("revision 1 is retained");
    let plus_7 = rev_step_fn(rev_plus_7).expect("revision 2 is retained");
    assert_eq!(plus_5.revision(), rev_plus_5);
    let f5 = plus_5.get();
    let f7 = plus_7.get();
    counter = 0;
    f5(&mut counter);
    assert_eq!(counter, 5, "the handle executes its own revision");
    counter = 0;
    f7(&mut counter);
    assert_eq!(
        counter, 7,
        "handles of different revisions run side by side"
    );
    counter = 0;
    rev_step(&mut counter);
    assert_eq!(
        counter, 9,
        "the active dispatch is unaffected by handle calls"
    );
    assert!(
        rev_step_fn(Revision::new(99)).is_none(),
        "unknown revisions yield no handle"
    );

    // Handles are Send + Clone: hammer two revisions from two threads while
    // the main thread swaps the active revision. Handle calls never read the
    // swappable dispatch pointers, so they are exempt from the feedback-loop
    // contract and may run concurrently with activate_revision.
    let t5 = std::thread::spawn({
        let handle = plus_5.clone();
        move || {
            let f = handle.get();
            let mut counter = 0;
            for _ in 0..10_000 {
                f(&mut counter);
            }
            counter
        }
    });
    let t7 = std::thread::spawn({
        let handle = plus_7.clone();
        move || {
            let f = handle.get();
            let mut counter = 0;
            for _ in 0..10_000 {
                f(&mut counter);
            }
            counter
        }
    });
    for _ in 0..100 {
        rt.activate_revision(rev_plus_5).expect("Can activate");
        rt.activate_revision(rev_plus_9).expect("Can activate");
    }
    assert_eq!(t5.join().expect("thread finished"), 50_000);
    assert_eq!(t7.join().expect("thread finished"), 70_000);

    // -- Per-revision panic routing ---------------------------------------------

    // Evolve a panicking implementation, then make a different revision
    // active: the handle call's panic must land in ITS revision's buffer.
    let rev_panic = rt.evolve(&agent, "panic please").await.expect("Can evolve");
    rt.activate_revision(rev_plus_5).expect("Can activate");
    let panicking = rev_step_fn(rev_panic).expect("revision is retained");
    counter = 0;
    (panicking.get())(&mut counter);
    assert_eq!(counter, 0, "the panic fired before any increment");
    let msg = panicking.take_panic().expect("the handle call panicked");
    assert!(msg.contains("boom"), "panic message is preserved: {msg}");
    assert!(
        rt.take_panic().is_none(),
        "the active revision's buffer is untouched by handle calls"
    );
    assert!(
        panicking.take_panic().is_none(),
        "the stored message is cleared on read"
    );
}

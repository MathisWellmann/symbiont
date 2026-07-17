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
}

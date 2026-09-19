// SPDX-License-Identifier: MPL-2.0
//! Unloading revisions: the registry releases the dylib, the file is
//! deleted once nothing pins it, the id turns into a tombstone, and the
//! bounded set of loaded revisions is observable.
//!
//! One test per binary: [`symbiont::Runtime`] is a process-wide singleton.
#![expect(
    unused_crate_dependencies,
    reason = "Integration tests don't use them all"
)]

mod common;

use std::path::PathBuf;

use common::{
    ScriptedAgent,
    Turn,
};
use metrics_util::{
    CompositeKey,
    debugging::{
        DebugValue,
        DebuggingRecorder,
    },
};
use symbiont::{
    Error,
    Profile,
    Revision,
    Runtime,
    Unloaded,
    observability,
};

type SnapshotEntry = (
    CompositeKey,
    Option<metrics::Unit>,
    Option<metrics::SharedString>,
    DebugValue,
);

/// The versioned dylib of `revision` inside the runtime's crate directory.
fn versioned_so(rt: &Runtime, revision: Revision) -> PathBuf {
    let needle = format!("_v{}.", revision.as_u64());
    std::fs::read_dir(rt.crate_dir())
        .expect("crate dir is readable")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    name.starts_with("libsymbiont_evolvable") && name.contains(&needle)
                })
        })
        .unwrap_or_else(|| {
            rt.crate_dir()
                .join(format!("libsymbiont_evolvable{needle}missing"))
        })
}

/// Whether `path` is mapped into this process (Linux only; `true` elsewhere
/// so the assertions that expect a mapping still hold and the ones that
/// expect none are skipped by the caller).
fn is_mapped(path: &std::path::Path) -> bool {
    if cfg!(target_os = "linux") {
        let maps = std::fs::read_to_string("/proc/self/maps").expect("/proc/self/maps is readable");
        maps.contains(path.to_str().expect("utf-8 path"))
    } else {
        true
    }
}

fn gauge_value(snapshot: &[SnapshotEntry], name: &str) -> Option<f64> {
    snapshot.iter().find_map(|(key, _, _, v)| match v {
        DebugValue::Gauge(g) if key.key().name() == name => Some(f64::from(*g)),
        _ => None,
    })
}

fn counter_value(snapshot: &[SnapshotEntry], name: &str, label: (&str, &str)) -> u64 {
    snapshot
        .iter()
        .filter(|(key, _, _, _)| {
            key.key().name() == name
                && key
                    .key()
                    .labels()
                    .any(|l| l.key() == label.0 && l.value() == label.1)
        })
        .filter_map(|(_, _, _, v)| match v {
            DebugValue::Counter(n) => Some(*n),
            _ => None,
        })
        .sum()
}

#[tokio::test]
#[cfg_attr(
    miri,
    ignore = "compiles and dlopens dylibs, which Miri does not support"
)]
#[tracing_test::traced_test]
#[expect(
    clippy::too_many_lines,
    reason = "The Runtime singleton allows one runtime per process, so the whole lifecycle lives in one sequential test"
)]
async fn revisions_can_be_unloaded() {
    symbiont::evolvable! {
        fn unload_step(counter: &mut usize) {
            *counter += 1;
        }
    };

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init");

    const PLUS_5: &str = "```rust\npub fn unload_step(counter: &mut usize) { *counter += 5; }\n```";
    let agent = ScriptedAgent::new([
        Turn::reply(PLUS_5),
        Turn::reply("```rust\npub fn unload_step(counter: &mut usize) { *counter += 7; }\n```"),
        Turn::reply("```rust\npub fn unload_step(counter: &mut usize) { *counter += 9; }\n```"),
        Turn::reply(PLUS_5),
    ]);

    let mut revs = Vec::new();
    for prompt in ["plus 5", "plus 7", "plus 9"] {
        revs.push(
            rt.evolve(&agent, prompt)
                .await
                .expect("Can evolve")
                .revision(),
        );
    }
    let [rev_5, rev_7, rev_9] = revs[..] else {
        unreachable!("three evolutions")
    };
    assert_eq!(
        (rev_5, rev_7, rev_9),
        (Revision::new(1), Revision::new(2), Revision::new(3))
    );
    assert_eq!(rt.active_revision(), rev_9);
    assert_eq!(rt.revision_count(), 4);
    assert_eq!(
        rt.loaded_revisions(),
        vec![Revision::INITIAL, rev_5, rev_7, rev_9]
    );
    for rev in rt.loaded_revisions() {
        assert!(rt.is_loaded(rev));
        assert!(versioned_so(rt, rev).exists(), "revision {rev} has a file");
    }

    // -- Refusals --------------------------------------------------------------

    let err = unsafe { rt.unload_revision(rev_9) }.expect_err("the active revision is refused");
    assert!(matches!(err, Error::UnloadActiveRevision { revision } if revision == rev_9));
    assert!(rt.is_loaded(rev_9));

    let err = unsafe { rt.unload_revision(Revision::new(99)) }.expect_err("unknown id is refused");
    assert!(matches!(
        err,
        Error::UnknownRevision { requested, latest }
            if requested == Revision::new(99) && latest == rev_9
    ));

    // -- Unload while a handle pins the revision --------------------------------

    let so_5 = versioned_so(rt, rev_5);
    assert!(
        is_mapped(&so_5),
        "sanity: the dylib is mapped before unload"
    );
    let handle = unload_step_fn(rev_5).expect("revision 1 is loaded");

    let outcome = unsafe { rt.unload_revision(rev_5) }.expect("Can unload");
    assert_eq!(outcome, Unloaded::Pinned { handles: 1 });

    // The id is a tombstone from the registry's point of view...
    assert!(!rt.is_loaded(rev_5));
    assert_eq!(rt.loaded_revisions(), vec![Revision::INITIAL, rev_7, rev_9]);
    assert_eq!(rt.revision_count(), 4, "ids are never reused");
    assert!(rt.revision_code(rev_5).is_none());
    assert!(
        unload_step_fn(rev_5).is_none(),
        "no new handles into a tombstone"
    );
    let err = rt
        .activate_revision(rev_5)
        .expect_err("a tombstone cannot be activated");
    assert!(matches!(err, Error::RevisionUnloaded { revision } if revision == rev_5));
    assert_eq!(
        rt.active_revision(),
        rev_9,
        "the refusal leaves the active revision alone"
    );

    // ...but the handle keeps the code mapped and callable.
    let mut counter = 0;
    handle.get()(&mut counter);
    assert_eq!(counter, 5, "the pinned revision still executes");
    assert!(
        so_5.exists(),
        "the file stays while a handle pins the dylib"
    );
    assert!(is_mapped(&so_5));

    // Clones pin too; the last one to go unmaps and deletes.
    let clone = handle.clone();
    drop(handle);
    assert!(so_5.exists(), "a clone still pins the dylib");
    drop(clone);
    assert!(!so_5.exists(), "the file is deleted with the last handle");
    if cfg!(target_os = "linux") {
        assert!(!is_mapped(&so_5), "dlclose unmapped the dylib");
    }

    // Unloading a tombstone is a no-op.
    let outcome = unsafe { rt.unload_revision(rev_5) }.expect("idempotent");
    assert_eq!(outcome, Unloaded::Now);

    // -- Unload with nothing pinning it ------------------------------------------

    let so_7 = versioned_so(rt, rev_7);
    let outcome = unsafe { rt.unload_revision(rev_7) }.expect("Can unload");
    assert_eq!(outcome, Unloaded::Now);
    assert!(
        !so_7.exists(),
        "the file is deleted before the call returns"
    );
    if cfg!(target_os = "linux") {
        assert!(!is_mapped(&so_7));
    }
    assert_eq!(rt.loaded_revisions(), vec![Revision::INITIAL, rev_9]);

    // Dispatch is unaffected: the active revision is still 9.
    counter = 0;
    unload_step(&mut counter);
    assert_eq!(counter, 9);

    // -- An unloaded source is not a dedup target -----------------------------------

    let rev_5_again = rt
        .evolve(&agent, "plus 5 again")
        .await
        .expect("Can evolve")
        .revision();
    assert_eq!(
        rev_5_again,
        Revision::new(4),
        "identical code to an unloaded revision is built under a new id"
    );
    assert_eq!(rt.active_revision(), rev_5_again);
    counter = 0;
    unload_step(&mut counter);
    assert_eq!(counter, 5);

    // -- Bulk retain -----------------------------------------------------------------

    let unloaded = unsafe { rt.retain_revisions([Revision::INITIAL, rev_7]) }.expect("Can retain");
    assert_eq!(
        unloaded,
        vec![rev_9],
        "keeps the elite and the active revision, ignores tombstones in `keep`"
    );
    assert_eq!(rt.loaded_revisions(), vec![Revision::INITIAL, rev_5_again]);
    assert!(!versioned_so(rt, rev_9).exists());
    assert!(versioned_so(rt, Revision::INITIAL).exists());
    assert!(versioned_so(rt, rev_5_again).exists());

    // Rolling back to a kept revision still works.
    rt.activate_revision(Revision::INITIAL)
        .expect("Can activate");
    counter = 0;
    unload_step(&mut counter);
    assert_eq!(counter, 1);

    // -- Metrics ----------------------------------------------------------------------

    let snapshot = snapshotter.snapshot().into_vec();
    assert_eq!(
        gauge_value(&snapshot, observability::REVISIONS_LOADED),
        Some(2.0),
        "the gauge tracks loaded revisions, not ids assigned"
    );
    assert_eq!(
        counter_value(
            &snapshot,
            observability::REVISION_UNLOADS,
            ("outcome", "now")
        ),
        2,
        "revisions 2 and 3 were unmapped immediately"
    );
    assert_eq!(
        counter_value(
            &snapshot,
            observability::REVISION_UNLOADS,
            ("outcome", "pinned")
        ),
        1,
        "revision 1 was pinned by a handle; the idempotent re-unload does not count"
    );
}

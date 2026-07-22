// SPDX-License-Identifier: MPL-2.0
//! End-to-end coverage for metrics emitted by a successful evolution.
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
use metrics_util::{
    CompositeKey,
    debugging::{
        DebugValue,
        DebuggingRecorder,
    },
};
use symbiont::{
    Profile,
    Runtime,
    observability,
};

const BASE_PROMPT: &str = "Implement the function. Code only.";

type SnapshotEntry = (
    CompositeKey,
    Option<metrics::Unit>,
    Option<metrics::SharedString>,
    DebugValue,
);

fn find<'a>(snapshot: &'a [SnapshotEntry], name: &str) -> Vec<&'a SnapshotEntry> {
    snapshot
        .iter()
        .filter(|(key, _, _, _)| key.key().name() == name)
        .collect()
}

fn has_label(key: &CompositeKey, k: &str, v: &str) -> bool {
    key.key().labels().any(|l| l.key() == k && l.value() == v)
}

#[tokio::test(flavor = "current_thread")]
#[cfg_attr(
    miri,
    ignore = "compiles and dlopens dylibs, which Miri does not support"
)]
async fn evolution_emits_metrics() {
    symbiont::evolvable! {
        fn metrics_step(counter: &mut usize) {
            *counter += 1;
        }
    };

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    // Unlike `with_local_recorder`, this remains active across `.await` points
    // on the single-threaded test runtime.
    let _guard = metrics::set_default_local_recorder(&recorder);

    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init runtime");
    let agent = ScriptedAgent::new([Turn::reply(
        "```rust\npub fn metrics_step(counter: &mut usize) {\n    \
         *counter += 10;\n}\n```",
    )]);
    let revision = rt
        .evolve(&agent, BASE_PROMPT)
        .await
        .expect("evolution should succeed");
    assert_eq!(revision.as_u64(), 1);

    let mut counter = 0;
    metrics_step(&mut counter);
    assert_eq!(counter, 10);

    let snapshot = snapshotter.snapshot().into_vec();

    let attempts = find(&snapshot, observability::EVOLVE_ATTEMPTS);
    assert!(
        attempts
            .iter()
            .any(|(_, _, _, v)| matches!(v, DebugValue::Histogram(values) if values == &[1.0]))
    );
    let durations = find(&snapshot, observability::EVOLVE_DURATION);
    assert!(
        durations
            .iter()
            .any(|(_, _, _, v)| matches!(v, DebugValue::Histogram(values) if values.len() == 1))
    );

    let runs = find(&snapshot, observability::LLM_RUNS);
    let ok_runs: u64 = runs
        .iter()
        .filter(|(key, _, _, _)| has_label(key, "outcome", "ok"))
        .filter_map(|(_, _, _, v)| match v {
            DebugValue::Counter(n) => Some(*n),
            _ => None,
        })
        .sum();
    assert_eq!(ok_runs, 1);

    for stage in ["llm", "parse_validate", "compile", "load"] {
        let count: usize = find(&snapshot, observability::PIPELINE_STAGE_DURATION)
            .iter()
            .filter(|(key, _, _, _)| has_label(key, "stage", stage))
            .filter_map(|(_, _, _, v)| match v {
                DebugValue::Histogram(values) => Some(values.len()),
                _ => None,
            })
            .sum();
        assert_eq!(count, 1, "stage {stage} was recorded");
    }

    assert!(
        find(&snapshot, observability::REVISIONS_LOADED)
            .iter()
            .any(|(_, _, _, v)| matches!(v, DebugValue::Gauge(g) if f64::from(*g) == 2.0))
    );
    assert!(
        find(&snapshot, observability::REVISION_ACTIVE)
            .iter()
            .any(|(_, _, _, v)| matches!(v, DebugValue::Gauge(g) if f64::from(*g) == 1.0))
    );

    let activations: u64 = find(&snapshot, observability::REVISION_ACTIVATIONS)
        .iter()
        .filter(|(key, _, _, _)| has_label(key, "source", "evolve"))
        .filter_map(|(_, _, _, v)| match v {
            DebugValue::Counter(n) => Some(*n),
            _ => None,
        })
        .sum();
    assert_eq!(activations, 1);

    for name in [
        observability::DYLIB_SOURCE_BYTES,
        observability::DYLIB_SIZE_BYTES,
    ] {
        assert!(
            find(&snapshot, name).iter().any(
                |(_, _, _, v)| matches!(v, DebugValue::Histogram(values) if values.len() == 2)
            )
        );
    }
}

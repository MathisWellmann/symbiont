// SPDX-License-Identifier: MPL-2.0
//! End-to-end coverage for the metrics emitted by a batched evolution.
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

/// Sum of a counter's value across all series with the given label.
fn counter_with_label(snapshot: &[SnapshotEntry], name: &str, k: &str, v: &str) -> u64 {
    find(snapshot, name)
        .iter()
        .filter(|(key, _, _, _)| has_label(key, k, v))
        .filter_map(|(_, _, _, value)| match value {
            DebugValue::Counter(c) => Some(*c),
            _ => None,
        })
        .sum()
}

/// The single recorded sample of a histogram that should have exactly one.
fn only_histogram_sample(snapshot: &[SnapshotEntry], name: &str) -> f64 {
    let entries = find(snapshot, name);
    assert_eq!(entries.len(), 1, "{name} should have exactly one series");
    match &entries[0].3 {
        DebugValue::Histogram(samples) => {
            assert_eq!(samples.len(), 1, "{name} should have one sample");
            samples[0].into_inner()
        }
        other => panic!("{name} should be a histogram, got {other:?}"),
    }
}

/// Source for a lane that returns `value`.
fn implementation(value: usize) -> String {
    format!(
        "```rust\npub fn metrics_batch_step(counter: &mut usize) {{ *counter += {value}; }}\n```"
    )
}

#[tokio::test(flavor = "current_thread")]
#[cfg_attr(
    miri,
    ignore = "compiles and dlopens dylibs, which Miri does not support"
)]
async fn batched_evolution_emits_batch_metrics() {
    symbiont::evolvable! {
        fn metrics_batch_step(counter: &mut usize) {
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

    // Three lanes with three outcomes worth measuring: one succeeds, one
    // duplicates it (dedup hit), one never produces code (lane failure).
    let prompts = [
        "Implement the function. Code only. Hint: winner",
        "Implement the function. Code only. Hint: twin",
        "Implement the function. Code only. Hint: hopeless",
    ];
    let hopeless = Vec::from_iter(std::iter::repeat_n(
        "No code for you.".to_string(),
        Runtime::MAX_EVOLVE_ATTEMPTS,
    ));
    let agent = RoutedAgent::new([
        ("winner", vec![implementation(10)]),
        ("twin", vec![implementation(10)]),
        (ANY_PROMPT, hopeless),
    ]);

    let results = rt.evolve_batch(&agent, &prompts).await;
    assert!(
        results[0].is_ok() && results[1].is_ok(),
        "two lanes succeed"
    );
    assert!(results[2].is_err(), "the hopeless lane fails");

    let snapshot = snapshotter.snapshot().into_vec();

    // Batch shape.
    assert_eq!(
        only_histogram_sample(&snapshot, observability::EVOLVE_BATCH_SIZE),
        3.0,
        "batch size is recorded once per batch, not once per lane"
    );
    assert!(
        only_histogram_sample(&snapshot, observability::EVOLVE_BATCH_DURATION) > 0.0,
        "batch duration must be recorded"
    );

    // Lane outcomes.
    assert_eq!(
        counter_with_label(
            &snapshot,
            observability::EVOLVE_BATCH_LANES,
            "outcome",
            "ok"
        ),
        2
    );
    assert_eq!(
        counter_with_label(
            &snapshot,
            observability::EVOLVE_BATCH_LANES,
            "outcome",
            "error"
        ),
        1
    );

    // The twin lane reused the winner's artifact instead of rebuilding it.
    let dedup = find(&snapshot, observability::REVISION_DEDUP_HITS);
    assert_eq!(
        dedup.len(),
        1,
        "the dedup counter should have been touched exactly once"
    );
    assert!(
        matches!(dedup[0].3, DebugValue::Counter(1)),
        "one lane duplicated another, got {:?}",
        dedup[0].3
    );

    // Only the successful lanes reached the build slot, and neither had to
    // queue behind the other for long enough to matter — but the metric must
    // exist for both.
    let waits = find(&snapshot, observability::BUILD_SLOT_WAIT);
    assert_eq!(waits.len(), 1, "one unlabelled build-slot-wait series");
    match &waits[0].3 {
        DebugValue::Histogram(samples) => assert_eq!(
            samples.len(),
            2,
            "one sample per lane that reached the build slot"
        ),
        other => panic!("expected a histogram, got {other:?}"),
    }

    // Per-lane duration stays per lane and is distinct from batch duration:
    // one sample per lane that reached a verdict, including the one that
    // exhausted its budget.
    let lane_durations = find(&snapshot, observability::EVOLVE_DURATION);
    assert_eq!(lane_durations.len(), 1, "one unlabelled series");
    match &lane_durations[0].3 {
        DebugValue::Histogram(samples) => assert_eq!(
            samples.len(),
            3,
            "one sample per lane — successes and the failure alike — not one per batch"
        ),
        other => panic!("expected a histogram, got {other:?}"),
    }
}

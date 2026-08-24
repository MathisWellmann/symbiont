// SPDX-License-Identifier: MPL-2.0
//! End-to-end coverage for metrics emitted by a successful evolution and by
//! a failed inference run.
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
use rig_agent::completion::PromptError;
use rig_core::completion::CompletionError;
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

/// The fatal provider failure must count once as a failed inference run and
/// once under its `reason`.
fn assert_failed_run_counted(snapshot: &[SnapshotEntry]) {
    let failed_runs: u64 = find(snapshot, observability::LLM_RUNS)
        .iter()
        .filter(|(key, _, _, _)| has_label(key, "outcome", "error"))
        .filter_map(|(_, _, _, v)| match v {
            DebugValue::Counter(n) => Some(*n),
            _ => None,
        })
        .sum();
    assert_eq!(failed_runs, 1);

    let inference_errors: u64 = find(snapshot, observability::INFERENCE_ERRORS)
        .iter()
        .filter_map(|(_, _, _, v)| match v {
            DebugValue::Counter(n) => Some(*n),
            _ => None,
        })
        .sum();
    assert_eq!(inference_errors, 1);
    assert!(
        find(snapshot, observability::INFERENCE_ERRORS)
            .iter()
            .any(|(key, _, _, _)| has_label(key, "reason", "http")),
        "the provider rejection must carry its reason label",
    );
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
        .expect("evolution should succeed")
        .revision();
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

    // A fatal provider failure must not evolve, but must still count. The
    // rejection arrives in the shape a provider with a request-id contract
    // preserves it in.
    let failing = ScriptedAgent::new([Turn::Fail(PromptError::CompletionError(
        CompletionError::from_http_response_with_request_id(
            http::StatusCode::UNAUTHORIZED,
            r#"{"error":{"message":"invalid api key"}}"#,
            None,
        ),
    ))]);
    rt.evolve(&failing, BASE_PROMPT)
        .await
        .expect_err("a provider failure must not evolve");
    assert_failed_run_counted(&snapshotter.snapshot().into_vec());
}

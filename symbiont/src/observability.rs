// SPDX-License-Identifier: MPL-2.0
//! Metrics instrumentation for the symbiont harness, built on the
//! [`metrics`] facade.
//!
//! All instrumentation lives behind the standard `metrics` facade macros: if
//! no recorder is installed (the default in tests and in host binaries that
//! don't opt in), every emission compiles down to a few atomic loads and is
//! effectively free. Long-running host binaries install a recorder once at
//! startup — typically the Prometheus exporter via [`init_observability`]
//! (feature `prometheus`) — and the whole harness becomes observable.
//!
//! ## Global labels: distinguishing processes in a fleet
//!
//! One process uses exactly one inference model (passed in once when the
//! recorder is installed), so `model` is stamped as a *global*
//! label on every series rather than threaded through individual emissions.
//! [`init_observability`] does this automatically; it also adds `crate_name`
//! and `instance` labels so many symbiont processes can be told apart in a
//! shared metrics backend:
//!
//! ```promql
//! sum by (model) (rate(symbiont_evolve_failures_total[5m]))
//! topk(5, sum by (crate_name, model) (rate(symbiont_llm_tokens_total[1h])))
//! ```
//!
//! If you use a different recorder, replicate this with its own
//! default-label mechanism (or a `Layer` that injects the labels).
//!
//! ## Metric catalogue
//!
//! | Metric                      | Kind      | Labels                 |
//! |-----------------------------|-----------|------------------------|
//! | [`EVOLVE_FAILURES`]         | counter   | `kind`                 |
//! | [`EVOLVE_ATTEMPTS`]         | histogram | —                      |
//! | [`EVOLVE_DURATION`]         | histogram | —                      |
//! | [`EVOLVE_CONTEXT_RESETS`]   | counter   | —                      |
//! | [`EVOLVE_REPEAT_RESETS`]    | counter   | —                      |
//! | [`EVOLVE_BATCH_SIZE`]       | histogram | —                      |
//! | [`EVOLVE_BATCH_DURATION`]   | histogram | —                      |
//! | [`EVOLVE_BATCH_LANES`]      | counter   | `outcome`              |
//! | [`BUILD_SLOT_WAIT`]         | histogram | —                      |
//! | [`PIPELINE_STAGE_DURATION`] | histogram | `stage`                |
//! | [`LLM_RUNS`]                | counter   | `outcome`              |
//! | [`LLM_TOKENS`]              | counter   | `kind`                 |
//! | [`LLM_RUN_INPUT_TOKENS`]    | histogram | —                      |
//! | [`LLM_RUN_OUTPUT_TOKENS`]   | histogram | —                      |
//! | [`LLM_RUN_MESSAGES`]        | histogram | —                      |
//! | [`REQUEST_BODY_BYTES`]      | histogram | —                      |
//! | [`INFERENCE_GATE_CAPACITY`] | gauge     | —                      |
//! | [`INFERENCE_IN_FLIGHT`]     | gauge     | —                      |
//! | [`INFERENCE_GATE_QUEUED`]   | gauge     | —                      |
//! | [`INFERENCE_GATE_WAIT`]     | histogram | —                      |
//! | [`LLM_TRANSIENT_RETRIES`]   | counter   | —                      |
//! | [`LLM_RETRY_BACKOFF`]       | histogram | —                      |
//! | [`REVISION_ACTIVE`]         | gauge     | —                      |
//! | [`REVISIONS_LOADED`]        | gauge     | —                      |
//! | [`REVISION_ACTIVATIONS`]    | counter   | `source`               |
//! | [`REVISION_DEDUP_HITS`]     | counter   | —                      |
//! | [`DYLIB_SIZE_BYTES`]        | histogram | —                      |
//! | [`DYLIB_SOURCE_BYTES`]      | histogram | —                      |
//!
//! All of them are registered with units and descriptions by
//! [`describe_metrics`], which [`init_observability`] calls for you.

use metrics::Unit;

/// Total failed evolution attempts, by failure kind (one of
/// `parse`, `signature`, `compile`, `no_rust_code`, `max_turns`, `llm`,
/// `dylib_load`, `io`, `other`). Emitted once per failed attempt inside the
/// self-healing loop of `Runtime::evolve`.
pub const EVOLVE_FAILURES: &str = "symbiont_evolve_failures_total";
/// Number of attempts a single `Runtime::evolve` call needed. `1` means the
/// agent produced valid, compiling code on the first try.
pub const EVOLVE_ATTEMPTS: &str = "symbiont_evolve_attempts";
/// Wall-clock seconds of a whole `Runtime::evolve` call, including all
/// self-healing retries and transient-error backoffs. For a batch this is
/// recorded once per lane, so the distribution describes lane latency rather
/// than batch latency — see [`EVOLVE_BATCH_DURATION`] for the latter.
pub const EVOLVE_DURATION: &str = "symbiont_evolve_duration_seconds";
/// Lanes per `Runtime::evolve_batch` call. Divide [`EVOLVE_BATCH_DURATION`] by
/// this to get seconds per candidate, which is the number that should fall as
/// the batch grows if the server is really batching.
pub const EVOLVE_BATCH_SIZE: &str = "symbiont_evolve_batch_size";
/// Wall-clock seconds of a whole `Runtime::evolve_batch` call: from the first
/// lane starting to the last one finishing. Because lanes overlap, this is far
/// below the sum of their [`EVOLVE_DURATION`]s — the ratio between the two is
/// the batching speedup.
pub const EVOLVE_BATCH_DURATION: &str = "symbiont_evolve_batch_duration_seconds";
/// Batch lanes that finished, by `outcome` (`ok`, `error`). Lanes fail
/// independently, so a rising `error` share against a steady batch size means
/// specific prompt variants are unproductive, not that the batch is broken.
pub const EVOLVE_BATCH_LANES: &str = "symbiont_evolve_batch_lanes_total";
/// Seconds a lane spent waiting for the build slot before it could compile.
/// Builds are serialized against one shared crate directory, so this is the
/// cost of that choice. It should stay near zero while inference dominates;
/// if it does not, the batch is bottlenecked on `cargo`, not on the model.
pub const BUILD_SLOT_WAIT: &str = "symbiont_build_slot_wait_seconds";
/// Times the chat history had to be discarded because the request exceeded
/// the model's context window. A rising value signals prompt/history bloat.
pub const EVOLVE_CONTEXT_RESETS: &str = "symbiont_evolve_context_window_resets_total";
/// Times the chat history had to be discarded because the agent repeated
/// the exact same rejected code on consecutive self-healing attempts. A
/// rising value signals a model that echoes its own broken answers instead
/// of applying corrections.
pub const EVOLVE_REPEAT_RESETS: &str = "symbiont_evolve_repeat_resets_total";
/// Wall-clock seconds per pipeline stage of one evolution attempt, labelled
/// by `stage` (`llm`, `parse_validate`, `compile`, `load`). The `llm` vs
/// `compile` split is the key capacity signal: one is paid API latency, the
/// other is local CPU.
pub const PIPELINE_STAGE_DURATION: &str = "symbiont_pipeline_stage_duration_seconds";
/// Completed agentic runs, by `outcome` (`ok`, `error`). Errors here are
/// provider or agent-loop failures, not code-validation failures.
pub const LLM_RUNS: &str = "symbiont_llm_runs_total";
/// Cumulative tokens consumed, by `kind` (`input`, `output`, `cached_input`).
/// This is the cost metric of the harness.
pub const LLM_TOKENS: &str = "symbiont_llm_tokens_total";
/// Input tokens of a single agentic run. Watch the distribution over time:
/// steady growth precedes context-window resets.
pub const LLM_RUN_INPUT_TOKENS: &str = "symbiont_llm_run_input_tokens";
/// Output tokens of a single agentic run.
pub const LLM_RUN_OUTPUT_TOKENS: &str = "symbiont_llm_run_output_tokens";
/// Messages produced by a single agentic run (assistant turns plus tool
/// exchanges). A rising mean indicates the agent needs more turns to
/// converge.
pub const LLM_RUN_MESSAGES: &str = "symbiont_llm_run_messages";
/// Serialized size of a single outbound request body to the inference
/// endpoint: the complete prompt payload (system preamble, chat history, tool
/// definitions, new turn) plus its JSON framing.
///
/// Emitted by [`crate::MeteredHttpClient`], which is the HTTP backend of the
/// agents built by [`crate::agent_builder`] and [`crate::init_agent`]. Unlike
/// [`LLM_RUN_INPUT_TOKENS`] this is per HTTP request rather than per agentic
/// run — every tool-calling turn and every retry is its own sample — and it
/// is known before the provider answers, so it is also recorded for requests
/// that the endpoint rejects for being too large. Divide it by
/// [`LLM_RUN_INPUT_TOKENS`] over the same window to get the bytes-per-token
/// ratio of the deployed tokenizer.
pub const REQUEST_BODY_BYTES: &str = "symbiont_llm_request_body_bytes";
/// Concurrent inference requests [`crate::InferenceGate`] is willing to admit,
/// i.e. [`crate::Runtime::max_in_flight`]. Exported so that saturation can be
/// read as [`INFERENCE_IN_FLIGHT`] over this, rather than against a limit the
/// dashboard would have to hardcode. `u64::MAX` worth of capacity (the
/// unlimited default) is reported as `0`, which keeps the ratio undefined
/// instead of pinning it at zero for hosts that never set a limit.
pub const INFERENCE_GATE_CAPACITY: &str = "symbiont_inference_gate_capacity";
/// Inference requests currently resident at the endpoint, as admitted by
/// [`crate::InferenceGate`]. This is the saturation signal: it should sit at
/// [`crate::Runtime::max_in_flight`] for as long as there is work left. Every
/// unit below that limit is a slot in the server's continuous batch that the
/// harness is failing to fill, and decode throughput scales with that batch.
pub const INFERENCE_IN_FLIGHT: &str = "symbiont_inference_in_flight";
/// Inference requests waiting for a slot at [`crate::InferenceGate`].
///
/// Read together with [`INFERENCE_IN_FLIGHT`]. Zero queued while in-flight
/// sits at the limit means the endpoint is the bottleneck, which is the
/// desired state. Zero queued while in-flight is *below* the limit means not
/// enough lanes are admitted to keep the server busy — the local stages
/// (compile, load) are absorbing them.
pub const INFERENCE_GATE_QUEUED: &str = "symbiont_inference_gate_queued";
/// Seconds a request spent waiting for an [`crate::InferenceGate`] slot, with
/// a zero recorded for every request admitted immediately. The share of
/// non-zero samples is how hard the limit is actually binding.
pub const INFERENCE_GATE_WAIT: &str = "symbiont_inference_gate_wait_seconds";
/// Transient HTTP errors from the provider (429, 5xx, 529) that triggered an
/// exponential-backoff retry.
pub const LLM_TRANSIENT_RETRIES: &str = "symbiont_llm_transient_retries_total";
/// Seconds slept in exponential backoff before retrying a transient error.
pub const LLM_RETRY_BACKOFF: &str = "symbiont_llm_retry_backoff_seconds";
/// Currently published revision id.
pub const REVISION_ACTIVE: &str = "symbiont_revision_active";
/// Revisions kept loaded in the registry. Because revisions are never
/// unmapped, this is a proxy for resident memory growth.
pub const REVISIONS_LOADED: &str = "symbiont_revisions_loaded";
/// Revision activations, by `source` (`evolve`, `manual`). `manual`
/// activations are rollbacks or re-deploys via `Runtime::activate_revision`.
pub const REVISION_ACTIVATIONS: &str = "symbiont_revision_activations_total";
/// Candidates that were byte-identical to an already-registered revision and
/// therefore reused it instead of being compiled again. Each hit is one
/// `cargo build` saved. A high rate against a batch means the prompt variants
/// are not actually diversifying the output — raise the sampling temperature
/// or make the per-lane hints more distinct.
pub const REVISION_DEDUP_HITS: &str = "symbiont_revision_dedup_hits_total";
/// Size in bytes of each successfully loaded dylib.
pub const DYLIB_SIZE_BYTES: &str = "symbiont_dylib_size_bytes";
/// Size in bytes of the generated Rust source per revision. Detects code
/// drift across evolutions.
pub const DYLIB_SOURCE_BYTES: &str = "symbiont_dylib_source_bytes";

/// Label values for the `kind` label of [`EVOLVE_FAILURES`].
pub(crate) mod failure_kind {
    pub(crate) const PARSE: &str = "parse";
    pub(crate) const SIGNATURE: &str = "signature";
    pub(crate) const UNSAFE_CODE: &str = "unsafe";
    pub(crate) const FORBIDDEN: &str = "forbidden";
    pub(crate) const COMPILE: &str = "compile";
    pub(crate) const NO_RUST_CODE: &str = "no_rust_code";
    pub(crate) const MAX_TURNS: &str = "max_turns";
    pub(crate) const LLM: &str = "llm";
    pub(crate) const DYLIB_LOAD: &str = "dylib_load";
    pub(crate) const IO: &str = "io";
    pub(crate) const OTHER: &str = "other";
}

/// Label values for the `stage` label of [`PIPELINE_STAGE_DURATION`].
pub(crate) mod stage {
    pub(crate) const LLM: &str = "llm";
    pub(crate) const PARSE_VALIDATE: &str = "parse_validate";
    pub(crate) const COMPILE: &str = "compile";
    pub(crate) const LOAD: &str = "load";
}

/// Register units and descriptions for every symbiont metric with the
/// installed recorder. Called by [`init_observability`]; call it manually if
/// you install your own recorder.
#[expect(
    clippy::too_many_lines,
    reason = "One `describe_*!` per metric in catalogue order; splitting the list would only hide which metrics are registered"
)]
pub fn describe_metrics() {
    use metrics::{
        describe_counter,
        describe_gauge,
        describe_histogram,
    };

    describe_counter!(
        EVOLVE_FAILURES,
        Unit::Count,
        "Failed evolution attempts by failure kind"
    );
    describe_histogram!(
        EVOLVE_ATTEMPTS,
        Unit::Count,
        "Attempts per Runtime::evolve call (1 = first try success)"
    );
    describe_histogram!(
        EVOLVE_DURATION,
        Unit::Seconds,
        "Wall-clock duration of Runtime::evolve calls, including retries"
    );
    describe_counter!(
        EVOLVE_CONTEXT_RESETS,
        Unit::Count,
        "Context-window overflows that discarded the chat history"
    );
    describe_counter!(
        EVOLVE_REPEAT_RESETS,
        Unit::Count,
        "Verbatim-repeated rejected code that discarded the chat history"
    );
    describe_histogram!(
        EVOLVE_BATCH_SIZE,
        Unit::Count,
        "Lanes per Runtime::evolve_batch call"
    );
    describe_histogram!(
        EVOLVE_BATCH_DURATION,
        Unit::Seconds,
        "Wall-clock duration of Runtime::evolve_batch calls"
    );
    describe_counter!(
        EVOLVE_BATCH_LANES,
        Unit::Count,
        "Finished batch lanes by outcome"
    );
    describe_histogram!(
        BUILD_SLOT_WAIT,
        Unit::Seconds,
        "Time a lane waited for the build slot before compiling"
    );
    describe_histogram!(
        PIPELINE_STAGE_DURATION,
        Unit::Seconds,
        "Duration of evolution pipeline stages"
    );
    describe_counter!(LLM_RUNS, Unit::Count, "Completed agentic runs by outcome");
    describe_counter!(LLM_TOKENS, Unit::Count, "Tokens consumed by the LLM");
    describe_histogram!(
        LLM_RUN_INPUT_TOKENS,
        Unit::Count,
        "Input tokens per agentic run"
    );
    describe_histogram!(
        LLM_RUN_OUTPUT_TOKENS,
        Unit::Count,
        "Output tokens per agentic run"
    );
    describe_histogram!(
        LLM_RUN_MESSAGES,
        Unit::Count,
        "Messages produced per agentic run"
    );
    describe_histogram!(
        REQUEST_BODY_BYTES,
        Unit::Bytes,
        "Serialized request body size per request to the inference endpoint"
    );
    describe_gauge!(
        INFERENCE_GATE_CAPACITY,
        Unit::Count,
        "Concurrent inference requests the gate admits (0 when unlimited)"
    );
    describe_gauge!(
        INFERENCE_IN_FLIGHT,
        Unit::Count,
        "Inference requests currently resident at the endpoint"
    );
    describe_gauge!(
        INFERENCE_GATE_QUEUED,
        Unit::Count,
        "Inference requests waiting for a concurrency slot"
    );
    describe_histogram!(
        INFERENCE_GATE_WAIT,
        Unit::Seconds,
        "Seconds a request waited for an inference concurrency slot"
    );
    describe_counter!(
        LLM_TRANSIENT_RETRIES,
        Unit::Count,
        "Transient provider errors retried with backoff"
    );
    describe_histogram!(
        LLM_RETRY_BACKOFF,
        Unit::Seconds,
        "Backoff slept before transient-error retries"
    );
    describe_gauge!(REVISION_ACTIVE, Unit::Count, "Currently active revision id");
    describe_gauge!(
        REVISIONS_LOADED,
        Unit::Count,
        "Revisions retained in the keep-all registry"
    );
    describe_counter!(
        REVISION_ACTIVATIONS,
        Unit::Count,
        "Revision activations by source"
    );
    describe_counter!(
        REVISION_DEDUP_HITS,
        Unit::Count,
        "Candidates that reused an identical registered revision instead of rebuilding"
    );
    describe_histogram!(
        DYLIB_SIZE_BYTES,
        Unit::Bytes,
        "Dylib file size per revision"
    );
    describe_histogram!(
        DYLIB_SOURCE_BYTES,
        Unit::Bytes,
        "Generated Rust source size per revision"
    );
}

/// Map an [`crate::Error`] to its `kind` label value for [`EVOLVE_FAILURES`].
pub(crate) fn failure_kind_of(e: &crate::Error) -> &'static str {
    use crate::Error::*;
    match e {
        CouldNotParseRust { .. } => failure_kind::PARSE,
        SignatureMismatch { .. } => failure_kind::SIGNATURE,
        UnsafeCode { .. } => failure_kind::UNSAFE_CODE,
        ForbiddenConstruct { .. } => failure_kind::FORBIDDEN,
        CompilationFailed { .. } => failure_kind::COMPILE,
        NoRustCode => failure_kind::NO_RUST_CODE,
        RigPrompt(rig_core::completion::PromptError::MaxTurnsError { .. }) => {
            failure_kind::MAX_TURNS
        }
        RigPrompt(_) => failure_kind::LLM,
        DylibLoad(_) => failure_kind::DYLIB_LOAD,
        Io(_) | WriteLib(_) => failure_kind::IO,
        _ => failure_kind::OTHER,
    }
}

/// Initialize metrics with a Prometheus exporter and the process-wide global
/// labels that distinguish this process in a fleet of harness binaries:
///
/// - `model`: the inference model slug passed in (`unknown` if empty). One
///   process uses one model for its lifetime, so it belongs on every series.
/// - `crate_name`: the host crate name passed in (typically
///   `env!("CARGO_PKG_NAME")`).
/// - `instance`: the `INSTANCE` env var, falling back to `<hostname>-<pid>`,
///   then `pid-<pid>`. Lets you distinguish processes even on the same host.
///
/// The exporter serves metrics over HTTP on `listen_addr`
/// (e.g. `127.0.0.1:9000/metrics`). Every symbiont metric is registered with
/// its unit and description.
///
/// # Errors
///
/// Returns [`crate::Error::Observability`] if the exporter cannot bind the
/// listener or a global recorder was already installed.
///
/// # Example
///
/// ```no_run
/// # #[cfg(feature = "prometheus")]
/// # fn f() -> symbiont::Result<()> {
/// symbiont::observability::init_observability(
///     env!("CARGO_PKG_NAME"),
///     "qwen3.6",
///     "127.0.0.1:9000".parse().unwrap(),
/// )?;
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "prometheus")]
pub fn init_observability(
    crate_name: &str,
    model: &str,
    listen_addr: std::net::SocketAddr,
) -> crate::Result<()> {
    use metrics_exporter_prometheus::PrometheusBuilder;

    let model = if model.is_empty() { "unknown" } else { model };
    let instance = std::env::var("INSTANCE")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| {
            std::fs::read_to_string("/proc/sys/kernel/hostname")
                .ok()
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
                .map(|hostname| format!("{hostname}-{}", std::process::id()))
        })
        .unwrap_or_else(|| format!("pid-{}", std::process::id()));

    PrometheusBuilder::new()
        .with_http_listener(listen_addr)
        .add_global_label("model", model)
        .add_global_label("crate_name", crate_name.to_owned())
        .add_global_label("instance", instance)
        .install()?;

    describe_metrics();
    Ok(())
}

#[cfg(test)]
mod tests {
    use metrics::{
        SharedString,
        counter,
        gauge,
        histogram,
        with_local_recorder,
    };
    use metrics_util::debugging::{
        DebugValue,
        DebuggingRecorder,
    };

    use super::*;

    /// Snapshot a single metric's labels and value by name.
    fn find<'a>(
        snapshot: &'a [(
            metrics_util::CompositeKey,
            Option<Unit>,
            Option<SharedString>,
            DebugValue,
        )],
        name: &str,
    ) -> Vec<&'a (
        metrics_util::CompositeKey,
        Option<Unit>,
        Option<SharedString>,
        DebugValue,
    )> {
        snapshot
            .iter()
            .filter(|(key, _, _, _)| key.key().name() == name)
            .collect()
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "crossbeam-epoch (via metrics-util) violates Stacked Borrows; known third-party false positive"
    )]
    fn emissions_reach_recorder_with_labels() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        with_local_recorder(&recorder, || {
            histogram!(EVOLVE_ATTEMPTS).record(3.0);
            gauge!(REVISION_ACTIVE).set(7.0);
            counter!(LLM_TOKENS, "kind" => "input").increment(42);
            counter!(LLM_TOKENS, "kind" => "output").increment(7);
        });

        let snapshot = snapshotter.snapshot().into_vec();

        let attempts = find(&snapshot, EVOLVE_ATTEMPTS);
        assert_eq!(attempts.len(), 1);
        assert!(matches!(attempts[0].3, DebugValue::Histogram(_)));

        let active = find(&snapshot, REVISION_ACTIVE);
        assert!(matches!(active[0].3, DebugValue::Gauge(v) if v == 7.0));

        let tokens = find(&snapshot, LLM_TOKENS);
        let input = tokens
            .iter()
            .find(|(key, _, _, _)| {
                key.key()
                    .labels()
                    .any(|l| l.key() == "kind" && l.value() == "input")
            })
            .expect("kind=input series exists");
        assert!(matches!(input.3, DebugValue::Counter(42)));
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "crossbeam-epoch (via metrics-util) violates Stacked Borrows; known third-party false positive"
    )]
    fn describe_metrics_registers_units_and_descriptions() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        with_local_recorder(&recorder, || {
            describe_metrics();
            // DebuggingRecorder only snapshots registered metrics, so touch
            // the ones under test after describing them.
            counter!(EVOLVE_FAILURES, "kind" => "compile").absolute(0);
            histogram!(EVOLVE_DURATION).record(0.0);
            histogram!(DYLIB_SIZE_BYTES).record(0.0);
            gauge!(REVISIONS_LOADED).set(0.0);
        });

        let snapshot = snapshotter.snapshot().into_vec();
        // Spot-check a representative sample across all three kinds.
        let expected = [
            (
                metrics_util::MetricKind::Counter,
                EVOLVE_FAILURES,
                Unit::Count,
            ),
            (
                metrics_util::MetricKind::Histogram,
                EVOLVE_DURATION,
                Unit::Seconds,
            ),
            (
                metrics_util::MetricKind::Histogram,
                DYLIB_SIZE_BYTES,
                Unit::Bytes,
            ),
            (
                metrics_util::MetricKind::Gauge,
                REVISIONS_LOADED,
                Unit::Count,
            ),
        ];
        for (kind, name, unit) in expected {
            let found = snapshot.iter().any(|(key, u, desc, _)| {
                key.kind() == kind && key.key().name() == name && *u == Some(unit) && desc.is_some()
            });
            assert!(found, "missing description for {name}");
        }
    }

    #[test]
    fn failure_kind_classification() {
        use crate::Error::*;

        assert_eq!(failure_kind_of(&NoRustCode), failure_kind::NO_RUST_CODE);
        assert_eq!(
            failure_kind_of(&CouldNotParseRust {
                code: String::new(),
                err: String::new()
            }),
            failure_kind::PARSE
        );
        assert_eq!(
            failure_kind_of(&SignatureMismatch {
                code: String::new(),
                expected: String::new(),
                got: String::new()
            }),
            failure_kind::SIGNATURE
        );
        assert_eq!(
            failure_kind_of(&UnsafeCode {
                code: String::new(),
                construct: String::new()
            }),
            failure_kind::UNSAFE_CODE
        );
        assert_eq!(
            failure_kind_of(&ForbiddenConstruct {
                code: String::new(),
                construct: String::new(),
                reason: String::new()
            }),
            failure_kind::FORBIDDEN
        );
        assert_eq!(
            failure_kind_of(&CompilationFailed {
                code: String::new(),
                err: String::new()
            }),
            failure_kind::COMPILE
        );
        assert_eq!(
            failure_kind_of(&DylibLoad("x".into())),
            failure_kind::DYLIB_LOAD
        );
        assert_eq!(failure_kind_of(&WriteLib("x".into())), failure_kind::IO);
        assert_eq!(failure_kind_of(&MutexPoison), failure_kind::OTHER);
    }
}

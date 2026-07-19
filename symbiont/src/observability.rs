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
//! One process uses exactly one inference model (read once from the `MODEL`
//! env var when the agent is built), so `model` is stamped as a *global*
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
//! | [`PIPELINE_STAGE_DURATION`] | histogram | `stage`                |
//! | [`LLM_RUNS`]                | counter   | `outcome`              |
//! | [`LLM_TOKENS`]              | counter   | `kind`                 |
//! | [`LLM_RUN_INPUT_TOKENS`]    | histogram | —                      |
//! | [`LLM_RUN_OUTPUT_TOKENS`]   | histogram | —                      |
//! | [`LLM_RUN_MESSAGES`]        | histogram | —                      |
//! | [`LLM_TRANSIENT_RETRIES`]   | counter   | —                      |
//! | [`LLM_RETRY_BACKOFF`]       | histogram | —                      |
//! | [`REVISION_ACTIVE`]         | gauge     | —                      |
//! | [`REVISIONS_LOADED`]        | gauge     | —                      |
//! | [`REVISION_ACTIVATIONS`]    | counter   | `source`               |
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
/// self-healing retries and transient-error backoffs.
pub const EVOLVE_DURATION: &str = "symbiont_evolve_duration_seconds";
/// Times the chat history had to be discarded because the request exceeded
/// the model's context window. A rising value signals prompt/history bloat.
pub const EVOLVE_CONTEXT_RESETS: &str = "symbiont_evolve_context_window_resets_total";
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
/// Size in bytes of each successfully loaded dylib.
pub const DYLIB_SIZE_BYTES: &str = "symbiont_dylib_size_bytes";
/// Size in bytes of the generated Rust source per revision. Detects code
/// drift across evolutions.
pub const DYLIB_SOURCE_BYTES: &str = "symbiont_dylib_source_bytes";

/// Label values for the `kind` label of [`EVOLVE_FAILURES`].
pub(crate) mod failure_kind {
    pub(crate) const PARSE: &str = "parse";
    pub(crate) const SIGNATURE: &str = "signature";
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
/// - `model`: the `MODEL` env var (`unknown` if unset). One process uses one
///   model for its lifetime, so it belongs on every series.
/// - `crate_name`: the host crate name passed in (typically
///   `env!("CARGO_PKG_NAME")`).
/// - `instance`: the `INSTANCE` env var, falling back to the hostname, then
///   `unknown`. Lets you correlate metrics with per-process logs.
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
///     "127.0.0.1:9000".parse().unwrap(),
/// )?;
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "prometheus")]
pub fn init_observability(
    crate_name: &str,
    listen_addr: std::net::SocketAddr,
) -> crate::Result<()> {
    use metrics_exporter_prometheus::PrometheusBuilder;

    let model = std::env::var("MODEL").unwrap_or_else(|_| "unknown".into());
    let instance = std::env::var("INSTANCE")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| {
            std::fs::read_to_string("/proc/sys/kernel/hostname")
                .ok()
                .map(|s| s.trim().to_owned())
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into());

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

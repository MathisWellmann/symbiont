//! Export an [`EvolutionTrace`](crate::EvolutionTrace) as a DeepSeek Harness
//! (`dsh`) session log.
//!
//! The harness stores one session as a JSON Lines file: a `session` header
//! record, then one session event record per line.
//!
//! The harness itself: <https://github.com/deepseek-ai/deepseek-harness>
//!
//! # The mapping
//!
//! | symbiont                                              | dsh                                  |
//! | ------------------------------------------------------| -------------------------------------|
//! | [`EvolutionTrace`]'s system prompt                    | `system/message` in the first step,  |
//! |                                                   | the first surface event of the log   |
//! | [`AttemptTrace`]                                      | one turn (`turn/start` … `turn/end`) |
//! | one assistant message plus the tool calls it made     | one step                             |
//! | [`EvolutionTrace::history`] user text                 | `user/message`                       |
//! | history assistant turn                                | `assistant/message`                  |
//! | history [`AssistantContent::ToolCall`]                | `tool/call`                          |
//! | history [`UserContent::ToolResult`]                   | `tool/result`                        |
//! | [`AttemptTrace::ladder`] and [`AttemptTrace::stages`] | a `notice` user message              |
//! | [`EvolutionTrace::outcome`]                           | a final `notice` user message        |
//!
//! The last two rows are the lossy ones. The harness has no vocabulary for a
//! reaction ladder or a build breakdown, so those travel as prose a human
//! reads. The [`EvolutionTrace`] stays the machine-readable record of a lane;
//! this format is how a human looks at it.
//!
//! # Fields the session takes from the caller
//!
//! The system prompt and the provider and model names the harness header
//! wants are recorded on the [`EvolutionTrace`] at run time, so the session
//! takes them from the trace. The caller supplies only
//!
//! - an **absolute start time**, because a trace holds only [`Duration`]s.
//!   Events are laid out from `DshSession::started_at` forward, spaced by
//!   the trace's own measurements: an attempt spans
//!   [`AttemptTrace::duration`], the model time inside it spans
//!   [`StageTimings::llm`], and the lane spans
//!   [`EvolutionTrace::duration`]. The harness derives every duration it
//!   shows by subtracting two event times, so those come out right; only the
//!   absolute instant is the caller's to supply.
//!
//! [`StageTimings::llm`]: crate::StageTimings::llm
//! [`EvolutionTrace::duration`]: crate::EvolutionTrace::duration
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(feature = "dsh-export")]
//! # {
//! use std::path::Path;
//!
//! use symbiont::{
//!     DocMode,
//!     DshSession,
//!     EvolutionTrace,
//! };
//!
//! async fn save(trace: &EvolutionTrace) -> Result<(), Box<dyn std::error::Error>> {
//!     let cwd = std::env::current_dir()?;
//!     let cwd = cwd.to_string_lossy();
//!
//!     let session = DshSession::builder()
//!         .cwd(&cwd)
//!         .build();
//!
//!     let root = Path::new(env!("HOME")).join(".dsh/sessions");
//!     let path = symbiont::export_dsh_session(trace, &session, &root)?;
//!     println!("wrote {}", path.display());
//!     Ok(())
//! }
//! # }
//! ```
mod export;
mod log;
#[cfg(feature = "dsh-export")]
mod zstd;

use std::time::Duration;

/// The record types of the harness log, from the `symbiont-dsh-log` crate.
pub use dsh_log::types;
pub use export::{
    DshSession,
    write_dsh_session,
};
use rig_core::completion::Usage;
#[cfg(feature = "dsh-export")]
pub use zstd::export_dsh_session;

use self::types::TokenUsage;

/// A duration as whole milliseconds, saturating rather than wrapping.
fn millis_of(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

/// One completion call's token accounting as a harness `TokenUsage`, or
/// `None` when the provider reported neither an input nor an output count.
///
/// Rig marks a counter the provider did not send as `None` and a reported
/// zero as `Some(0)`, so the optional harness counts pass through as they
/// are: "unreported" never reads as "no cache hits".
///
/// The harness defines its counts as **disjoint** — billed input is
/// `inputTokens + cacheReadTokens + cacheWriteTokens` — so the cached tokens
/// are taken out of the input count (see [`uncached_input`]).
fn token_usage(usage: &Usage) -> Option<TokenUsage> {
    if usage.input_tokens.is_none() && usage.output_tokens.is_none() {
        return None;
    }

    Some(TokenUsage {
        input_tokens: uncached_input(usage),
        output_tokens: usage.output_tokens.unwrap_or_default(),
        cache_read_tokens: usage.cached_input_tokens,
        cache_write_tokens: usage.cache_creation_input_tokens,
        reasoning_tokens: usage.reasoning_tokens,
    })
}

/// The input tokens of `usage` that were not served from or written to the
/// provider's cache.
///
/// Rig's [`Usage`] contract counts cache reads and writes inside
/// `input_tokens` on every provider, so they are subtracted. Saturating: the
/// counts come from the provider, and one that breaks the contract must not
/// take the export down.
fn uncached_input(usage: &Usage) -> u64 {
    let cache = usage
        .cached_input_tokens
        .unwrap_or_default()
        .saturating_add(usage.cache_creation_input_tokens.unwrap_or_default());
    usage.input_tokens.unwrap_or_default().saturating_sub(cache)
}

#[cfg(test)]
pub(super) mod tests {
    use std::time::UNIX_EPOCH;

    use rig_core::{
        completion::Usage,
        message::{
            AssistantContent,
            Message,
            ToolCall,
            ToolFunction,
            ToolName,
            ToolResultContent,
        },
    };
    use serde_json::{
        Value,
        json,
    };

    use super::*;
    use crate::{
        EvolutionTrace,
        LadderEvent,
        TraceOutcome,
        evolution_trace::{
            BuildRecord,
            RunTrace,
            StageTimings,
        },
        evolve_info::Lane,
        revision::Revision,
    };

    /// The stage timings of an attempt that got all the way through a build.
    pub(super) fn built_stages() -> StageTimings {
        let mut stages = StageTimings::default();
        stages.set_llm(Some(Duration::from_millis(900)));
        stages.set_parse_validate(Some(Duration::from_micros(80)));
        stages.set_build(Some(BuildRecord::Built {
            slot_wait: Duration::from_millis(2),
            compile: Duration::from_secs(3),
            load: Duration::from_millis(1),
            autofixes: Vec::new(),
        }));
        stages
    }

    /// A lane that called a tool, failed to compile, self-healed and then
    /// registered — the shape every field of the exporter has to survive.
    pub(super) fn sample_trace() -> EvolutionTrace {
        let call = ToolCall::from_wire(
            "call_1",
            ToolFunction::new(
                ToolName::new("api_index").expect("a non-empty name"),
                json!({ "path": "prelude" }),
            ),
        );
        let history = vec![
            Message::user("write a sort"),
            Message::from(vec![AssistantContent::ToolCall(call.clone())]),
            Message::tool_results(vec![
                call.result(vec![ToolResultContent::text("pub fn sort(..)")]),
            ]),
            Message::assistant("```rust\nfn sort() {}\n```"),
            Message::user("it did not compile: E0277"),
            Message::assistant("```rust\nfn sort() { /* fixed */ }\n```"),
        ];

        let mut trace = EvolutionTrace::new(
            "sglang".to_string(),
            "Qwen/Qwen3.8-27B-FP8".to_string(),
            Lane::from(3),
            "you write rust".to_string(),
            "write a sort".to_string(),
        );
        trace.push_attempt(
            1,
            "write a sort".to_string(),
            Some(
                RunTrace::builder()
                    .produced(0..4)
                    .response("```rust\nfn sort() {}\n```".to_string())
                    .usage(Usage::new())
                    .completion_calls(Vec::new())
                    .build(),
            ),
            StageTimings::default(),
            None,
            LadderEvent::SelfHeal {
                kind: "compile".to_string(),
                diagnostics: "E0277: the trait bound is not satisfied".to_string(),
                api_hints: Vec::new(),
            },
            Duration::from_secs(4),
        );
        trace.push_attempt(
            2,
            "it did not compile: E0277".to_string(),
            Some(
                RunTrace::builder()
                    .produced(4..6)
                    .response("```rust\nfn sort() { /* fixed */ }\n```".to_string())
                    .usage(Usage::new())
                    .completion_calls(Vec::new())
                    .build(),
            ),
            built_stages(),
            None,
            LadderEvent::Registered {
                revision: Revision::new(1),
            },
            Duration::from_secs(5),
        );
        trace.set_history(history);
        trace.set_outcome(TraceOutcome::Registered {
            revision: Revision::new(1),
        });
        trace.set_duration(Duration::from_secs(9));
        trace
    }

    pub(super) fn sample_session() -> DshSession<'static> {
        DshSession::builder()
            .cwd("/tmp/project")
            .started_at(UNIX_EPOCH + Duration::from_millis(1_700_000_000_000))
            .context_window(131_072)
            .build()
    }

    pub(super) fn export(trace: &EvolutionTrace) -> Vec<Value> {
        let session = sample_session();

        let mut buffer = Vec::new();
        write_dsh_session(trace, &session, &mut buffer).expect("writing to a Vec");
        String::from_utf8(buffer)
            .expect("valid utf-8")
            .lines()
            .map(|line| serde_json::from_str(line).expect("each line is its own JSON document"))
            .collect()
    }

    /// A `Usage` that reports `input` prompt and `output` completion tokens.
    pub(crate) fn usage_of(input: u64, output: u64) -> Usage {
        Usage::new()
            .input_tokens(input)
            .output_tokens(output)
            .total_tokens(input + output)
    }

    /// A usage the provider did not report travels as an absent record, and
    /// an unreported cache count as an absent field, never as a measured
    /// zero.
    #[test]
    fn unreported_usage_and_unreported_cache_counts_are_absent() {
        assert_eq!(token_usage(&Usage::new()), None);

        let usage = usage_of(10, 5).reasoning_tokens(3);
        let mapped = token_usage(&usage).expect("a reported usage maps");
        assert_eq!(mapped.input_tokens, 10);
        assert_eq!(mapped.cache_read_tokens, None);
        assert_eq!(mapped.cache_write_tokens, None);

        // Absent on the wire too, not `null`.
        let json = serde_json::to_string(&mapped).expect("a token usage serializes");
        assert_eq!(
            json,
            r#"{"inputTokens":10,"outputTokens":5,"reasoningTokens":3}"#
        );
    }

    /// A reported zero is a measurement: no cache hits, not "unknown".
    #[test]
    fn reported_zero_counts_stay_present() {
        let usage = Usage::new()
            .input_tokens(0)
            .cached_input_tokens(0)
            .cache_creation_input_tokens(0);
        let mapped = token_usage(&usage).expect("a reported zero input maps");
        assert_eq!(mapped.input_tokens, 0);
        assert_eq!(mapped.output_tokens, 0);
        assert_eq!(mapped.cache_read_tokens, Some(0));
        assert_eq!(mapped.cache_write_tokens, Some(0));
    }

    /// Rig counts cache reads and writes inside the input on every provider:
    /// the export takes them out, so the harness's disjoint counts bill each
    /// token once.
    #[test]
    fn cache_counts_are_taken_out_of_the_input() {
        let usage = usage_of(48_000, 300).cached_input_tokens(46_500);
        let mapped = token_usage(&usage).expect("a reported usage maps");
        assert_eq!(mapped.input_tokens, 1_500);
        assert_eq!(mapped.cache_read_tokens, Some(46_500));
        assert_eq!(mapped.cache_write_tokens, None);

        let usage = usage_of(7_500, 50)
            .cached_input_tokens(7_000)
            .cache_creation_input_tokens(200);
        let mapped = token_usage(&usage).expect("a reported usage maps");
        assert_eq!(mapped.input_tokens, 300);
        assert_eq!(mapped.cache_read_tokens, Some(7_000));
        assert_eq!(mapped.cache_write_tokens, Some(200));
    }

    /// A provider that breaks rig's contract (cache larger than the input)
    /// yields zero uncached input instead of an underflow.
    #[test]
    fn cache_larger_than_the_input_saturates() {
        let usage = usage_of(300, 50).cached_input_tokens(7_000);
        assert_eq!(uncached_input(&usage), 0);
        assert_eq!(uncached_input(&Usage::new()), 0);
    }
}

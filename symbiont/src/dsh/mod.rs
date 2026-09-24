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
/// `None` when the provider reported nothing.
///
/// Rig documents all-zero [`Usage`] as its sentinel for missing provider
/// metrics and does not distinguish that from a genuine all-zero report, so
/// an empty record becomes an absent one rather than a measured zero.
///
/// The harness defines its counts as **disjoint** — billed input is
/// `inputTokens + cacheReadTokens + cacheWriteTokens` — so the cached tokens
/// are taken out of the input count where rig's provider folded them in
/// (see [`uncached_input`]). A cache count of zero stays absent: most
/// providers do not report cache activity at all, and "unreported" must not
/// read as "no hits".
fn token_usage(usage: &Usage) -> Option<TokenUsage> {
    if usage.input_tokens == 0 && usage.output_tokens == 0 && usage.reasoning_tokens == 0 {
        return None;
    }

    let reported = |count: u64| (count > 0).then_some(count);
    Some(TokenUsage {
        input_tokens: uncached_input(usage),
        output_tokens: usage.output_tokens,
        cache_read_tokens: reported(usage.cached_input_tokens),
        cache_write_tokens: reported(usage.cache_creation_input_tokens),
        reasoning_tokens: reported(usage.reasoning_tokens),
    })
}

/// The input tokens of `usage` that were not served from or written to the
/// provider's cache.
///
/// Rig leaves the convention to the provider. OpenAI-compatible APIs
/// (OpenAI, sglang, vLLM) fold cache hits into the prompt count, and their
/// total is prompt plus completion. Anthropic reports the uncached input
/// alone, and rig's total adds the cache counts on top of it. The total tells
/// the two apart; without one, a cache count no larger than the input is
/// read as folded in, the convention of every OpenAI-compatible server.
fn uncached_input(usage: &Usage) -> u64 {
    let cache = usage
        .cached_input_tokens
        .saturating_add(usage.cache_creation_input_tokens);
    if cache == 0 {
        return usage.input_tokens;
    }
    let disjoint_total = usage
        .input_tokens
        .saturating_add(cache)
        .saturating_add(usage.output_tokens);
    let already_disjoint = usage.total_tokens == disjoint_total || cache > usage.input_tokens;
    if already_disjoint {
        usage.input_tokens
    } else {
        usage.input_tokens - cache
    }
}

#[cfg(test)]
pub(super) mod tests {
    use std::time::UNIX_EPOCH;

    use rig_core::{
        completion::Usage,
        message::{
            AssistantContent,
            Message,
            Text,
            ToolCall,
            ToolCallId,
            ToolFunction,
            ToolResult,
            ToolResultContent,
            UserContent,
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
        let call_id = ToolCallId::new("call_1").expect("a non-empty id");
        let history = vec![
            Message::user("write a sort"),
            Message::Assistant {
                id: None,
                content: vec![AssistantContent::ToolCall(ToolCall {
                    id: call_id.clone(),
                    provider: None,
                    function: ToolFunction {
                        name: "api_index".to_string(),
                        arguments: json!({ "path": "prelude" }),
                    },
                    signature: None,
                    additional_params: None,
                })],
            },
            Message::User {
                content: vec![UserContent::ToolResult(ToolResult {
                    call: call_id,
                    provider: None,
                    name: "api_index".to_string(),
                    content: vec![ToolResultContent::Text(Text::from(
                        "pub fn sort(..)".to_string(),
                    ))],
                })],
            },
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
        let mut usage = Usage::new();
        usage.input_tokens = input;
        usage.output_tokens = output;
        usage.total_tokens = input + output;
        usage
    }

    /// Rig reports all-zero usage when the provider reported nothing, so an
    /// empty record must travel as an absent field rather than a measured
    /// zero, and so must a cache count of zero.
    #[test]
    fn unreported_usage_and_unreported_cache_counts_are_absent() {
        assert_eq!(token_usage(&Usage::new()), None);

        let mut usage = usage_of(10, 5);
        usage.reasoning_tokens = 3;
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

    /// An OpenAI-compatible server (sglang) counts cache hits inside the
    /// prompt, and its total is prompt plus completion: the export takes them
    /// out, so the harness's disjoint counts bill each token once.
    #[test]
    fn cache_hits_folded_into_the_prompt_are_taken_out_of_the_input() {
        let mut usage = usage_of(48_000, 300);
        usage.cached_input_tokens = 46_500;
        let mapped = token_usage(&usage).expect("a reported usage maps");
        assert_eq!(mapped.input_tokens, 1_500);
        assert_eq!(mapped.cache_read_tokens, Some(46_500));
        assert_eq!(mapped.cache_write_tokens, None);
    }

    /// Anthropic reports the uncached input alone and rig's total adds the
    /// cache counts on top: those counts are already disjoint and pass as
    /// they are.
    #[test]
    fn disjoint_cache_counts_pass_unchanged() {
        let mut usage = usage_of(300, 50);
        usage.cached_input_tokens = 7_000;
        usage.cache_creation_input_tokens = 200;
        usage.total_tokens = 300 + 7_000 + 200 + 50;
        let mapped = token_usage(&usage).expect("a reported usage maps");
        assert_eq!(mapped.input_tokens, 300);
        assert_eq!(mapped.cache_read_tokens, Some(7_000));
        assert_eq!(mapped.cache_write_tokens, Some(200));

        // Without a total, a cache count larger than the input cannot be a
        // part of it either.
        usage.total_tokens = 0;
        assert_eq!(uncached_input(&usage), 300);
    }
}

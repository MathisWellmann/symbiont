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
//! | [`DshSession`]'s system prompt                        | `request/header` → `header.system`   |
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
//! The harness header wants these fields, so [`DshSession`] takes them from
//! the caller:
//!
//! - the **system prompt** — pass what [`crate::system_prompt`] returned for
//!   the run;
//! - the **provider and model names**, which rig's [`CompletionCall`] does not
//!   record;
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
//! [`CompletionCall`]: crate::CompletionCall
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
//!     let prompt = symbiont::system_prompt(None, DocMode::Inline).await?;
//!     let cwd = std::env::current_dir()?;
//!     let cwd = cwd.to_string_lossy();
//!
//!     let session = DshSession::builder()
//!         .system_prompt(&prompt)
//!         .provider("openrouter")
//!         .model("moonshotai/kimi-k3")
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
pub mod types;
#[cfg(feature = "dsh-export")]
mod zstd;

use std::time::Duration;

pub use export::{
    DshSession,
    write_dsh_session,
};
use rig_core::completion::Usage;
#[cfg(feature = "dsh-export")]
pub use zstd::export_dsh_session;

use crate::dsh::types::TokenUsage;

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
/// The cache fields are deliberately dropped. The harness defines its counts
/// as **disjoint** — billed input is `inputTokens + cacheReadTokens +
/// cacheWriteTokens` — while rig leaves it to the provider whether
/// `input_tokens` already contains the cached tokens. Reporting rig's cache
/// counts alongside its input count would double-bill them on the providers
/// that fold them in, so only what is unambiguous travels.
fn token_usage(usage: &Usage) -> Option<TokenUsage> {
    if usage.input_tokens == 0 && usage.output_tokens == 0 && usage.reasoning_tokens == 0 {
        return None;
    }

    Some(TokenUsage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cache_read_tokens: None,
        cache_write_tokens: None,
        reasoning_tokens: (usage.reasoning_tokens > 0).then_some(usage.reasoning_tokens),
    })
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
            LadderEvent::SelfHeal {
                kind: "compile".to_string(),
                diagnostics: "E0277: the trait bound is not satisfied".to_string(),
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
            .provider("local")
            .model("kimi")
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
    /// zero. The cache counts are dropped because the harness defines its
    /// counts as disjoint while rig does not, and reporting both would
    /// double-bill a provider that folds cache hits into its input count.
    #[test]
    fn unreported_usage_is_absent_and_cache_counts_are_dropped() {
        assert_eq!(token_usage(&Usage::new()), None);

        let mut usage = usage_of(10, 5);
        usage.reasoning_tokens = 3;
        usage.cached_input_tokens = 7;
        usage.cache_creation_input_tokens = 2;

        let mapped = token_usage(&usage).expect("a reported usage maps");
        assert_eq!(mapped.input_tokens, 10);
        assert_eq!(mapped.output_tokens, 5);
        assert_eq!(mapped.reasoning_tokens, Some(3));
        assert_eq!(mapped.cache_read_tokens, None);
        assert_eq!(mapped.cache_write_tokens, None);

        // The dropped counts must be absent on the wire too, not `null`.
        let json = serde_json::to_string(&mapped).expect("a token usage serializes");
        assert_eq!(
            json,
            r#"{"inputTokens":10,"outputTokens":5,"reasoningTokens":3}"#
        );
    }
}

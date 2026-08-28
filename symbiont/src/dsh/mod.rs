//! Export an [`EvolutionTrace`](crate::EvolutionTrace) as a DeepSeek Harness
//! (`dsh`) session log.
//!
//! The harness stores one session as a JSON Lines file: a `session` header
//! record, then one session event record per line. `seq` is dense and
//! zero-based, `time` is Unix epoch milliseconds, and the message-producing
//! events (`user/message`, `assistant/message`, `tool/result`) carry a
//! `surfaceOp` that places them on the ordered transcript the UI shows.
//!
//! [`types`] is the Rust mirror of that format. [`dsh_lines`] projects a
//! trace onto [`types::LogLine`] records, [`write_dsh_session`] serializes
//! them, and [`export_dsh_session`] writes the zstd artifact the harness
//! reads.
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
//! [`EvolutionTrace`]: crate::EvolutionTrace
//! [`EvolutionTrace::history`]: crate::EvolutionTrace::history
//! [`EvolutionTrace::outcome`]: crate::EvolutionTrace::outcome
//! [`AttemptTrace`]: crate::AttemptTrace
//! [`AttemptTrace::ladder`]: crate::AttemptTrace::ladder
//! [`AttemptTrace::stages`]: crate::AttemptTrace::stages
//! [`AttemptTrace::duration`]: crate::AttemptTrace::duration
//! [`AssistantContent::ToolCall`]: rig_core::message::AssistantContent::ToolCall
//! [`UserContent::ToolResult`]: rig_core::message::UserContent::ToolResult
//!
//! # What the trace does not hold
//!
//! Three fields the harness header wants are not in an [`EvolutionTrace`], so
//! [`DshSession`] takes them from the caller:
//!
//! - the **system prompt**, which the trace omits by design (see the
//!   [`crate::EvolutionTrace`] module docs) — pass [`crate::system_prompt`];
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
//! ```
mod export;
mod log;
pub mod types;

use std::time::Duration;

pub use export::{
    DshSession,
    dsh_lines,
    export_dsh_session,
    write_dsh_session,
};
use rig_core::completion::Usage;

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
    use super::*;

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

//! Module relating to deepseek-harness export capabilities.
mod export;
mod log;

use std::time::Duration;

pub use export::{
    DshSession,
    export_dsh_session,
    write_dsh_session,
};
use rig_core::completion::Usage;
use serde_json::{
    Value,
    json,
};

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
fn token_usage(usage: &Usage) -> Option<Value> {
    if usage.input_tokens == 0 && usage.output_tokens == 0 && usage.reasoning_tokens == 0 {
        return None;
    }

    let mut out = json!({
        "inputTokens": usage.input_tokens,
        "outputTokens": usage.output_tokens,
    });
    if usage.reasoning_tokens > 0 {
        out["reasoningTokens"] = json!(usage.reasoning_tokens);
    }
    Some(out)
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
        assert_eq!(mapped["inputTokens"], 10);
        assert_eq!(mapped["outputTokens"], 5);
        assert_eq!(mapped["reasoningTokens"], 3);
        assert!(mapped.get("cacheReadTokens").is_none());
        assert!(mapped.get("cacheWriteTokens").is_none());
    }
}

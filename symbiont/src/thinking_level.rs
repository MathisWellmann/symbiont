// SPDX-License-Identifier: MPL-2.0
//! Thinking and reasoning configuration for LLM agents.

use serde_json::Value;

/// Thinking/reasoning effort level for LLM agents.
///
/// Controls whether and how much reasoning ("thinking") the model performs
/// before responding. This translates across various inference providers and servers:
/// - **llama-server / llama.cpp & vLLM**: `chat_template_kwargs.enable_thinking`, `chat_template_kwargs.thinking`, `enable_thinking`, and `reasoning_effort`.
/// - **OpenAI / vLLM / Groq**: `reasoning_effort` (`"none"`, `"low"`, `"medium"`, `"high"`).
/// - **OpenRouter**: `reasoning.effort` (`"none"`, `"low"`, `"medium"`, `"high"`, `"max"`).
/// - **Google Gemini**: `thinking_level` (`"MINIMAL"`, `"LOW"`, `"MEDIUM"`, `"HIGH"`) or `thinking_budget`.
/// - **Anthropic**: `thinking` (`disabled`, `adaptive`, or token budget).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThinkingLevel {
    /// Disable reasoning entirely for lowest latency.
    #[default]
    Disabled,
    /// Light reasoning effort.
    Low,
    /// Moderate reasoning effort.
    Medium,
    /// Deep reasoning effort.
    High,
    /// Maximum reasoning effort supported by the model/provider.
    Max,
}

impl ThinkingLevel {
    /// Returns `true` if thinking is enabled (any non-disabled level).
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        !matches!(self, Self::Disabled)
    }

    /// Convert this thinking level into an additional parameters JSON object
    /// compatible across OpenRouter, OpenAI, vLLM, and llama.cpp.
    #[must_use]
    pub fn to_additional_params(&self) -> Value {
        match self {
            Self::Disabled => serde_json::json!({
                "enable_thinking": false,
                "reasoning_effort": "none",
                "chat_template_kwargs": {
                    "enable_thinking": false,
                    "thinking": false,
                },
                "reasoning": {
                    "effort": "none",
                    "max_tokens": 0,
                },
            }),
            Self::Low => serde_json::json!({
                "enable_thinking": true,
                "reasoning_effort": "low",
                "chat_template_kwargs": {
                    "enable_thinking": true,
                    "thinking": true,
                    "reasoning_effort": "low",
                },
                "reasoning": {
                    "effort": "low",
                },
            }),
            Self::Medium => serde_json::json!({
                "enable_thinking": true,
                "reasoning_effort": "medium",
                "chat_template_kwargs": {
                    "enable_thinking": true,
                    "thinking": true,
                    "reasoning_effort": "medium",
                },
                "reasoning": {
                    "effort": "medium",
                },
            }),
            Self::High => serde_json::json!({
                "enable_thinking": true,
                "reasoning_effort": "high",
                "chat_template_kwargs": {
                    "enable_thinking": true,
                    "thinking": true,
                    "reasoning_effort": "high",
                },
                "reasoning": {
                    "effort": "high",
                },
            }),
            Self::Max => serde_json::json!({
                "enable_thinking": true,
                "reasoning_effort": "high",
                "chat_template_kwargs": {
                    "enable_thinking": true,
                    "thinking": true,
                    "reasoning_effort": "max",
                },
                "reasoning": {
                    "effort": "max",
                },
            }),
        }
    }
}

impl From<bool> for ThinkingLevel {
    fn from(enabled: bool) -> Self {
        if enabled {
            Self::Medium
        } else {
            Self::Disabled
        }
    }
}

impl std::fmt::Display for ThinkingLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Disabled => f.write_str("disabled"),
            Self::Low => f.write_str("low"),
            Self::Medium => f.write_str("medium"),
            Self::High => f.write_str("high"),
            Self::Max => f.write_str("max"),
        }
    }
}

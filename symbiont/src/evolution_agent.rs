// SPDX-License-Identifier: MPL-2.0
//! The [`EvolutionAgent`] trait: the minimal contract the [`crate::Runtime`]
//! requires from an agent.
//!
//! One call to [`EvolutionAgent::run`] is one complete agentic run: the
//! implementation is responsible for any tool-calling turns and returns only
//! the final text alongside the new messages and token usage.
//!
//! A blanket implementation is provided for [`rig_agent::Agent`], which
//! delegates to rig's `PromptRequest` so rig owns the tool-calling loop
//! (multi-turn depth, tool dispatch, invalid-tool-call retries, hooks).

use rig_agent::{
    agent::{
        Agent,
        CompletionCall,
        PromptRequest,
    },
    completion::PromptError,
};
use rig_core::{
    completion::Usage,
    message::Message,
};

/// The result of one complete agentic run.
#[derive(Debug, Clone)]
pub struct AgentRun {
    /// The final assistant text of the run.
    pub output: String,
    /// New messages produced during this run (the prompt, assistant turns and
    /// any tool exchanges), ready to be appended to the chat history.
    pub new_messages: Vec<Message>,
    /// Aggregated token usage across all turns of the run.
    pub usage: Usage,
    /// One entry per HTTP completion request the run made, including any
    /// rig-internal retry of an invalid tool call. [`Self::usage`] stays the
    /// aggregate; these are the per-request breakdown, each with its own
    /// token usage, provider ids and finish reason.
    pub completion_calls: Vec<CompletionCall>,
}

/// The minimal contract the [`crate::Runtime`] requires from an agent:
/// one complete agentic run per call.
///
/// Implementations handle any tool-calling turns internally; the runtime only
/// consumes the final text, the new messages for its chat history, and the
/// token usage.
pub trait EvolutionAgent {
    /// Run the agent once with the given `prompt` and prior chat `history`,
    /// driving any tool-calling turns to completion.
    fn run(
        &self,
        prompt: &str,
        history: Vec<Message>,
    ) -> impl Future<Output = Result<AgentRun, PromptError>> + Send;
}

/// Clear a completion call's `raw` wire body.
///
/// The provider's verbatim response duplicates text the run already carries in
/// `new_messages` and `output`, so keeping it would store the largest string
/// of the run a third time. Everything else on the call — token usage, the
/// provider ids, and the finish reason — is not recoverable from the
/// transcript and is kept.
fn drop_raw(call: CompletionCall) -> CompletionCall {
    CompletionCall {
        raw: serde_json::Value::Null,
        ..call
    }
}

impl EvolutionAgent for Agent {
    fn run(
        &self,
        prompt: &str,
        history: Vec<Message>,
    ) -> impl Future<Output = Result<AgentRun, PromptError>> + Send {
        // `PromptRequest` clones the agent's internals, so the returned future
        // does not borrow `self`. Rig runs the tool-calling loop inside
        // `send()`, bounded by the agent's `default_max_turns`.
        let request = PromptRequest::from_agent(self, prompt)
            .history(history)
            .extended_details();
        async move {
            let response = request.await?;
            Ok(AgentRun {
                output: response.output,
                new_messages: response.messages.unwrap_or_default(),
                usage: response.usage,
                completion_calls: response
                    .completion_calls
                    .into_iter()
                    .map(drop_raw)
                    .collect(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use rig_core::completion::Usage;

    use super::*;

    /// The wire body is dropped; every field that is not recoverable from the
    /// transcript survives.
    #[test]
    fn drop_raw_keeps_everything_but_the_wire_body() {
        let call = CompletionCall::new(3, Usage::new())
            .with_raw(serde_json::json!({ "choices": [{ "message": "…" }] }));
        assert!(!call.raw.is_null(), "precondition: the call carries a body");

        let stripped = drop_raw(call.clone());

        assert!(stripped.raw.is_null(), "the wire body is dropped");
        assert_eq!(stripped.call_index, call.call_index);
        assert_eq!(stripped.usage, call.usage);
        assert_eq!(stripped.finish_reason, call.finish_reason);
        assert_eq!(stripped.message_id, call.message_id);
        assert_eq!(stripped.response_id, call.response_id);
        assert_eq!(stripped.provider_request_id, call.provider_request_id);
    }
}

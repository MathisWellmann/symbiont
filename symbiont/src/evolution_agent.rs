// SPDX-License-Identifier: MPL-2.0
//! The [`EvolutionAgent`] trait: the minimal contract the [`crate::Runtime`]
//! requires from an agent.
//!
//! One call to [`EvolutionAgent::run`] is one complete agentic run: the
//! implementation is responsible for any tool-calling turns and returns only
//! the final text alongside the new messages and token usage.
//!
//! A blanket implementation is provided for [`rig_core::agent::Agent`], which
//! delegates to rig's `PromptRequest` so rig owns the tool-calling loop
//! (multi-turn depth, tool dispatch, invalid-tool-call retries, hooks).

use rig_core::{
    agent::{
        Agent,
        PromptHook,
        PromptRequest,
    },
    completion::{
        CompletionModel,
        PromptError,
        Usage,
    },
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

impl<M, P> EvolutionAgent for Agent<M, P>
where
    M: CompletionModel + 'static,
    P: PromptHook<M> + 'static,
{
    fn run(
        &self,
        prompt: &str,
        history: Vec<Message>,
    ) -> impl Future<Output = Result<AgentRun, PromptError>> + Send {
        // `PromptRequest` clones the agent's internals, so the returned future
        // does not borrow `self`. Rig runs the tool-calling loop inside
        // `send()`, bounded by the agent's `default_max_turns`.
        let request = PromptRequest::from_agent(self, prompt)
            .with_history(history)
            .extended_details();
        async move {
            let response = request.await?;
            Ok(AgentRun {
                output: response.output,
                new_messages: response.messages.unwrap_or_default(),
                usage: response.usage,
            })
        }
    }
}

// SPDX-License-Identifier: MPL-2.0
//! The [`EvolutionAgent`] trait: the minimal contract the [`crate::Runtime`]
//! requires from an agent.
//!
//! One call to [`EvolutionAgent::run`] is one complete agentic run: the
//! implementation is responsible for any tool-calling turns and returns only
//! the final text alongside the new messages and token usage.
//!
//! An implementation is provided for [`crate::Agent`], which delegates to
//! rig's `PromptRequest` so rig owns the tool-calling loop
//! (multi-turn depth, tool dispatch, invalid-tool-call retries, hooks).
//!
//! A run that fails part-way is not a run that did nothing. A request that
//! times out on the twentieth turn follows nineteen turns of tool exchanges
//! the model paid for; rig reports the transcript for some of those failures
//! (`MaxTurnsError`) and for none of the transport ones, and never the token
//! usage. [`RunError`] therefore pairs the error with a [`PartialRun`]: what
//! the run produced before it failed, so the caller can keep the exchanges in
//! its history and the tokens in its accounting. [`crate::Agent`] fills it
//! from a hook that watches every request the run makes.

use std::sync::{
    Arc,
    Mutex,
};

use rig_agent::{
    agent::{
        AgentHook,
        CompletionCall,
        CompletionCallAction,
        CompletionCallEvent,
        CompletionResponseEvent,
        Extended,
        HookContext,
        ObservationAction,
        PromptRequest,
    },
    completion::PromptError,
};
use rig_core::{
    completion::Usage,
    message::{
        Message,
        ToolChoice,
    },
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
    /// One entry per HTTP completion request that the run made. This includes
    /// any retry of an invalid tool call that rig makes internally.
    /// [`Self::usage`] stays the aggregate. These entries are the breakdown
    /// per request, each with its own token usage, provider ids and finish
    /// reason.
    ///
    /// The blanket implementation for [`Agent`] clears the `raw` field of each
    /// entry, which holds the verbatim response body of the provider. The
    /// assistant text in that body is already in [`Self::new_messages`] and
    /// [`Self::output`]. To keep it stores the largest string of the run a
    /// third time and gives no more information. An implementor that needs the
    /// wire payload can fill the field.
    pub completion_calls: Vec<CompletionCall>,
}

/// What a failed run produced before it failed.
///
/// `new_messages` is the run's own transcript up to and including the last
/// request that was sent: the prompt, then every accepted assistant turn and
/// tool exchange. The turn that failed is not in it - there is no response
/// to record - but its request's tool results are, because they were the
/// prompt of that request. `usage` and `completion_calls` cover the requests
/// that were answered.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PartialRun {
    /// The run's messages so far, in the shape of [`AgentRun::new_messages`].
    pub new_messages: Vec<Message>,
    /// Token usage of the answered requests.
    pub usage: Usage,
    /// One entry per answered request.
    pub completion_calls: Vec<CompletionCall>,
}

impl PartialRun {
    /// Did the model answer at least once? A partial run that holds only the
    /// prompt is a run that never got going, and there is nothing to keep.
    #[must_use]
    pub fn made_progress(&self) -> bool {
        !self.completion_calls.is_empty()
    }
}

/// A failed run: why, and what it produced first.
#[derive(Debug)]
pub struct RunError {
    /// The error that ended the run.
    pub error: PromptError,
    /// What the run produced before the error, when the implementation can
    /// tell. `None` from an implementation that does not track it; the
    /// runtime then falls back to the transcript some rig errors carry.
    /// Boxed to keep the error the size of `PromptError` alone.
    pub partial: Option<Box<PartialRun>>,
}

impl From<PromptError> for RunError {
    fn from(error: PromptError) -> Self {
        Self {
            error,
            partial: None,
        }
    }
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(f)
    }
}

impl std::error::Error for RunError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// The minimal contract that the [`crate::Runtime`] requires from an agent:
/// one complete agentic run per call.
///
/// An implementation handles any tool-calling turns internally. The runtime
/// reads only the final text, the new messages for its chat history, and the
/// token usage.
pub trait EvolutionAgent {
    /// Run the agent once with `prompt` and the earlier chat `history`. The
    /// implementation drives any tool-calling turns to completion.
    fn run(
        &self,
        prompt: &str,
        history: Vec<Message>,
    ) -> impl Future<Output = Result<AgentRun, RunError>> + Send;

    /// Like [`Self::run`], but the model may not call tools: the request
    /// asks for a plain answer and a tool call in the reply is an error.
    ///
    /// The runtime uses this after a run exhausted its tool-call turn budget
    /// without producing code. Nudging such a run to "answer now" while its
    /// tools stay available does not work: a model that spent fifty turns
    /// looking up documentation resumes looking it up. Withdrawing the tools
    /// leaves it one thing to do.
    ///
    /// There is no default: the runtime relies on this request carrying no
    /// tools, and a default that forwarded to [`Self::run`] would silently
    /// void that for any implementation that registers tools. An agent that
    /// has no tools implements this by forwarding to [`Self::run`].
    fn run_without_tools(
        &self,
        prompt: &str,
        history: Vec<Message>,
    ) -> impl Future<Output = Result<AgentRun, RunError>> + Send;

    /// The agent's system prompt (preamble).
    ///
    /// The runtime records it in the [`EvolutionTrace`](crate::EvolutionTrace)
    /// of every lane it runs, so an exported session shows which prompt drove
    /// it.
    fn system_prompt(&self) -> String;

    /// The base URL of the inference endpoint this agent talks to.
    fn provider(&self) -> &str;

    /// The model the Agent uses.
    fn model(&self) -> &str;
}

/// Clear the `raw` wire body of a completion call.
///
/// The verbatim response of the provider repeats text that the run already
/// carries in `new_messages` and `output`.
fn drop_raw(call: CompletionCall) -> CompletionCall {
    CompletionCall {
        raw: serde_json::Value::Null,
        ..call
    }
}

/// A hook that keeps what a run has produced so far, so a run that dies
/// mid-loop can still report it.
///
/// Every request rig sends fires `on_completion_call` with the transcript it
/// is about to send (the input history, then the run's own messages so far,
/// then this turn's prompt); the hook keeps the part after the input. Every
/// answered request fires `on_completion_response` with its usage. On success
/// rig's own response supersedes all of this; on failure it is all there is.
struct Recorder {
    input_len: usize,
    partial: Arc<Mutex<PartialRun>>,
}

impl Recorder {
    fn new(input_len: usize) -> (Self, Arc<Mutex<PartialRun>>) {
        let partial = Arc::new(Mutex::new(PartialRun::default()));
        (
            Self {
                input_len,
                partial: Arc::clone(&partial),
            },
            partial,
        )
    }
}

impl AgentHook for Recorder {
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        if let Ok(mut partial) = self.partial.lock() {
            let mut messages: Vec<Message> = event
                .history
                .get(self.input_len..)
                .unwrap_or_default()
                .to_vec();
            messages.push(event.prompt.clone());
            partial.new_messages = messages;
        }
        CompletionCallAction::Continue
    }

    async fn on_completion_response(
        &self,
        _ctx: &HookContext,
        event: CompletionResponseEvent<'_>,
    ) -> ObservationAction {
        if let Ok(mut partial) = self.partial.lock() {
            partial.usage += event.usage;
            let index = partial.completion_calls.len();
            partial.completion_calls.push(
                CompletionCall::new(index, event.usage).with_identity(event.identity.clone()),
            );
        }
        ObservationAction::Continue
    }
}

/// Drive a prepared request to completion and shape its response. On
/// failure the error carries what `recorded` saw of the run.
async fn send(
    request: PromptRequest<Extended>,
    recorded: Arc<Mutex<PartialRun>>,
) -> Result<AgentRun, RunError> {
    match request.await {
        Ok(response) => Ok(AgentRun {
            output: response.output,
            new_messages: response.messages.unwrap_or_default(),
            usage: response.usage,
            completion_calls: response
                .completion_calls
                .into_iter()
                .map(drop_raw)
                .collect(),
        }),
        Err(error) => Err(RunError {
            error,
            partial: recorded
                .lock()
                .ok()
                .filter(|partial| partial.made_progress())
                .map(|partial| Box::new(partial.clone())),
        }),
    }
}

impl EvolutionAgent for crate::Agent {
    fn run(
        &self,
        prompt: &str,
        history: Vec<Message>,
    ) -> impl Future<Output = Result<AgentRun, RunError>> + Send {
        // `PromptRequest` clones the agent's internals, so the returned future
        // does not borrow `self`. Rig runs the tool-calling loop inside
        // `send()`, bounded by the agent's `default_max_turns`.
        let (recorder, recorded) = Recorder::new(history.len());
        send(
            PromptRequest::from_agent(&self.inner, prompt)
                .history(history)
                .extended_details()
                .add_hook(recorder),
            recorded,
        )
    }

    fn run_without_tools(
        &self,
        prompt: &str,
        history: Vec<Message>,
    ) -> impl Future<Output = Result<AgentRun, RunError>> + Send {
        // `tool_choice: none` goes out on the wire, so a compliant server
        // never returns a tool call; should one arrive anyway, rig refuses
        // to dispatch it and reports `PromptError::UnknownToolCall` with the
        // transcript, which the runtime turns into one more nudge.
        let (recorder, recorded) = Recorder::new(history.len());
        send(
            PromptRequest::from_agent(&self.inner, prompt)
                .history(history)
                .tool_choice(ToolChoice::None)
                .extended_details()
                .add_hook(recorder),
            recorded,
        )
    }

    fn system_prompt(&self) -> String {
        self.inner.run_spec().preamble.clone().unwrap_or_default()
    }

    fn provider(&self) -> &str {
        &self.provider
    }

    fn model(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use rig_core::completion::Usage;

    use super::*;

    /// The function removes the wire body. Every field that the transcript
    /// cannot give stays.
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

    /// The production `run` against rig's mock model: the recorder rides on
    /// the real agent loop, so what it keeps of a run that dies on its
    /// second request is what a real lane gets.
    mod recorder {
        use rig_agent::{
            agent::AgentBuilder,
            completion::PromptError,
            test_utils::{
                MockAddTool,
                MockCompletionModel,
                MockTurn,
            },
        };
        use rig_core::message::{
            AssistantContent,
            Message,
            UserContent,
        };
        use serde_json::json;

        use super::*;

        fn usage(input: u64, output: u64) -> Usage {
            let mut usage = Usage::new();
            usage.input_tokens = input;
            usage.output_tokens = output;
            usage
        }

        fn agent(turns: impl IntoIterator<Item = MockTurn>) -> crate::Agent {
            let inner = AgentBuilder::new(MockCompletionModel::new(turns))
                .tool(MockAddTool)
                .default_max_turns(5)
                .build();
            crate::Agent::new(inner, "mock", "mock-model")
        }

        fn is_tool_call(message: &Message) -> bool {
            matches!(message, Message::Assistant { content, .. }
                if content.iter().any(|c| matches!(c, AssistantContent::ToolCall(_))))
        }

        fn is_tool_result(message: &Message) -> bool {
            matches!(message, Message::User { content }
                if content.iter().any(|c| matches!(c, UserContent::ToolResult(_))))
        }

        /// One answered tool turn, then the provider goes away. The partial
        /// run is the prompt, the tool call and its result - the input
        /// history is not in it, the prompt is in it once - with the usage
        /// and the one completion call of the answered turn.
        #[tokio::test]
        async fn a_run_that_fails_after_a_tool_turn_reports_what_it_had() {
            let agent = agent([
                MockTurn::tool_call("call-1", "add", json!({"x": 1, "y": 2}))
                    .with_usage(usage(100, 7)),
                MockTurn::request_error("connection reset"),
            ]);
            let history = vec![Message::user("earlier"), Message::assistant("ok")];

            let RunError { error, partial } = agent
                .run("add one and two", history)
                .await
                .expect_err("the second request fails");
            assert!(
                matches!(error, PromptError::CompletionError(_)),
                "{error:?}"
            );
            let partial = *partial.expect("the run answered once");

            assert_eq!(partial.new_messages.len(), 3, "{:?}", partial.new_messages);
            assert_eq!(partial.new_messages[0], Message::user("add one and two"));
            assert!(is_tool_call(&partial.new_messages[1]));
            assert!(is_tool_result(&partial.new_messages[2]));
            assert_eq!(partial.usage, usage(100, 7));
            assert_eq!(partial.completion_calls.len(), 1);
            assert_eq!(partial.completion_calls[0].usage, usage(100, 7));
        }

        /// A failure on the first request answered nothing: no partial, so
        /// the runtime resends the prompt instead of appending a duplicate.
        #[tokio::test]
        async fn a_run_that_fails_at_once_reports_nothing() {
            let agent = agent([MockTurn::request_error("connection reset")]);
            let RunError { partial, .. } = agent
                .run("add one and two", Vec::new())
                .await
                .expect_err("the first request fails");
            assert!(partial.is_none());
        }

        /// A run that finishes carries rig's own account; the recorder
        /// changes nothing about it.
        #[tokio::test]
        async fn a_finished_run_is_reported_by_rig() {
            let agent = agent([
                MockTurn::tool_call("call-1", "add", json!({"x": 1, "y": 2}))
                    .with_usage(usage(100, 7)),
                MockTurn::text("3").with_usage(usage(120, 1)),
            ]);
            let run = agent
                .run("add one and two", Vec::new())
                .await
                .expect("the run finishes");
            assert_eq!(run.output, "3");
            assert_eq!(run.usage, usage(220, 8));
            assert_eq!(run.completion_calls.len(), 2);
            assert_eq!(run.new_messages.len(), 4, "{:?}", run.new_messages);
        }

        /// Under `run_without_tools` the wire request carries
        /// `tool_choice: none`, and a tool call that comes back anyway is
        /// refused. rig refuses it before the response is observed, so the
        /// recorder sees no answered request and reports no partial; the
        /// error itself carries the transcript, which is the runtime's
        /// fallback (`Error::aborted_run_messages`).
        #[tokio::test]
        async fn without_tools_a_tool_call_is_refused() {
            let model = MockCompletionModel::new([
                MockTurn::tool_call("call-1", "add", json!({"x": 1, "y": 2}))
                    .with_usage(usage(50, 3)),
                MockTurn::text("never requested"),
            ]);
            let recorded = model.clone();
            let inner = AgentBuilder::new(model)
                .tool(MockAddTool)
                .default_max_turns(5)
                .build();
            let agent = crate::Agent::new(inner, "mock", "mock-model");

            let RunError { error, partial } = agent
                .run_without_tools("answer now", Vec::new())
                .await
                .expect_err("a tool call under tool_choice none is refused");
            assert!(
                matches!(error, PromptError::UnknownToolCall { ref tool_name, .. } if tool_name == "add"),
                "{error:?}"
            );
            let requests = recorded.requests();
            assert_eq!(requests.len(), 1);
            assert_eq!(requests[0].tool_choice, Some(ToolChoice::None));
            assert!(partial.is_none(), "{partial:?}");
            let PromptError::UnknownToolCall { chat_history, .. } = error else {
                unreachable!()
            };
            assert_eq!(chat_history[0], Message::user("answer now"));
            assert!(is_tool_call(&chat_history[1]));
        }
    }
}

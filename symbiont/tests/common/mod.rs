// SPDX-License-Identifier: MPL-2.0
//! Shared agent test doubles for the integration tests.
//!
//! [`ScriptedAgent`] replays a fixed sequence of [`Turn`]s and records the
//! exact prompt and chat-history length it receives on every call, so each
//! test can assert on the precise feedback the runtime generated.
//!
//! [`RoutedAgent`] instead answers based on what the prompt *contains*. Batch
//! lanes run concurrently and finish in nondeterministic order, so a FIFO
//! script cannot address a particular lane — routing on prompt content can.
#![allow(
    dead_code,
    reason = "Each integration test binary only uses a subset of these helpers"
)]

use std::{
    collections::VecDeque,
    sync::Mutex,
};

use rig_agent::completion::PromptError;
use rig_core::{
    completion::Usage,
    message::Message,
};
use symbiont::{
    AgentRun,
    EvolutionAgent,
};

/// A single scripted agent turn.
pub(crate) enum Turn {
    /// Respond with this canned assistant text.
    Reply(String),
    /// Respond with this canned assistant text and this token usage.
    ReplyWithUsage(String, Usage),
    /// Fail the run with this error.
    Fail(PromptError),
}

impl Turn {
    /// Convenience constructor for a canned reply.
    pub(crate) fn reply(text: &str) -> Self {
        Self::Reply(text.to_string())
    }

    /// Convenience constructor for a canned reply with explicit token usage.
    pub(crate) fn reply_with_usage(text: &str, usage: Usage) -> Self {
        Self::ReplyWithUsage(text.to_string(), usage)
    }
}

/// Deterministic [`EvolutionAgent`] test double.
///
/// Pops one scripted [`Turn`] per [`EvolutionAgent::run`] call. Panics if the
/// runtime asks for more turns than were scripted, so a misbehaving retry
/// loop fails the test loudly instead of looping forever.
pub(crate) struct ScriptedAgent {
    /// Remaining scripted turns.
    script: Mutex<VecDeque<Turn>>,
    /// Prompts received, in call order.
    prompts: Mutex<Vec<String>>,
    /// Chat-history length received on each call, in call order.
    history_lens: Mutex<Vec<usize>>,
}

impl ScriptedAgent {
    /// Create an agent that replays `turns` in order.
    pub(crate) fn new(turns: impl IntoIterator<Item = Turn>) -> Self {
        Self {
            script: Mutex::new(VecDeque::from_iter(turns)),
            prompts: Mutex::new(Vec::new()),
            history_lens: Mutex::new(Vec::new()),
        }
    }

    /// Number of times the runtime invoked this agent.
    pub(crate) fn calls(&self) -> usize {
        self.prompts.lock().expect("Mutex is not poisoned").len()
    }

    /// The prompt received on call `idx` (0-based).
    pub(crate) fn prompt(&self, idx: usize) -> String {
        self.prompts.lock().expect("Mutex is not poisoned")[idx].clone()
    }

    /// The chat-history length received on call `idx` (0-based).
    pub(crate) fn history_len(&self, idx: usize) -> usize {
        self.history_lens.lock().expect("Mutex is not poisoned")[idx]
    }
}

impl EvolutionAgent for ScriptedAgent {
    async fn run(&self, prompt: &str, history: Vec<Message>) -> Result<AgentRun, PromptError> {
        self.prompts
            .lock()
            .expect("Mutex is not poisoned")
            .push(prompt.to_string());
        self.history_lens
            .lock()
            .expect("Mutex is not poisoned")
            .push(history.len());

        let turn = self
            .script
            .lock()
            .expect("Mutex is not poisoned")
            .pop_front()
            .expect("ScriptedAgent ran out of scripted turns — unexpected extra retry");

        let (text, usage) = match turn {
            Turn::Reply(text) => (text, Usage::new()),
            Turn::ReplyWithUsage(text, usage) => (text, usage),
            Turn::Fail(err) => return Err(err),
        };
        let new_messages = vec![Message::user(prompt), Message::assistant(text.as_str())];
        Ok(AgentRun {
            output: text,
            new_messages,
            usage,
        })
    }
}

/// An [`EvolutionAgent`] that answers by matching the prompt against a routing
/// table, rather than by position in a script.
///
/// Concurrent batch lanes interleave their calls in nondeterministic order, so
/// "the n-th call" is not a stable way to address a lane. Each route is a
/// `(needle, reply)` pair: the first route whose `needle` appears in the prompt
/// wins. A lane can therefore be given a deterministic answer no matter when it
/// happens to run.
///
/// Replies are consumed in order per route, so a route can hand out a broken
/// answer first and a good one on the retry.
///
/// Note that the runtime's correction prompts replace the base prompt rather
/// than extending it, so a lane's retries no longer carry its distinguishing
/// hint. Give such a lane the [`ANY_PROMPT`] catch-all route, placed last.
pub(crate) struct RoutedAgent {
    /// Routing table: prompt needle -> remaining replies for that route.
    routes: Mutex<Vec<(String, VecDeque<String>)>>,
    /// Prompts received, in call order.
    prompts: Mutex<Vec<String>>,
}

/// Needle that matches every prompt: every string contains the empty string.
/// Use it as the last route of a [`RoutedAgent`] to catch retries, whose
/// correction prompts no longer carry the lane's distinguishing hint.
pub(crate) const ANY_PROMPT: &str = "";

impl RoutedAgent {
    /// Build an agent from `(needle, replies)` routes. Routes are matched in
    /// the given order, so put more specific needles first and
    /// [`ANY_PROMPT`] last.
    pub(crate) fn new<N, R>(routes: impl IntoIterator<Item = (N, R)>) -> Self
    where
        N: Into<String>,
        R: IntoIterator,
        R::Item: Into<String>,
    {
        Self {
            routes: Mutex::new(Vec::from_iter(routes.into_iter().map(
                |(needle, replies)| {
                    (
                        needle.into(),
                        VecDeque::from_iter(replies.into_iter().map(Into::into)),
                    )
                },
            ))),
            prompts: Mutex::new(Vec::new()),
        }
    }

    /// Number of times the runtime invoked this agent, across all lanes.
    pub(crate) fn calls(&self) -> usize {
        self.prompts.lock().expect("Mutex is not poisoned").len()
    }

    /// Every prompt received, in (nondeterministic) call order.
    pub(crate) fn prompts(&self) -> Vec<String> {
        self.prompts.lock().expect("Mutex is not poisoned").clone()
    }
}

impl EvolutionAgent for RoutedAgent {
    async fn run(&self, prompt: &str, _history: Vec<Message>) -> Result<AgentRun, PromptError> {
        self.prompts
            .lock()
            .expect("Mutex is not poisoned")
            .push(prompt.to_string());

        let text = {
            let mut routes = self.routes.lock().expect("Mutex is not poisoned");
            let (_, replies) = routes
                .iter_mut()
                .find(|(needle, _)| prompt.contains(needle.as_str()))
                .expect("RoutedAgent received a prompt matching no route");
            replies
                .pop_front()
                .expect("RoutedAgent route ran out of replies — unexpected extra retry")
        };

        let new_messages = vec![Message::user(prompt), Message::assistant(text.as_str())];
        Ok(AgentRun {
            output: text,
            new_messages,
            usage: Usage::new(),
        })
    }
}

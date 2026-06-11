// SPDX-License-Identifier: MPL-2.0
//! Shared scripted-agent test double for the backpressure integration tests.
//!
//! [`ScriptedAgent`] replays a fixed sequence of [`Turn`]s and records the
//! exact prompt and chat-history length it receives on every call, so each
//! test can assert on the precise feedback the runtime generated.
#![allow(
    dead_code,
    reason = "Each integration test binary only uses a subset of these helpers"
)]

use std::{
    collections::VecDeque,
    sync::Mutex,
};

use rig_core::{
    completion::{
        PromptError,
        Usage,
    },
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
    /// Fail the run with this error.
    Fail(PromptError),
}

impl Turn {
    /// Convenience constructor for a canned reply.
    pub(crate) fn reply(text: &str) -> Self {
        Self::Reply(text.to_string())
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

        match turn {
            Turn::Reply(text) => {
                let new_messages = vec![Message::user(prompt), Message::assistant(text.as_str())];
                Ok(AgentRun {
                    output: text,
                    new_messages,
                    usage: Usage::new(),
                })
            }
            Turn::Fail(err) => Err(err),
        }
    }
}

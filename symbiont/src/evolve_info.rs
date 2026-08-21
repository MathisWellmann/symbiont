// SPDX-License-Identifier: MPL-2.0
//! The result of a successful evolution call.

use getset::{
    CopyGetters,
    Getters,
};
use rig_core::completion::Usage;
use serde::{
    Deserialize,
    Serialize,
};

use crate::{
    evolution_trace::EvolutionTrace,
    revision::Revision,
};

/// Everything a caller needs from a successful [`crate::Runtime::evolve`]
/// call.
///
/// The `usage` field is the sum of every LLM request that the call made. It
/// includes the self-healing retries whose output the runtime rejected. Those
/// runs also consumed tokens, so they are part of the real cost of the call.
#[derive(Debug, Clone, Getters, CopyGetters, Serialize, Deserialize)]
pub struct EvolveInfo {
    /// The revision that the runtime registered the new implementation under.
    #[getset(get_copy = "pub")]
    revision: Revision,

    /// The token usage of the LLM requests of the call.
    #[getset(get = "pub")]
    usage: Usage,

    /// The full trajectory of the lane that produced this revision. It holds
    /// every prompt and nudge, every response and tool exchange, every
    /// recovery decision of the harness, and the timings of each stage.
    #[getset(get = "pub")]
    trace: EvolutionTrace,
}

impl EvolveInfo {
    /// Create a new instance.
    pub(crate) fn new(revision: Revision, usage: Usage, trace: EvolutionTrace) -> Self {
        Self {
            revision,
            usage,
            trace,
        }
    }

    /// Take the trajectory. This consumes the info.
    #[must_use]
    pub fn into_trace(self) -> EvolutionTrace {
        self.trace
    }
}

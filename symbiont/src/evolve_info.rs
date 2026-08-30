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

/// A dedicated lane with an independent retry ladder allows concurrent code evolution.
#[derive(
    Debug,
    Clone,
    Copy,
    Eq,
    PartialEq,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    derive_more::Display,
    derive_more::From,
)]
pub struct Lane(u32);

impl Lane {
    /// Get the inner value.
    pub fn get(&self) -> u32 {
        self.0
    }
}

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

    /// The full trajectory of the lane that produced this revision. It holds
    /// every prompt and nudge, every response and tool exchange, every
    /// recovery decision of the harness, and the timings of each stage.
    #[getset(get = "pub")]
    trace: EvolutionTrace,
}

impl EvolveInfo {
    /// Create a new instance.
    pub(crate) fn new(revision: Revision, trace: EvolutionTrace) -> Self {
        Self { revision, trace }
    }

    /// Take the trajectory. This consumes the info.
    #[must_use]
    pub fn into_trace(self) -> EvolutionTrace {
        self.trace
    }

    /// The lane it evolved in.
    #[inline(always)]
    pub fn lane(&self) -> Lane {
        self.trace.lane()
    }

    /// The token usage information.
    pub fn usage(&self) -> Usage {
        self.trace.usage()
    }
}

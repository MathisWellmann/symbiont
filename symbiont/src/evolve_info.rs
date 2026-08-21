// SPDX-License-Identifier: MPL-2.0
//! The result of a successful evolution call.

use getset::{
    CopyGetters,
    Getters,
};
use rig_core::completion::Usage;

use crate::revision::Revision;

/// Everything a caller needs from a successful [`crate::Runtime::evolve`]
/// call.
///
/// The `usage` field is the sum over every LLM request the call made,
/// including self-healing retries whose output was rejected: those runs
/// consumed tokens too, so they are part of the call's real cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Getters, CopyGetters)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct EvolveInfo {
    /// The revision the new implementation was registered under.
    #[getset(get_copy = "pub")]
    revision: Revision,

    /// Token usage of the call's LLM requests.
    #[getset(get = "pub")]
    usage: Usage,
}

impl EvolveInfo {
    /// Create a new instance.
    pub(crate) fn new(revision: Revision, usage: Usage) -> Self {
        Self { revision, usage }
    }
}

// SPDX-License-Identifier: MPL-2.0
//! The built-in tools the evolution agent can call.
//!
//! Every tool is a [`rig_core::tool::PortableTool`] and answers from state
//! the process already holds: none of them runs cargo or does I/O inside a
//! tool call, so a call never blocks on a build behind the inference timeout.
//!
//! | Tool | Type | Registered by |
//! |---|---|---|
//! | `api_index` | [`ApiIndexTool`] | [`crate::agent_builder`] in a [`DocMode`](crate::DocMode) with tools |
//! | `api_doc` | [`ApiDocTool`] | [`crate::agent_builder`] in a [`DocMode`](crate::DocMode) with tools |
//! | `revision_source` | [`RevisionSourceTool`] | the host, with `.tool(..)` on the builder |
//!
//! Tool errors go through an explicit [`rig_core::tool::ToolExecutionError`]
//! constructor rather than the default of `PortableTool::map_error`: the
//! default redacts the message, and the model then reads "the tool failed"
//! without learning what to call next.

mod doc;
mod revision;

pub use doc::{
    ApiDocArgs,
    ApiDocTool,
    ApiIndexArgs,
    ApiIndexTool,
};
pub use revision::{
    RevisionSourceArgs,
    RevisionSourceError,
    RevisionSourceTool,
};

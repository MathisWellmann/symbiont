// SPDX-License-Identifier: MPL-2.0
//! The built-in tools the evolution agent can call.
//!
//! Every tool is a [`rig_core::tool::PortableTool`].
//!
//! | Tool | Type | Registered by |
//! |---|---|---|
//! | `api_index` | [`ApiIndexTool`] | [`crate::agent_builder`] in a [`DocMode`](crate::DocMode) with tools |
//! | `api_doc` | [`ApiDocTool`] | [`crate::agent_builder`] in a [`DocMode`](crate::DocMode) with tools |
//! | `revision_source` | [`RevisionSourceTool`] | the host, with `.tool(..)` or [`crate::with_revision_tools`] |
//! | `build_revision` | [`BuildRevisionTool`] | the host, with [`crate::with_revision_tools`] |
//! | `edit_revision` | [`EditRevisionTool`] | the host, with [`crate::with_revision_tools`] |
//! | `submit_revision` | [`SubmitRevisionTool`] | the host, with [`crate::with_revision_tools`] |
//!
//! The documentation tools answer from state the process already holds and
//! do no I/O. The revision tools run the pipeline: parse, validate, compile,
//! register. A compile inside a tool call holds no inference slot. The gate
//! admits *requests*, and rig calls a tool between two requests, so the lane
//! occupies the endpoint no more than a lane that answers with code. What a
//! tool build does contend for is the process-wide build slot, the same one
//! a response's build waits for.
//!
//! The revision tools need the lane they are called from. The ladder
//! attaches it to the run's task (see [`ToolContext`](context::ToolContext)),
//! so the tools work inside [`crate::Runtime::evolve`] and refuse elsewhere.
//!
//! Tool errors go through an explicit [`rig_core::tool::ToolExecutionError`]
//! constructor rather than the default of `PortableTool::map_error`: the
//! default redacts the message, and the model then reads "the tool failed"
//! without learning what to call next.

mod build;
pub(crate) mod context;
mod doc;
mod edit;
mod pipeline;
mod revision;
mod submit;

pub use build::{
    BuildRevisionArgs,
    BuildRevisionTool,
};
pub use context::RevisionToolError;
pub(crate) use context::revision_list;
pub use doc::{
    ApiDocArgs,
    ApiDocTool,
    ApiIndexArgs,
    ApiIndexTool,
};
pub use edit::{
    EditRevisionArgs,
    EditRevisionTool,
};
pub use revision::{
    RevisionSourceArgs,
    RevisionSourceError,
    RevisionSourceTool,
};
pub use submit::{
    SubmitRevisionArgs,
    SubmitRevisionTool,
};

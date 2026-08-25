// SPDX-License-Identifier: MPL-2.0
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/MathisWellmann/symbiont/main/assets/logo.svg"
)]
#![doc = include_str!("../README.md")]
// Under Miri the timing code falls back to `std::time::Instant`, because
// minstant's `#[ctor]` probes the TSC via `rdtsc`, which Miri cannot
// interpret. Leaving the crate unused (and thus unlinked) keeps its ctor
// out of the interpreted binary.
#![cfg_attr(miri, allow(unused_crate_dependencies))]

mod compiler;
#[cfg(debug_assertions)]
mod debug_call_counter;
mod decl;
mod doc_index;
mod doc_string;
mod doc_tools;
mod dylib_config;
mod dylib_dependency;
mod error;
mod evolution_agent;
mod evolution_trace;
mod evolve_error;
mod evolve_failure;
mod evolve_info;
mod inference;
mod inference_gate;
mod init_tracing;
mod metered_http;
pub mod observability;
mod parser;
mod profile;
mod revision;
mod runtime;
mod system_prompt;
mod thinking_level;
mod unwind;
mod utils;
mod validation;

pub use decl::{
    EvolvableDecl,
    FullSource,
};
pub use doc_index::{
    DocIndex,
    DocIndexError,
};
pub use doc_tools::{
    ApiDocTool,
    ApiIndexTool,
};
pub use dylib_config::DylibConfig;
pub use dylib_dependency::{
    DylibDependency,
    DylibPatch,
};
pub use error::{
    Error,
    Result,
};
pub use evolution_agent::{
    AgentRun,
    EvolutionAgent,
};
pub use evolution_trace::{
    AttemptTrace,
    BuildRecord,
    EvolutionTrace,
    LadderEvent,
    RunTrace,
    StageTimings,
    TraceOutcome,
};
pub use evolve_error::EvolveError;
pub use evolve_failure::EvolveFailure;
pub use evolve_info::EvolveInfo;
pub use inference::{
    DOC_TOOLS_MAX_TURNS,
    INFERENCE_REQUEST_TIMEOUT,
    agent_builder,
    agent_builder_from_env,
    init_agent,
    init_agent_from_env,
};
pub use init_tracing::init_tracing;
pub use metered_http::MeteredHttpClient;
pub use profile::Profile;
pub use revision::{
    Revision,
    RevisionFn,
};
// Reachable through `AgentRun::completion_calls`, so hosts need it nameable
// without depending on `rig-agent` directly.
pub use rig_agent::agent::CompletionCall;
// Reachable through `EvolveInfo::usage` and `EvolutionTrace::usage`, so hosts
// tallying token spend need it nameable without depending on `rig-core`
// directly.
pub use rig_core::completion::Usage;
// Reachable through `AgentRun::new_messages` and `EvolutionTrace::history`, so
// hosts inspecting the agent transcript need it nameable without depending
// on `rig-core` directly.
pub use rig_core::message::Message;
pub use runtime::Runtime;
// Reachable through `ThinkingLevel::to_additional_params`, so hosts merging
// provider parameters need it nameable without depending on `serde_json`
// directly.
pub use serde_json::Value;
/// Evolvable return types must implement [`Default`]: when an evolved
/// implementation panics, the in-dylib `catch_unwind` wrapper substitutes
/// `Default::default()` as a safe placeholder return value. The bound is
/// enforced at the declaration site:
///
/// ```compile_fail
/// struct NoDefault;
///
/// symbiont::evolvable! {
///     fn make() -> NoDefault;
/// }
/// ```
pub use symbiont_macros::evolvable;
pub use system_prompt::{
    DocMode,
    system_prompt,
};
pub use thinking_level::ThinkingLevel;

/// type alias for the return type of `init_agent`
pub type Agent = rig_agent::Agent;

/// Type alias for the pre-configured agent builder.
///
/// The builder is always in rig's `WithBuilderTools` state: a
/// [`DocMode`](crate::DocMode) with tools registers `api_index` and
/// `api_doc` on it, and the other modes carry an empty tool set. Register
/// your own tools on it with rig's builder API before calling `.build()`,
/// e.g. `.tool(MyTool).default_max_turns(10).build()`.
///
/// The state cannot depend on the [`DocMode`](crate::DocMode):
/// [`agent_builder`] has one return type, and registering a tool is a
/// one-way typestate step in rig. The state is therefore fixed for every
/// mode. An empty tool set builds the same [`Agent`] that rig's
/// `NoToolConfig` state builds, so only the builder API differs:
///
/// - Before 0.29 this alias was `rig_agent::AgentBuilder<NoToolConfig>`. Code
///   that names that state for the value [`agent_builder`] returns has to
///   name this alias instead.
/// - `.tool_server_handle(..)` lives on `NoToolConfig` and is out of reach
///   here. A host that shares one `ToolServer` between agents builds its own
///   `rig_agent::AgentBuilder`, with [`system_prompt`] as the preamble and,
///   for a tool [`DocMode`](crate::DocMode), [`ApiIndexTool`] and
///   [`ApiDocTool`] over a [`DocIndex`] on its own tool server.
pub type AgentBuilder = rig_agent::AgentBuilder<rig_agent::agent::WithBuilderTools>;

/// Internal module for macro-generated dispatch code.
///
/// Not part of the public API — used by `evolvable!` expansion.
#[doc(hidden)]
pub mod __internal {
    #[cfg(debug_assertions)]
    pub use crate::debug_call_counter::{
        CallGuard,
        enter_call,
    };
    pub use crate::runtime::revision_fn_lookup;
}

#[cfg(test)]
mod tests {
    #[expect(unused, reason = "Used in benchmarks.")]
    use criterion::*;
    // Only used in integration tests; linked here to satisfy
    // `unused_crate_dependencies` for the lib test target.
    use http as _;
}

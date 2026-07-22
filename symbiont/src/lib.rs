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
mod doc_string;
mod dylib_config;
mod dylib_dependency;
mod error;
mod evolution_agent;
mod evolve_failure;
mod inference;
mod init_tracing;
pub mod observability;
mod parser;
mod profile;
mod revision;
mod runtime;
mod system_prompt;
mod unwind;
mod utils;
mod validation;

pub use decl::{
    EvolvableDecl,
    FullSource,
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
pub use evolve_failure::EvolveFailure;
pub use inference::{
    agent_builder,
    init_agent,
};
pub use init_tracing::init_tracing;
pub use profile::Profile;
pub use revision::{
    Revision,
    RevisionFn,
};
use rig_core::providers::openrouter::CompletionModel;
pub use runtime::Runtime;
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

/// type alias for the return type of `init_agent`
pub type Agent = rig_core::agent::Agent<CompletionModel>;

/// Type alias for the pre-configured agent builder.
///
/// Register your own tools on it with rig's builder API before calling
/// `.build()`, e.g. `.tool(MyTool).default_max_turns(5).build()`.
/// Note that registering the first tool transitions the builder's typestate
/// (to `AgentBuilder<_, _, WithBuilderTools>`); the resulting [`Agent`] is
/// unchanged and works with [`Runtime::evolve`] either way.
pub type AgentBuilder = rig_core::agent::AgentBuilder<CompletionModel>;

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

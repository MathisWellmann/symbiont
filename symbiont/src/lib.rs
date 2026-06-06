// SPDX-License-Identifier: MPL-2.0
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/MathisWellmann/symbiont/main/assets/logo.svg"
)]
#![doc = include_str!("../README.md")]

mod compiler;
#[cfg(debug_assertions)]
mod debug_call_counter;
mod decl;
mod doc_string;
mod dylib_config;
mod dylib_dependency;
mod error;
mod inference;
mod init_tracing;
mod parser;
mod profile;
mod runtime;
mod system_prompt;
mod unwind;
mod update_pointers;
mod utils;
mod validation;

pub use decl::{
    EvolvableDecl,
    FullSource,
};
pub use dylib_config::DylibConfig;
pub use dylib_dependency::DylibDependency;
pub use error::{
    Error,
    Result,
};
pub use inference::init_agent;
pub use init_tracing::init_tracing;
pub use profile::Profile;
use rig_core::providers::openrouter::CompletionModel;
pub use runtime::Runtime;
pub use symbiont_macros::evolvable;

/// type alias for the return type of `init_agent`
pub type Agent = rig_core::agent::Agent<CompletionModel>;

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
}

#[cfg(test)]
mod tests {
    #[expect(unused, reason = "Used in benchmarks.")]
    use criterion::*;
}

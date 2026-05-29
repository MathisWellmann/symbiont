// SPDX-License-Identifier: MPL-2.0
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/MathisWellmann/symbiont/main/assets/logo.svg"
)]
#![doc = include_str!("../README.md")]

mod compiler;
mod decl;
mod dylib_config;
mod dylib_dependency;
mod error;
mod inference;
mod init_tracing;
mod parser;
mod runtime;
mod unwind;
mod utils;
mod validation;

pub use compiler::Profile;
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
pub use runtime::Runtime;
pub use symbiont_macros::evolvable;

/// Internal module for macro-generated dispatch code.
///
/// Not part of the public API — used by `evolvable!` expansion.
#[doc(hidden)]
pub mod __internal {
    #[cfg(debug_assertions)]
    pub use crate::runtime::{
        CallGuard,
        enter_call,
    };
}

#[cfg(test)]
mod tests {
    #[expect(unused, reason = "Used in benchmarks.")]
    use criterion::*;
}

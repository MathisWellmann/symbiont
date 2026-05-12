// SPDX-License-Identifier: MPL-2.0
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/MathisWellmann/symbiont/main/assets/logo.svg"
)]
#![doc = include_str!("../README.md")]

pub mod error;
pub mod inference;
pub mod runtime;

mod compiler;
mod decl;
mod init_tracing;
mod parser;
mod unwind;
mod utils;
mod validation;

// Re-export the proc macro.
// Re-export key types.
pub use compiler::Profile;
pub use decl::{
    EvolvableDecl,
    FullSource,
};
pub use error::{
    Error,
    Result,
};
pub use init_tracing::init_tracing;
pub use runtime::Runtime;
pub use symbiont_macros::{
    evolvable,
    shared,
};

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

// SPDX-License-Identifier: MPL-2.0
//! Errors of this crate.

/// Errors that can occur during symbiont runtime operations.
#[derive(Debug, thiserror::Error)]
#[expect(missing_docs, reason = "Self explaining")]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Syn(#[from] syn::Error),

    #[error(transparent)]
    RigPrompt(#[from] rig::completion::PromptError),

    #[error(transparent)]
    RigHttp(#[from] rig::http_client::Error),

    #[error("The text does not contain any rust code.")]
    NoRustCode,

    #[error("Could not parse Rust code.")]
    CouldNotParseRust,

    #[error("Failed to write lib.rs: {0}")]
    WriteLib(String),

    #[error("Validation failed: signature mismatch for '{name}'. Expected: {expected}. Got: {got}")]
    SignatureMismatch {
        name: String,
        expected: String,
        got: String,
    },

    #[error("Compilation failed:\n{0}")]
    CompilationFailed(String),

    #[error("No evolvable functions found. Use the evolvable! macro to declare at least one.")]
    NoEvolvableFunctions,

    #[error("Runtime already initialized. Call Runtime::init() only once.")]
    AlreadyInitialized,

    #[error("Failed to load dylib: {0}")]
    DylibLoad(String),
}

/// Result type alias for symbiont operations.
pub type Result<T, E = Error> = std::result::Result<T, E>;

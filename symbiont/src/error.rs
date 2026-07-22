// SPDX-License-Identifier: MPL-2.0
//! Errors of this crate.

use crate::Revision;

/// Errors that can occur during symbiont runtime operations.
#[derive(Debug, thiserror::Error)]
#[expect(missing_docs, reason = "Self explaining")]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Syn(#[from] syn::Error),

    #[error(transparent)]
    RigPrompt(#[from] rig_core::completion::PromptError),

    #[error(transparent)]
    RigHttp(#[from] rig_core::http_client::Error),

    #[error("The mutex was poisoned")]
    MutexPoison,

    #[error("The text does not contain any rust code.")]
    NoRustCode,

    #[error("Could not parse Rust code: {err}")]
    CouldNotParseRust { code: String, err: String },

    #[error("Failed to write lib.rs: {0}")]
    WriteLib(String),

    #[error("Validation failed: signature mismatch in {got}. Expected: {expected}")]
    SignatureMismatch {
        code: String,
        expected: String,
        got: String,
    },

    #[error("Unsafe code is forbidden in evolvable code: found {construct}")]
    UnsafeCode { code: String, construct: String },

    #[error("Compilation failed:\n{err}")]
    CompilationFailed { code: String, err: String },

    #[error("No evolvable functions found. Use the evolvable! macro to declare at least one.")]
    NoEvolvableFunctions,

    #[error("Runtime already initialized. Call Runtime::init() only once.")]
    AlreadyInitialized,

    #[error("Failed to load dylib: {0}")]
    DylibLoad(String),

    #[error("Unknown revision {requested}; the latest registered revision is {latest}")]
    UnknownRevision {
        requested: Revision,
        latest: Revision,
    },

    #[error("Evolution failed after {attempts} attempts. Last error: {last_error}")]
    MaxRetriesExceeded {
        attempts: usize,
        last_error: Box<Error>,
    },

    #[cfg(feature = "prometheus")]
    #[error(transparent)]
    Observability(#[from] metrics_exporter_prometheus::BuildError),

    #[error("Could not run cargo doc command")]
    CargoDoc,

    #[error("Could not convert json docs to markdown")]
    MdDoc,

    #[error(transparent)]
    Fmt(#[from] std::fmt::Error),
}

/// Result type alias for symbiont operations.
pub type Result<T, E = Error> = std::result::Result<T, E>;

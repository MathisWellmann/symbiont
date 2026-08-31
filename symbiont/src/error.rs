// SPDX-License-Identifier: MPL-2.0
//! Errors of this crate.

use std::fmt::Write;

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
    RigPrompt(#[from] rig_agent::completion::PromptError),

    #[error(transparent)]
    RigHttp(#[from] rig_core::http_client::Error),

    #[error(transparent)]
    DocIndex(#[from] crate::DocIndexError),

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

    #[error("Forbidden construct in evolvable code: found {construct} ({reason})")]
    ForbiddenConstruct {
        code: String,
        construct: String,
        reason: String,
    },

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

    #[error("cargo doc generation failed: {err}")]
    CargoDoc { err: String },

    #[error("Failed to read rustdoc JSON at {path}: {err}")]
    RustdocJson { path: String, err: String },

    #[error("Could not convert json docs to markdown")]
    MdDoc,

    #[error(transparent)]
    Fmt(#[from] std::fmt::Error),

    #[error("A DocMode with tools requires an `opt_crate_name` to be set.")]
    InvalidDocMode,
}

impl Error {
    /// Convert the error into a nudging prompt for the Agent
    pub(crate) fn to_nudge(self, prompt: &mut String) -> Result<(), Self> {
        use Error::*;
        match self {
            NoRustCode => prompt.push_str(
                "nudge: Your response did not contain a rust code block. Please try again and make sure its wrapped like this: ```CODE```",
            ),
            CouldNotParseRust { code, err } => write!(prompt,
                "nudge: Your generated code ```{code}``` is not valid Rust. Parse error: ```{err}```. Fix the syntax error and respond with the full corrected code.",
            ).expect("Can write to prompt"),
            RigPrompt(rig_agent::completion::PromptError::MaxTurnsError { .. }) => prompt.push_str(
                "nudge: You exhausted the tool-call turn budget before producing code. Respond with the final Rust code block now.",
            ),
            SignatureMismatch {
                code: _,
                expected,
                got,
            } => write!(prompt,
                "nudge: Signature mismatch, got: {got}.\n
                Expected `{expected}`.\n
                Fix ONLY this function's signature (argument types and return type must match exactly, argument names may differ).",
            ).expect("Can write to prompt"),
            UnsafeCode { code, construct } => write!(prompt,
                "nudge: Your generated code contains {construct}, but unsafe code is forbidden in evolvable code. \
                Rewrite it in safe Rust only: no `unsafe` blocks, `unsafe fn`, `unsafe impl`, `unsafe trait`, \
                `extern` blocks, unsafe attributes, or `unsafe` tokens inside macros. \
                Keep the logic and the function signatures unchanged. Full code: ```{code}```",
            ).expect("Can write to prompt"),
            ForbiddenConstruct { code: _, construct, reason } => write!(prompt,
                "nudge: Your generated code contains {construct}, which is forbidden in evolvable code: {reason}.\n
                Rewrite the code without it, keeping the logic and the function signatures unchanged if possible.",
            ).expect("Can write to prompt"),
            CompilationFailed{code: _, err} => write!(prompt,
                "nudge: Your generated code failed to compile. Compiler output:\n```\n{err}\n```\n\
                Fix the compilation errors while preserving the existing logic and behaviour if possible.\n
                Change only the expressions the compiler diagnostics point at.",
            ).expect("Can write to prompt"),
            _ => {
                return Err(self);
            },
        };
        Ok(())
    }
}

/// Result type alias for symbiont operations.
pub type Result<T, E = Error> = std::result::Result<T, E>;

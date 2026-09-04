// SPDX-License-Identifier: MPL-2.0
//! Errors of this crate.

use std::fmt::Write;

use rig_core::message::Message;

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

    #[error("Function `{fn_name}` is not implemented: {reason}")]
    UnimplementedFunction {
        code: String,
        fn_name: String,
        reason: String,
    },

    /// The candidate did not compile. `err` is the text the model reads:
    /// every error rendered by rustc, numbered `[E1]..[En]`, with locations
    /// in the candidate's own line numbers. `diagnostics` is the same set,
    /// structured: spans as byte ranges into `code`, error codes and rustc's
    /// suggestions. Empty when cargo itself failed before rustc reported
    /// anything; `err` is cargo's stderr then.
    #[error("Compilation failed:\n{err}")]
    CompilationFailed {
        code: String,
        err: String,
        diagnostics: Vec<crate::Diagnostic>,
    },

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
    pub(crate) fn nudge(self, prompt: &mut String) -> Result<(), Self> {
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
            UnimplementedFunction { code, fn_name, reason } => write!(prompt,
                "nudge: Your generated code leaves `fn {fn_name}` unimplemented: {reason}.\n
                A stub or the unchanged default body is not an evolution.\n
                Respond with a complete, real implementation matching the declared signature.\n
                Full code: ```{code}```",
            ).expect("Can write to prompt"),
            CompilationFailed{code: _, err, diagnostics: _} => write!(prompt,
                "nudge: Your generated code failed to compile. Line numbers refer to your code block. Compiler output:\n```\n{err}\n```\n\
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

impl Error {
    /// The messages an aborted agent run added before it failed, if the
    /// error carries them.
    ///
    /// Rig reports the canonical transcript it reached when a run dies
    /// inside the tool-calling loop: [`PromptError::MaxTurnsError`],
    /// [`PromptError::PromptCancelled`] and [`PromptError::UnknownToolCall`]
    /// all carry `chat_history` — the input history the caller passed in,
    /// followed by every message the run produced (the prompt, assistant
    /// turns, and tool results). Skipping the first `input_len` messages
    /// yields exactly the run's own messages.
    ///
    /// A retry that appends these sees the tool exchanges the aborted run
    /// already made instead of replaying the identical request that just
    /// exhausted its budget. Errors without a transcript (provider HTTP
    /// failures, validation errors, ...) return `None`: those runs either
    /// produced nothing or the pipeline owns what they produced.
    pub(crate) fn aborted_run_messages(&self, input_len: usize) -> Option<Vec<Message>> {
        use rig_agent::completion::PromptError;

        let full: &[Message] = match self {
            Error::RigPrompt(PromptError::MaxTurnsError { chat_history, .. })
            | Error::RigPrompt(PromptError::UnknownToolCall { chat_history, .. }) => {
                chat_history.as_slice()
            }
            Error::RigPrompt(PromptError::PromptCancelled { chat_history, .. }) => {
                chat_history.as_slice()
            }
            _ => return None,
        };
        Some(full.iter().skip(input_len).cloned().collect())
    }
}

/// Result type alias for symbiont operations.
pub type Result<T, E = Error> = std::result::Result<T, E>;

#[cfg(test)]
mod tests {
    use rig_agent::completion::PromptError;
    use rig_core::message::Message;

    use super::*;

    /// The messages the aborted run produced, with the input history the
    /// caller passed in skipped by count.
    #[test]
    fn max_turns_error_yields_the_run_delta() {
        let input = [Message::user("base prompt")];
        let run = vec![
            Message::user("base prompt"),
            Message::assistant("tool call"),
            Message::user("tool result"),
        ];
        let err = Error::RigPrompt(PromptError::MaxTurnsError {
            max_turns: 3,
            chat_history: Box::new(run.clone()),
            prompt: Box::new(Message::user("base prompt")),
        });

        let recovered = err
            .aborted_run_messages(input.len())
            .expect("MaxTurnsError carries a transcript");

        assert_eq!(recovered, run[1..]);
    }

    /// The transcript-less shape: when the run reached no state, the
    /// recovered delta is empty but still `Some`, and the retry proceeds
    /// exactly as before the fix.
    #[test]
    fn max_turns_error_with_empty_transcript_yields_nothing() {
        let err = Error::RigPrompt(PromptError::MaxTurnsError {
            max_turns: 3,
            chat_history: Box::new(Vec::new()),
            prompt: Box::new(Message::user("base prompt")),
        });

        let recovered = err
            .aborted_run_messages(0)
            .expect("MaxTurnsError carries a transcript");

        assert!(recovered.is_empty());
    }

    /// The other two abort variants carry the same field, so the same
    /// recovery applies to them.
    #[test]
    fn cancelled_and_unknown_tool_call_errors_are_recovered_too() {
        let run = vec![Message::user("base prompt"), Message::assistant("partial")];

        let cancelled = Error::RigPrompt(PromptError::PromptCancelled {
            chat_history: run.clone(),
            reason: "hook terminated".to_string(),
        });
        assert_eq!(
            cancelled
                .aborted_run_messages(0)
                .expect("carries transcript"),
            run
        );

        let unknown_tool = Error::RigPrompt(PromptError::UnknownToolCall {
            tool_name: "nope".to_string(),
            available_tools: Vec::new(),
            allowed_tools: Vec::new(),
            chat_history: Box::new(run.clone()),
        });
        assert_eq!(
            unknown_tool
                .aborted_run_messages(1)
                .expect("carries transcript"),
            run[1..]
        );
    }

    /// Errors without a transcript must not fabricate messages.
    #[test]
    fn other_errors_carry_no_messages() {
        assert!(Error::NoRustCode.aborted_run_messages(0).is_none());
        assert!(
            Error::RigPrompt(PromptError::CompletionError(
                rig_core::completion::CompletionError::ProviderError("boom".to_string()),
            ))
            .aborted_run_messages(0)
            .is_none()
        );
    }
}

// SPDX-License-Identifier: MPL-2.0
//! Records of failed evolution attempts that fed backpressure to the agent.

use getset::{
    CopyGetters,
    Getters,
};

use crate::{
    error::Error,
    observability::failure_kind_of,
};

/// One failed attempt inside the self-healing loop of
/// [`crate::Runtime::evolve`].
///
/// Captures exactly the failures that are rendered back into the retry
/// prompt as backpressure: missing code blocks, parse errors, exhausted
/// tool-call turn budgets, signature mismatches, forbidden unsafe code,
/// and compilation failures.
/// Hosts can drain these via [`crate::Runtime::take_evolve_failures`] and
/// persist them for offline analysis of common failure patterns, e.g. to
/// tune prompts or the documented API surface.
#[derive(Debug, Clone, Getters, CopyGetters)]
pub struct EvolveFailure {
    /// 1-based attempt index within a single `evolve` call.
    #[getset(get_copy = "pub")]
    attempt: usize,
    /// Failure kind label; the same values as the `kind` label of
    /// [`crate::observability::EVOLVE_FAILURES`]: one of `no_rust_code`,
    /// `parse`, `max_turns`, `signature`, `unsafe` or `compile`.
    #[getset(get_copy = "pub")]
    kind: &'static str,
    /// The generated source that failed. Empty when the agent produced no
    /// code at all (`no_rust_code`, `max_turns`).
    #[getset(get = "pub")]
    generated_code: String,
    /// The diagnostics fed back to the agent: rustc stderr for `compile`,
    /// the parse error for `parse`, the mismatch description for
    /// `signature`, the offending construct for `unsafe`, and the
    /// corrective nudge otherwise.
    #[getset(get = "pub")]
    diagnostics: String,
}

impl EvolveFailure {
    /// Build a record from an evolution error, returning `None` for errors
    /// that do not feed backpressure to the agent (transient HTTP errors,
    /// IO failures, dylib load failures, ...).
    pub fn from_error(error: &Error, attempt: usize) -> Option<Self> {
        let (generated_code, diagnostics) = match error {
            Error::NoRustCode => (String::new(), error.to_string()),
            Error::RigPrompt(rig_core::completion::PromptError::MaxTurnsError { .. }) => {
                (String::new(), error.to_string())
            }
            Error::CouldNotParseRust { code, err } => (code.clone(), err.clone()),
            Error::SignatureMismatch {
                code,
                expected,
                got,
            } => (
                code.clone(),
                format!("signature mismatch in `{got}`; expected `{expected}`"),
            ),
            Error::UnsafeCode { code, construct } => (code.clone(), construct.clone()),
            Error::CompilationFailed { code, err } => (code.clone(), err.clone()),
            _ => return None,
        };
        Some(Self {
            attempt,
            kind: failure_kind_of(error),
            generated_code,
            diagnostics,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compilation_failure_is_recorded() {
        let failure = EvolveFailure::from_error(
            &Error::CompilationFailed {
                code: "fn f() {}".to_string(),
                err: "error[E0308]: mismatched types".to_string(),
            },
            3,
        )
        .expect("compilation failures feed backpressure");

        assert_eq!(failure.attempt(), 3);
        assert_eq!(failure.kind(), "compile");
        assert_eq!(failure.generated_code(), "fn f() {}");
        assert_eq!(failure.diagnostics(), "error[E0308]: mismatched types");
    }

    #[test]
    fn signature_mismatch_is_recorded() {
        let failure = EvolveFailure::from_error(
            &Error::SignatureMismatch {
                code: "fn g(x: u8) {}".to_string(),
                expected: "fn g(x: u32)".to_string(),
                got: "fn g(x: u8)".to_string(),
            },
            1,
        )
        .expect("signature mismatches feed backpressure");

        assert_eq!(failure.kind(), "signature");
        assert!(failure.diagnostics().contains("fn g(x: u32)"));
    }

    #[test]
    fn unsafe_code_is_recorded() {
        let failure = EvolveFailure::from_error(
            &Error::UnsafeCode {
                code: "pub fn f() { unsafe {} }".to_string(),
                construct: "an `unsafe` block: `unsafe { }`".to_string(),
            },
            2,
        )
        .expect("unsafe code feeds backpressure");

        assert_eq!(failure.attempt(), 2);
        assert_eq!(failure.kind(), "unsafe");
        assert_eq!(failure.generated_code(), "pub fn f() { unsafe {} }");
        assert!(failure.diagnostics().contains("an `unsafe` block"));
    }

    #[test]
    fn no_rust_code_is_recorded_without_source() {
        let failure = EvolveFailure::from_error(&Error::NoRustCode, 1)
            .expect("missing code blocks feed backpressure");

        assert_eq!(failure.kind(), "no_rust_code");
        assert!(failure.generated_code().is_empty());
    }

    #[test]
    fn non_backpressure_errors_are_ignored() {
        assert!(EvolveFailure::from_error(&Error::MutexPoison, 1).is_none());
        assert!(EvolveFailure::from_error(&Error::DylibLoad("boom".to_string()), 1).is_none());
    }
}

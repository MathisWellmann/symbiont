// SPDX-License-Identifier: MPL-2.0
//! The error of a failed evolution, carrying the lane's full trajectory.

use crate::{
    error::Error,
    evolution_trace::EvolutionTrace,
};

/// A failed [`crate::Runtime::evolve`] call, with the
/// [`EvolutionTrace`] of the lane that failed.
///
/// The trace is the whole point of this type: a failed lane is exactly the
/// case where the trajectory matters, and returning a bare [`Error`] would
/// throw away every prompt, response and recovery decision that led to it.
/// Inspect it with [`Self::trace`], or match on the underlying failure with
/// [`Self::error`].
///
/// # Discarding the trace
///
/// `EvolveError` converts to [`Error`] with `?`, which **drops the trace**.
/// That is convenient in a host whose `main` returns [`Error`] and which does
/// not persist traces; if you want the trajectory, match instead of
/// propagating:
///
/// ```rust,ignore
/// match runtime.evolve(&agent, prompt).await {
///     Ok(info) => persist(info.trace()),
///     Err(err) => {
///         persist(err.trace());
///         return Err(err.into());
///     }
/// }
/// ```
#[derive(Debug, thiserror::Error, Getters)]
#[error("{error}")]
pub struct EvolveError {
    /// The failure the retry ladder ended on.
    #[source]
    #[getset(get = "pub")]
    error: Error,

    /// The lane's full trajectory.
    ///
    /// Boxed to keep `Result<EvolveInfo, EvolveError>` small: the trace is
    /// wider than the error itself, and it is the cold path of a call that
    /// spends most of its time waiting on inference.
    #[getset(get = "pub")]
    trace: Box<EvolutionTrace>,
}

impl EvolveError {
    /// Pair a failure with the trajectory that produced it.
    pub(crate) fn new(error: Error, trace: EvolutionTrace) -> Self {
        Self {
            error,
            trace: Box::new(trace),
        }
    }

    /// Take the trajectory, consuming the error.
    #[must_use]
    pub fn into_trace(self) -> EvolutionTrace {
        *self.trace
    }

    /// Split into the failure and the trajectory.
    #[must_use]
    pub fn into_parts(self) -> (Error, EvolutionTrace) {
        (self.error, *self.trace)
    }
}

impl From<EvolveError> for Error {
    /// **Discards the trace.** See the type-level documentation.
    fn from(error: EvolveError) -> Self {
        error.error
    }
}

impl From<Error> for EvolveError {
    /// Wraps a failure that happened outside any lane — before one started, or
    /// in shared runtime state — so it carries an empty trace. A failure
    /// *within* a lane always carries that lane's real trajectory.
    fn from(error: Error) -> Self {
        Self {
            error,
            trace: Box::new(EvolutionTrace::empty()),
        }
    }
}

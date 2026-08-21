// SPDX-License-Identifier: MPL-2.0
//! The error of a failed evolution. It carries the full trajectory of the
//! lane.

use getset::Getters;

use crate::{
    error::Error,
    evolution_trace::EvolutionTrace,
};

/// A failed [`crate::Runtime::evolve`] call, with the
/// [`EvolutionTrace`] of the lane that failed.
///
/// The trace is the purpose of this type. A failed lane is the case where the
/// trajectory matters most. A bare [`Error`] discards every prompt, response
/// and recovery decision that caused the failure. Read the trajectory with
/// [`Self::trace`]. Match on the failure itself with [`Self::error`].
///
/// # How the trace is lost
///
/// The `?` operator converts `EvolveError` to [`Error`], which **removes the
/// trace**. This is convenient for a host that returns [`Error`] from `main`
/// and does not persist traces. If you want the trajectory, match on the
/// result instead:
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
    /// The failure that the retry ladder ended on.
    #[source]
    #[getset(get = "pub")]
    error: Error,

    /// The full trajectory of the lane.
    ///
    /// Boxed to keep `Result<EvolveInfo, EvolveError>` small. The trace is
    /// wider than the error itself. This is also the cold path of a call that
    /// waits on inference for most of its time.
    #[getset(get = "pub")]
    trace: Box<EvolutionTrace>,
}

impl EvolveError {
    /// Join a failure to the trajectory that produced it.
    pub(crate) fn new(error: Error, trace: EvolutionTrace) -> Self {
        Self {
            error,
            trace: Box::new(trace),
        }
    }

    /// Take the trajectory. This consumes the error.
    #[must_use]
    pub fn into_trace(self) -> EvolutionTrace {
        *self.trace
    }

    /// Divide the error into the failure and the trajectory.
    #[must_use]
    pub fn into_parts(self) -> (Error, EvolutionTrace) {
        (self.error, *self.trace)
    }
}

impl From<EvolveError> for Error {
    /// **This removes the trace.** Read the documentation of
    /// [`EvolveError`].
    fn from(error: EvolveError) -> Self {
        error.error
    }
}

impl From<Error> for EvolveError {
    /// Wraps a failure that occurred outside any lane, either before a lane
    /// started or in shared runtime state. Such an error carries an empty
    /// trace. A failure inside a lane always carries the real trajectory of
    /// that lane.
    fn from(error: Error) -> Self {
        Self {
            error,
            trace: Box::new(EvolutionTrace::empty()),
        }
    }
}

// SPDX-License-Identifier: MPL-2.0
//! The `evaluate_revision` tool: the host's judgement of a revision, on
//! demand, for the agent.

use std::fmt::Write;

use rig_core::tool::{
    PortableTool,
    ToolExecutionError,
};
use serde::Deserialize;

use crate::{
    EXPECT_WRITE,
    Revision,
    Runtime,
    tools::context::ToolContext,
};

/// The error of an `evaluate_revision` call.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EvaluateRevisionError {
    /// No revision with the requested number is registered.
    #[error("revision {requested} is not registered; registered revisions: 0..={latest}")]
    UnknownRevision {
        /// The number the agent asked for.
        requested: Revision,
        /// The highest registered revision.
        latest: Revision,
    },
    /// The host's evaluation did not produce a report.
    #[error("the evaluation of revision {revision} failed: {reason}")]
    Failed {
        /// The revision under evaluation.
        revision: Revision,
        /// The host's account of the failure, as the agent reads it.
        reason: String,
    },
}

/// Make the text of the error visible to the model. See [`crate::tools`].
fn model_visible(error: EvaluateRevisionError) -> ToolExecutionError {
    match error {
        EvaluateRevisionError::UnknownRevision { .. } => {
            ToolExecutionError::not_found(error.to_string()).with_retryable(false)
        }
        EvaluateRevisionError::Failed { .. } => {
            ToolExecutionError::other(error.to_string()).with_retryable(false)
        }
    }
}

/// The `evaluate_revision` tool: run the host's evaluation of one
/// registered revision and hand the report to the agent.
///
/// Symbiont cannot know what makes one revision better than another: the
/// fitness function is the host's. This tool is the adapter. The host
/// supplies the evaluation as a closure over a [`Revision`]; the tool checks
/// that the revision is registered, runs the closure, and returns its
/// report as text. Call the revision through the `<name>_fn(revision)`
/// accessors `evolvable!` generates, so the evaluation runs the revision
/// under test and not the active one.
///
/// With `build_revision` and `submit_revision` this closes the loop inside
/// one run: build variants, evaluate each, submit the best. The closure runs
/// inside the agent run, between two inference requests, so it should be
/// quick relative to inference; a long evaluation is better run by the host
/// after the lane, over the revisions the trace names.
///
/// Not registered by [`crate::with_revision_tools`], which has no
/// evaluation to offer: register it with `.tool(..)` on the builder.
///
/// ```no_run
/// # use symbiont::{EvaluateRevisionTool, Revision, Runtime};
/// # async fn example(rt: &'static Runtime) {
/// symbiont::evolvable! {
///     fn score(x: f64) -> f64 { x }
/// };
/// let tool = EvaluateRevisionTool::new(
///     rt,
///     "Run the benchmark against a revision and report its mean error.",
///     async |revision: Revision| {
///         let f = score_fn(revision).ok_or("revision vanished")?;
///         let error: f64 = (1..=10).map(|i| (f.get()(f64::from(i)) - 2.0 * f64::from(i)).abs()).sum();
///         Ok(format!("mean absolute error: {:.3}", error / 10.0))
///     },
/// );
/// # let _ = tool;
/// # }
/// ```
#[derive(Clone)]
pub struct EvaluateRevisionTool<F> {
    runtime: &'static Runtime,
    description: String,
    evaluate: F,
    /// Scored evaluations in a row without a new best that end the lane's
    /// search; `None` never stops it.
    patience: Option<usize>,
}

/// The host's judgement of one revision: the report the agent reads and,
/// optionally, a score the tool ranks revisions by.
///
/// A closure that returns plain text (`Result<String, String>`) reports
/// without a score. With a score the tool keeps the lane's leaderboard and
/// can apply the stopping rule of [`EvaluateRevisionTool::with_stopping_rule`].
#[derive(Debug, Clone, PartialEq)]
pub struct Evaluation {
    report: String,
    score: Option<f64>,
}

impl Evaluation {
    /// A report without a score.
    pub fn new(report: impl Into<String>) -> Self {
        Self {
            report: report.into(),
            score: None,
        }
    }

    /// Attach the score the leaderboard ranks by; higher is better. A score
    /// that is not finite (a revision that could not be scored) is not
    /// ranked.
    #[must_use]
    pub fn with_score(mut self, score: f64) -> Self {
        self.score = Some(score);
        self
    }
}

impl From<String> for Evaluation {
    fn from(report: String) -> Self {
        Self::new(report)
    }
}

impl<F, Fut, E> EvaluateRevisionTool<F>
where
    F: Fn(Revision) -> Fut + Send + Sync,
    Fut: Future<Output = Result<E, String>> + Send,
    E: Into<Evaluation>,
{
    /// Create the tool over `runtime` with the host's `evaluate` closure.
    ///
    /// `description` is what the model reads about the tool: say what the
    /// evaluation measures and how to read the report, so the model can
    /// compare two reports. The tool adds how to address a revision.
    ///
    /// The closure returns the report as text, or an [`Evaluation`] that
    /// also carries a score.
    pub fn new(runtime: &'static Runtime, description: impl Into<String>, evaluate: F) -> Self {
        Self {
            runtime,
            description: description.into(),
            evaluate,
            patience: None,
        }
    }

    /// End a lane's search for variants once `patience` scored revisions it
    /// built in a row did not beat its best one: the tool says so, and the
    /// build tools refuse every further candidate, so the agent submits the
    /// best instead of tuning on.
    ///
    /// The v0.34 sliding traces are the reason: a fifth of a lane's model
    /// time went into revisions built after the one it finally submitted,
    /// every one of them worse. Only scored evaluations count, so a closure
    /// that returns plain text never stops a lane.
    #[must_use]
    pub fn with_stopping_rule(mut self, patience: usize) -> Self {
        self.patience = Some(patience.max(1));
        self
    }

    /// The highest registered revision. The registry is never empty: the
    /// initial build is revision `0`.
    fn latest(&self) -> Revision {
        Revision::new(self.runtime.revision_count().saturating_sub(1))
    }
}

/// The arguments of [`EvaluateRevisionTool`].
#[derive(Debug, Deserialize)]
pub struct EvaluateRevisionArgs {
    /// The revision to evaluate. Omit to evaluate the active revision.
    #[serde(default)]
    revision: Option<Revision>,
}

impl EvaluateRevisionArgs {
    /// Arguments for `revision`; `None` evaluates the active revision.
    pub fn new(revision: Option<Revision>) -> Self {
        Self { revision }
    }
}

impl<F, Fut, E> PortableTool for EvaluateRevisionTool<F>
where
    F: Fn(Revision) -> Fut + Send + Sync,
    Fut: Future<Output = Result<E, String>> + Send,
    E: Into<Evaluation>,
{
    const NAME: &'static str = "evaluate_revision";
    type Args = EvaluateRevisionArgs;
    type Output = String;
    type Error = EvaluateRevisionError;

    fn description(&self) -> String {
        let mut out = self.description.clone();
        if !out.ends_with(['.', '!', '?']) {
            out.push('.');
        }
        out.push_str(
            " Pass the number a build tool answered with; omit `revision` to evaluate the active \
             revision, the one your candidates compete with. The first line of the answer names \
             the revision; the report follows.",
        );
        if let Some(patience) = self.patience {
            write!(
                out,
                " A scored report ends with the leaderboard of this task. After {patience} \
                 revisions you built in a row score no better than your best one, the search \
                 stops: the build tools refuse further candidates and you submit the best."
            )
            .expect(EXPECT_WRITE);
        }
        out
    }

    fn map_error(&self, error: Self::Error) -> ToolExecutionError {
        model_visible(error)
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "revision": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "The revision to evaluate. Omit to evaluate the active revision."
                }
            }
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let active = self.runtime.active_revision();
        let revision = args.revision.unwrap_or(active);
        if self.runtime.revision_code(revision).is_none() {
            return Err(EvaluateRevisionError::UnknownRevision {
                requested: revision,
                latest: self.latest(),
            });
        }
        let evaluation: Evaluation = (self.evaluate)(revision)
            .await
            .map_err(|reason| EvaluateRevisionError::Failed { revision, reason })?
            .into();
        let mut out = render(revision, active, &evaluation.report);
        // The leaderboard is the lane's: outside an evolution there is none.
        if let (Some(score), Some(ctx)) = (evaluation.score, ToolContext::current()) {
            ctx.record_score(revision, score, self.patience)
                .render(&mut out);
        }
        Ok(out)
    }
}

/// The answer: one header line, then the host's report.
fn render(revision: Revision, active: Revision, report: &str) -> String {
    let standing = if revision == active { " (active)" } else { "" };
    let mut out = String::with_capacity(report.len() + 32);
    writeln!(out, "Revision {revision}{standing}:").expect(EXPECT_WRITE);
    out.push_str(report);
    if !report.ends_with('\n') {
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use rig_core::tool::ToolErrorKind;

    use super::*;

    #[test]
    fn render_names_the_revision_and_its_standing() {
        assert_eq!(
            render(Revision::new(3), Revision::new(3), "score 0.5"),
            "Revision 3 (active):\nscore 0.5\n"
        );
        assert_eq!(
            render(Revision::new(4), Revision::new(3), "score 0.4\n"),
            "Revision 4:\nscore 0.4\n"
        );
    }

    #[test]
    fn errors_stay_visible_to_the_model() {
        let mapped = model_visible(EvaluateRevisionError::UnknownRevision {
            requested: Revision::new(9),
            latest: Revision::new(2),
        });
        assert_eq!(mapped.kind(), ToolErrorKind::NotFound);
        assert!(mapped.message().contains("0..=2"), "{}", mapped.message());
        let mapped = model_visible(EvaluateRevisionError::Failed {
            revision: Revision::new(1),
            reason: "the benchmark panicked".to_string(),
        });
        assert_eq!(mapped.kind(), ToolErrorKind::Other);
        assert_eq!(mapped.retryable(), Some(false));
        assert!(
            mapped.message().contains("the benchmark panicked"),
            "{}",
            mapped.message()
        );
    }
}

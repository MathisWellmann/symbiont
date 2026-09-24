// SPDX-License-Identifier: MPL-2.0
//! The per-lane state behind the revision tools.
//!
//! The tools are `Copy` handles over the process-wide [`crate::Runtime`] and
//! one agent serves every lane of a batch, so nothing lane-specific can live
//! in a tool. The lane's state lives here instead, installed as a task-local
//! around every agent run by [`crate::Runtime::evolve_lane`]. Rig drives the
//! tool-calling loop inline in the run's future, so a tool call finds the
//! context of the lane that issued it. The same property already carries the
//! inference gate scope.
//!
//! The context holds what a tool call needs to know about the lane: the
//! candidate an edit refers to, the revisions built so far, the build budget,
//! the verdicts of rejected candidates, and the revision the agent chose.
//! The ladder reads the last two after the run and drains the build records
//! into the attempt's trace.

use std::{
    collections::HashMap,
    fmt::Write,
    sync::{
        Arc,
        Mutex,
        MutexGuard,
    },
};

use rig_core::tool::ToolExecutionError;

use crate::{
    EXPECT_WRITE,
    Revision,
    ToolBuild,
    edit::EditBase,
};

tokio::task_local! {
    /// The context of the lane whose agent run this task belongs to.
    static TOOL_CONTEXT: ToolContext;
}

/// Why a revision tool refused a call. The text is what the model reads.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RevisionToolError {
    /// The tool was called on a task with no evolution running: outside
    /// `Runtime::evolve`, or from a task the agent spawned.
    #[error(
        "this tool only works inside an evolution run; no lane is attached to the current task"
    )]
    OutsideEvolve,
    /// The lane spent its build budget.
    #[error(
        "build budget exhausted: {used} of {max} candidates built in this lane. Do not build \
         another one. Revisions built here: {}. Submit the best of them with `submit_revision`, \
         or reply with the complete code block.",
        revision_list(built)
    )]
    BudgetExhausted {
        /// Candidates built so far, the budget included.
        used: usize,
        /// The budget.
        max: usize,
        /// The revisions the lane registered through the tools.
        built: Vec<Revision>,
    },
    /// The agent chose a revision this lane did not build.
    #[error(
        "revision {requested} was not built in this lane; you can only submit one of the \
         revisions you built here: {}",
        revision_list(built)
    )]
    NotBuiltHere {
        /// The revision the agent asked for.
        requested: Revision,
        /// The revisions the lane registered through the tools.
        built: Vec<Revision>,
    },
    /// The agent asked to edit a revision that is not registered.
    #[error(
        "revision {requested} is not registered; registered revisions: 0..={latest}. Omit `base` \
         to edit the candidate you last built or had rejected."
    )]
    UnknownRevision {
        /// The revision the agent asked for.
        requested: Revision,
        /// The highest registered revision.
        latest: Revision,
    },
    /// The lane registered a revision already, and the agent sent a complete
    /// candidate instead of a change to it.
    #[error(
        "send a change, not a complete candidate: revisions {} are registered in this lane, so \
         change one of them with `edit_revision` (SEARCH/REPLACE hunks for the lines that \
         change, or a replacement function) and pass `base` = {} to edit the latest. Retyping \
         the whole program costs minutes of generation for a change of a few lines. Nothing was \
         built and no build of the budget was spent.",
        revision_list(built),
        built.last().map_or_else(|| "N".to_string(), ToString::to_string)
    )]
    EditRequired {
        /// The revisions the lane registered through the tools.
        built: Vec<Revision>,
    },
    /// The lane's evaluations stopped improving: the stopping rule ended
    /// the search for variants.
    #[error(
        "the stopping rule ended this search: the last {patience} revisions you evaluated did \
         not beat revision {best} (score {score}). Do not build another variant. Submit \
         revision {best} with `submit_revision` and end your reply with a short summary."
    )]
    Stopped {
        /// The best revision the lane built and evaluated.
        best: Revision,
        /// Its score, as the leaderboard renders it.
        score: String,
        /// Evaluations without improvement that ended the search.
        patience: usize,
    },
    /// The agent asked to edit the last candidate, and there is none yet.
    #[error(
        "nothing to edit: no candidate was built or rejected in this lane yet. Send a complete \
         candidate to `build_revision` first, or pass `base` to edit a registered revision."
    )]
    NoEditBase,
    /// The harness itself failed (IO, dylib load): nothing the agent can
    /// repair.
    #[error("the harness failed: {0}")]
    Harness(String),
}

impl RevisionToolError {
    /// Make the text of the error visible to the model. The default of
    /// `PortableTool::map_error` hides it; see [`crate::tools`].
    pub(crate) fn model_visible(self) -> ToolExecutionError {
        let text = self.to_string();
        match self {
            Self::OutsideEvolve
            | Self::BudgetExhausted { .. }
            | Self::EditRequired { .. }
            | Self::Stopped { .. } => {
                ToolExecutionError::permission_denied(text).with_retryable(false)
            }
            Self::NotBuiltHere { .. } | Self::UnknownRevision { .. } | Self::NoEditBase => {
                ToolExecutionError::not_found(text).with_retryable(false)
            }
            Self::Harness(_) => ToolExecutionError::other(text).with_retryable(false),
        }
    }
}

/// `1, 4, 7`, or `none`.
pub(crate) fn revision_list(revisions: &[Revision]) -> String {
    if revisions.is_empty() {
        return "none".to_string();
    }
    let mut out = String::new();
    for (i, revision) in revisions.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        write!(out, "{revision}").expect(EXPECT_WRITE);
    }
    out
}

/// The lane state the revision tools read and write. Cheap to clone: a
/// handle to shared state.
///
/// No method holds the lock across an `await`, and a tool call locks only
/// to read or record: two tool calls of one turn that rig runs concurrently
/// do their builds side by side and interleave only their bookkeeping.
#[derive(Debug, Clone)]
pub(crate) struct ToolContext {
    inner: Arc<Mutex<Inner>>,
}

#[derive(Debug)]
struct Inner {
    /// The candidate an edit refers to: the most recent one a response or a
    /// tool build had rejected by the compiler, or registered. `None` on the
    /// first attempt and after a reset, when the agent no longer sees the
    /// code the base would refer to.
    edit_base: Option<EditBase>,
    /// The revisions the tools registered in this lane, in build order.
    built: Vec<Revision>,
    /// The revision the agent chose with `submit_revision`, until the
    /// ladder takes it.
    submitted: Option<Revision>,
    /// Builds the tools spent, against `max_builds`.
    builds_used: usize,
    /// The build budget of the lane.
    max_builds: usize,
    /// Candidates the compiler rejected in this lane, by source, with the
    /// verdict the agent read. A resent candidate gets its verdict back
    /// without a build.
    rejected: HashMap<String, String>,
    /// The first score each evaluated revision got, in evaluation order.
    scores: Vec<Scored>,
    /// Evaluations of revisions built here, since the best one, that did not
    /// beat it.
    stale: usize,
    /// Set once `stale` reached the patience of the evaluating tool: the
    /// lane builds nothing more.
    stopped: Option<Scored>,
    /// Build records since the ladder last drained them.
    builds: Vec<ToolBuild>,
}

impl Inner {
    /// The refusal of every build once the stopping rule ended the search.
    fn stopped_refusal(&self) -> Option<RevisionToolError> {
        self.stopped.map(|best| RevisionToolError::Stopped {
            best: best.revision,
            score: render_score(best.score),
            patience: self.stale,
        })
    }
}

impl ToolContext {
    /// A fresh context for a lane that may spend `max_builds` tool builds.
    pub(crate) fn new(max_builds: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                edit_base: None,
                built: Vec::new(),
                submitted: None,
                builds_used: 0,
                max_builds,
                rejected: HashMap::new(),
                scores: Vec::new(),
                stale: 0,
                stopped: None,
                builds: Vec::new(),
            })),
        }
    }

    /// Run `fut` with this context installed for the tools it calls.
    pub(crate) async fn scope<F: Future>(&self, fut: F) -> F::Output {
        TOOL_CONTEXT.scope(self.clone(), fut).await
    }

    /// The context of the lane running on the current task, if any.
    pub(crate) fn current() -> Option<Self> {
        TOOL_CONTEXT.try_with(Self::clone).ok()
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The candidate an edit refers to.
    pub(crate) fn edit_base(&self) -> Option<EditBase> {
        self.lock().edit_base.clone()
    }

    /// Set, or with `None` forget, the candidate an edit refers to.
    pub(crate) fn set_edit_base(&self, base: Option<EditBase>) {
        self.lock().edit_base = base;
    }

    /// The revisions the tools registered in this lane, in build order.
    pub(crate) fn built(&self) -> Vec<Revision> {
        self.lock().built.clone()
    }

    /// Note a revision the tools registered. A revision the lane already
    /// built (a candidate deduplicated onto it) is listed once.
    pub(crate) fn push_built(&self, revision: Revision) {
        let mut inner = self.lock();
        if !inner.built.contains(&revision) {
            inner.built.push(revision);
        }
    }

    /// Refuse a complete candidate once the lane registered a revision: from
    /// then on the agent changes what it has instead of retyping it.
    ///
    /// The v0.34 traces of the sliding harness show why: successive full
    /// candidates were a median 97% identical to the one before, and those
    /// re-emissions were two thirds of all output tokens a lane generated.
    ///
    /// A lane whose stopping rule ended the search gets
    /// [`RevisionToolError::Stopped`] instead, the refusal every build tool
    /// gives it: a stopped lane always has a registered revision, and telling
    /// it to edit one would contradict the order to submit the best.
    pub(crate) fn require_edit(&self) -> Result<(), RevisionToolError> {
        let inner = self.lock();
        if let Some(refusal) = inner.stopped_refusal() {
            return Err(refusal);
        }
        if inner.built.is_empty() {
            return Ok(());
        }
        Err(RevisionToolError::EditRequired {
            built: inner.built.clone(),
        })
    }

    /// Put the first score of `revision` on the lane's leaderboard and
    /// return where the lane stands.
    ///
    /// Only a revision's first evaluation counts: scoring the same code again
    /// tells the search nothing new. A revision the lane did not build (the
    /// active one, evaluated for reference) is listed but neither becomes the
    /// best nor counts against the patience, since only a revision built here
    /// can be submitted. After `patience` evaluations of revisions built here
    /// in a row that do not beat the best one, the lane stops: every further
    /// build is refused with [`RevisionToolError::Stopped`]. `None` never
    /// stops.
    pub(crate) fn record_score(
        &self,
        revision: Revision,
        score: f64,
        patience: Option<usize>,
    ) -> Standing {
        let mut inner = self.lock();
        let first = !inner.scores.iter().any(|s| s.revision == revision);
        let built_here = inner.built.contains(&revision);
        if first && score.is_finite() {
            inner.scores.push(Scored {
                revision,
                score,
                built_here,
            });
            if built_here && inner.stopped.is_none() {
                if best_of(&inner.scores).map(|best| best.revision) == Some(revision) {
                    inner.stale = 0;
                } else {
                    inner.stale += 1;
                }
                if patience.is_some_and(|patience| inner.stale >= patience) {
                    inner.stopped = best_of(&inner.scores);
                }
            }
        }
        let mut leaderboard = inner.scores.clone();
        leaderboard.sort_by(|a, b| b.score.total_cmp(&a.score));
        Standing {
            leaderboard,
            best: best_of(&inner.scores),
            stale: inner.stale,
            patience,
            stopped: inner.stopped.is_some(),
        }
    }

    /// Spend one build of the budget, or report it exhausted.
    pub(crate) fn reserve_build(&self) -> Result<BuildBudget, RevisionToolError> {
        let mut inner = self.lock();
        if let Some(refusal) = inner.stopped_refusal() {
            return Err(refusal);
        }
        if inner.builds_used >= inner.max_builds {
            return Err(RevisionToolError::BudgetExhausted {
                used: inner.builds_used,
                max: inner.max_builds,
                built: inner.built.clone(),
            });
        }
        inner.builds_used += 1;
        Ok(BuildBudget {
            used: inner.builds_used,
            max: inner.max_builds,
        })
    }

    /// Hand back a reserved build that ran no compile: the candidate turned
    /// out byte-identical to a registered revision. Returns where the budget
    /// stands after the refund.
    pub(crate) fn refund_build(&self) -> BuildBudget {
        let mut inner = self.lock();
        inner.builds_used = inner.builds_used.saturating_sub(1);
        BuildBudget {
            used: inner.builds_used,
            max: inner.max_builds,
        }
    }

    /// The verdict the compiler already gave `source` in this lane, if any.
    pub(crate) fn rejected_verdict(&self, source: &str) -> Option<String> {
        self.lock().rejected.get(source).cloned()
    }

    /// Remember the compiler's verdict on `source`.
    pub(crate) fn remember_rejected(&self, source: String, verdict: String) {
        self.lock().rejected.insert(source, verdict);
    }

    /// Record the agent's choice. Only a revision this lane built through
    /// the tools qualifies: submitting the active revision would be no
    /// evolution, which a response cannot get away with either.
    pub(crate) fn submit(&self, revision: Revision) -> Result<(), RevisionToolError> {
        let mut inner = self.lock();
        if !inner.built.contains(&revision) {
            return Err(RevisionToolError::NotBuiltHere {
                requested: revision,
                built: inner.built.clone(),
            });
        }
        inner.submitted = Some(revision);
        Ok(())
    }

    /// The revision the agent chose during the run that just ended, if any.
    pub(crate) fn take_submitted(&self) -> Option<Revision> {
        self.lock().submitted.take()
    }

    /// Record one tool build for the trace.
    pub(crate) fn record(&self, build: ToolBuild) {
        self.lock().builds.push(build);
    }

    /// The tool builds since the last drain, for the attempt's trace.
    pub(crate) fn take_builds(&self) -> Vec<ToolBuild> {
        std::mem::take(&mut self.lock().builds)
    }
}

/// Where the lane's build budget stands after a reservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BuildBudget {
    /// Builds spent, this one included.
    used: usize,
    /// The budget.
    max: usize,
}

impl BuildBudget {
    /// Builds spent, this one included.
    pub(crate) fn used(self) -> usize {
        self.used
    }

    /// The budget.
    pub(crate) fn max(self) -> usize {
        self.max
    }
}

/// One revision's first score on the lane's leaderboard.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Scored {
    /// The evaluated revision.
    revision: Revision,
    /// The host's score; higher is better.
    score: f64,
    /// The lane built it, so it can be submitted and competes for the best.
    built_here: bool,
}

/// The best-scoring revision built in the lane; the earlier one on a tie.
fn best_of(scores: &[Scored]) -> Option<Scored> {
    scores
        .iter()
        .filter(|scored| scored.built_here)
        .fold(None, |best: Option<Scored>, scored| match best {
            Some(best) if best.score >= scored.score => Some(best),
            _ => Some(*scored),
        })
}

/// A score as the leaderboard and the refusal render it.
pub(crate) fn render_score(score: f64) -> String {
    format!("{score:.4}")
}

/// Where a lane stands after an evaluation: the leaderboard, the best
/// revision built here, and how close the stopping rule is.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Standing {
    /// Every scored revision, the best first.
    leaderboard: Vec<Scored>,
    /// The best revision built in the lane.
    best: Option<Scored>,
    /// Evaluations since the best that did not beat it.
    stale: usize,
    /// Evaluations without improvement that stop the lane; `None` never.
    patience: Option<usize>,
    /// The stopping rule fired: the lane builds nothing more.
    stopped: bool,
}

impl Standing {
    /// Revisions the leaderboard lists; the rest are summarized.
    const SHOWN: usize = 8;

    /// The leaderboard block of the evaluation answer.
    pub(crate) fn render(&self, out: &mut String) {
        writeln!(out, "\nLeaderboard of this task (score, higher is better):").expect(EXPECT_WRITE);
        let best = self.best.map(|best| best.revision);
        for (rank, scored) in self.leaderboard.iter().take(Self::SHOWN).enumerate() {
            let marker = if Some(scored.revision) == best {
                "  <- best"
            } else if scored.built_here {
                ""
            } else {
                "  (not built here; reference only)"
            };
            writeln!(
                out,
                "  {}. revision {}: {}{marker}",
                rank + 1,
                scored.revision,
                render_score(scored.score)
            )
            .expect(EXPECT_WRITE);
        }
        if self.leaderboard.len() > Self::SHOWN {
            writeln!(out, "  ... {} more", self.leaderboard.len() - Self::SHOWN)
                .expect(EXPECT_WRITE);
        }
        match (self.stopped, best, self.patience) {
            (true, Some(best), _) => writeln!(
                out,
                "Stopping rule: {} evaluations in a row did not beat revision {best}. The search \
                 is over and further builds are refused: submit revision {best} with \
                 `submit_revision` now.",
                self.stale
            )
            .expect(EXPECT_WRITE),
            (false, Some(best), Some(patience)) if self.stale > 0 => writeln!(
                out,
                "{} of your last evaluations did not beat revision {best}; after {patience} in a \
                 row the search stops and you submit the best. Change the idea, not only a \
                 constant, or submit revision {best} now.",
                self.stale
            )
            .expect(EXPECT_WRITE),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use rig_core::tool::ToolErrorKind;

    use super::*;

    #[test]
    fn revision_list_names_every_revision_or_none() {
        assert_eq!(revision_list(&[]), "none");
        assert_eq!(revision_list(&[Revision::new(1), Revision::new(4)]), "1, 4");
    }

    #[test]
    fn the_budget_runs_out_after_max_builds() {
        let ctx = ToolContext::new(2);
        assert_eq!(ctx.reserve_build(), Ok(BuildBudget { used: 1, max: 2 }));
        assert_eq!(ctx.reserve_build(), Ok(BuildBudget { used: 2, max: 2 }));
        ctx.push_built(Revision::new(3));
        let err = ctx.reserve_build().expect_err("the budget is spent");
        assert_eq!(
            err,
            RevisionToolError::BudgetExhausted {
                used: 2,
                max: 2,
                built: vec![Revision::new(3)],
            }
        );
        assert!(err.to_string().contains("2 of 2"), "{err}");
        assert!(err.to_string().contains("Revisions built here: 3"), "{err}");
    }

    #[test]
    fn a_refunded_build_frees_the_budget_again() {
        let ctx = ToolContext::new(1);
        ctx.reserve_build().expect("the one build");
        assert!(ctx.reserve_build().is_err(), "spent");
        assert_eq!(ctx.refund_build().used(), 0);
        assert_eq!(ctx.reserve_build().map(BuildBudget::used), Ok(1));
        // A refund never goes below zero.
        ctx.refund_build();
        assert_eq!(ctx.refund_build().used(), 0);
    }

    #[test]
    fn a_revision_built_twice_is_listed_once() {
        let ctx = ToolContext::new(1);
        ctx.push_built(Revision::new(5));
        ctx.push_built(Revision::new(5));
        assert_eq!(ctx.built(), vec![Revision::new(5)]);
    }

    /// A lane with revisions 1..=n built.
    fn lane_with(built: u64) -> ToolContext {
        let ctx = ToolContext::new(10);
        for revision in 1..=built {
            ctx.push_built(Revision::new(revision));
        }
        ctx
    }

    /// The best revision built here leads; each evaluation that does not beat
    /// it counts towards the patience, a new best resets the count, and at the
    /// patience the lane stops building and names the best to submit.
    #[test]
    fn the_stopping_rule_fires_after_patience_evaluations_without_a_new_best() {
        let ctx = lane_with(5);
        let patience = Some(2);
        let s = ctx.record_score(Revision::new(1), 0.05, patience);
        assert_eq!(s.best.map(|b| b.revision), Some(Revision::new(1)));
        assert_eq!((s.stale, s.stopped), (0, false));
        let s = ctx.record_score(Revision::new(2), 0.03, patience);
        assert_eq!((s.stale, s.stopped), (1, false));
        // A new best resets the count.
        let s = ctx.record_score(Revision::new(3), 0.08, patience);
        assert_eq!(s.best.map(|b| b.revision), Some(Revision::new(3)));
        assert_eq!(s.stale, 0);
        // Re-evaluating a revision tells nothing new and does not count.
        let s = ctx.record_score(Revision::new(2), 0.03, patience);
        assert_eq!(s.stale, 0);
        ctx.record_score(Revision::new(4), 0.08, patience); // a tie is no improvement
        assert!(ctx.reserve_build().is_ok(), "one short of the patience");
        let s = ctx.record_score(Revision::new(5), 0.01, patience);
        assert!(s.stopped);
        assert_eq!(
            s.leaderboard
                .iter()
                .map(|x| x.revision.as_u64())
                .collect::<Vec<_>>(),
            vec![3, 4, 1, 2, 5],
            "the leaderboard ranks by score"
        );
        let err = ctx.reserve_build().expect_err("the search is over");
        assert!(
            matches!(&err, RevisionToolError::Stopped { best, .. } if *best == Revision::new(3)),
            "{err:?}"
        );
        assert!(err.to_string().contains("Submit revision 3"), "{err}");
        // A complete candidate gets the same refusal, not the order to edit
        // a revision.
        assert_eq!(ctx.require_edit(), Err(err));
        let mut out = String::new();
        s.render(&mut out);
        assert!(out.contains("1. revision 3: 0.0800  <- best"), "{out}");
        assert!(out.contains("Stopping rule: 2 evaluations"), "{out}");
    }

    /// A revision the lane did not build (the active one, evaluated for
    /// reference) is listed but never becomes the best to submit, and does
    /// not count against the patience; without a patience nothing stops.
    #[test]
    fn reference_revisions_and_no_patience_never_stop_a_lane() {
        let ctx = lane_with(1);
        let s = ctx.record_score(Revision::new(0), 0.5, Some(1));
        assert_eq!(s.best, None, "revision 0 was not built here");
        assert!(!s.stopped);
        let s = ctx.record_score(Revision::new(1), 0.1, Some(1));
        assert_eq!(s.best.map(|b| b.revision), Some(Revision::new(1)));
        let mut out = String::new();
        s.render(&mut out);
        assert!(
            out.contains("revision 0: 0.5000  (not built here; reference only)"),
            "{out}"
        );

        let ctx = lane_with(3);
        for (revision, score) in [(1, 0.3), (2, 0.2), (3, 0.1)] {
            assert!(
                !ctx.record_score(Revision::new(revision), score, None)
                    .stopped
            );
        }
        assert!(ctx.reserve_build().is_ok());
    }

    /// A complete candidate is welcome until the lane registers a revision,
    /// and refused after, naming the revision to edit.
    #[test]
    fn complete_candidates_stop_after_the_first_registration() {
        let ctx = ToolContext::new(3);
        assert_eq!(ctx.require_edit(), Ok(()));
        ctx.push_built(Revision::new(4));
        ctx.push_built(Revision::new(7));
        let err = ctx.require_edit().expect_err("a revision is registered");
        assert_eq!(
            err,
            RevisionToolError::EditRequired {
                built: vec![Revision::new(4), Revision::new(7)]
            }
        );
        let text = err.to_string();
        assert!(text.contains("revisions 4, 7 are registered"), "{text}");
        assert!(text.contains("`base` = 7"), "{text}");
        let mapped = err.model_visible();
        assert_eq!(mapped.kind(), ToolErrorKind::PermissionDenied);
        assert!(
            mapped.message().contains("edit_revision"),
            "the model reads it"
        );
    }

    #[test]
    fn only_a_revision_built_here_can_be_submitted() {
        let ctx = ToolContext::new(1);
        assert_eq!(
            ctx.submit(Revision::new(5)),
            Err(RevisionToolError::NotBuiltHere {
                requested: Revision::new(5),
                built: Vec::new(),
            })
        );
        assert_eq!(ctx.take_submitted(), None);
        ctx.push_built(Revision::new(5));
        ctx.submit(Revision::new(5)).expect("built here");
        assert_eq!(ctx.take_submitted(), Some(Revision::new(5)));
        assert_eq!(ctx.take_submitted(), None, "taken once");
    }

    #[test]
    fn a_rejected_candidate_is_remembered_by_source() {
        let ctx = ToolContext::new(1);
        assert_eq!(ctx.rejected_verdict("fn f() {}"), None);
        ctx.remember_rejected("fn f() {}".to_string(), "E1".to_string());
        assert_eq!(ctx.rejected_verdict("fn f() {}"), Some("E1".to_string()));
        assert_eq!(ctx.rejected_verdict("fn g() {}"), None);
    }

    #[tokio::test]
    async fn the_context_is_visible_inside_its_scope_only() {
        assert!(ToolContext::current().is_none());
        let ctx = ToolContext::new(1);
        ctx.scope(async {
            let seen = ToolContext::current().expect("inside the scope");
            seen.push_built(Revision::new(2));
        })
        .await;
        assert_eq!(ctx.built(), vec![Revision::new(2)], "one shared state");
        assert!(ToolContext::current().is_none());
    }

    #[test]
    fn errors_stay_visible_to_the_model() {
        let err = RevisionToolError::OutsideEvolve.model_visible();
        assert_eq!(err.kind(), ToolErrorKind::PermissionDenied);
        assert!(err.message().contains("no lane"), "{}", err.message());
        assert_eq!(err.retryable(), Some(false));
        let err = RevisionToolError::NotBuiltHere {
            requested: Revision::new(9),
            built: vec![Revision::new(1)],
        }
        .model_visible();
        assert_eq!(err.kind(), ToolErrorKind::NotFound);
        assert!(err.message().contains("built here: 1"), "{}", err.message());
    }
}

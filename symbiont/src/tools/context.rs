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
            Self::OutsideEvolve | Self::BudgetExhausted { .. } => {
                ToolExecutionError::permission_denied(text).with_retryable(false)
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
    /// Builds the tools spent, against `max_builds`.
    builds_used: usize,
    /// The build budget of the lane.
    max_builds: usize,
    /// Candidates the compiler rejected in this lane, by source, with the
    /// verdict the agent read. A resent candidate gets its verdict back
    /// without a build.
    rejected: HashMap<String, String>,
    /// Build records since the ladder last drained them.
    builds: Vec<ToolBuild>,
}

impl ToolContext {
    /// A fresh context for a lane that may spend `max_builds` tool builds.
    pub(crate) fn new(max_builds: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                edit_base: None,
                built: Vec::new(),
                builds_used: 0,
                max_builds,
                rejected: HashMap::new(),
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

    /// Spend one build of the budget, or report it exhausted.
    pub(crate) fn reserve_build(&self) -> Result<BuildBudget, RevisionToolError> {
        let mut inner = self.lock();
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
    }
}

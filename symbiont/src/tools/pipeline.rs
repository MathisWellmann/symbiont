// SPDX-License-Identifier: MPL-2.0
//! The pipeline pass behind `build_revision` and `edit_revision`: the same
//! parse, validate and build a response goes through, with the verdict
//! rendered as the tool's answer.

use std::fmt::Write;
// Under Miri the timing falls back to `std::time::Instant`; see `lib.rs`.
#[cfg(miri)]
use std::time::Instant;

use metrics::counter;
#[cfg(not(miri))]
use minstant::Instant;
use tracing::info;

use crate::{
    BuildRecord,
    EXPECT_WRITE,
    Revision,
    Runtime,
    StageTimings,
    ToolBuild,
    ToolBuildOutcome,
    diagnostics::render_fixes_for_prompt,
    edit::EditBase,
    error::{
        Error,
        Result,
    },
    observability::{
        TOOL_BUILDS,
        failure_kind_of,
    },
    parser::Candidate,
    tools::context::{
        BuildBudget,
        RevisionToolError,
        ToolContext,
        revision_list,
    },
};

/// What the pass ended with, before the verdict is rendered.
enum Step {
    /// Registered under this revision, freshly built or deduplicated.
    Registered(Revision, BuildBudget),
    /// The pipeline rejected the candidate.
    Rejected(Error),
    /// The compiler rejected this exact source earlier in the lane.
    Repeated(String),
}

/// Parse, validate and build the candidate `parse` yields, inside the lane
/// of the current task. The answer is the text the model reads: the
/// revision it registered, or the nudge a rejected response would get.
///
/// A candidate that parses and validates but whose source the compiler
/// already rejected in this lane gets the earlier verdict back without a
/// build. A candidate that gets to the compiler spends one build of the
/// lane's budget; a parse or validation failure costs microseconds and
/// spends nothing.
///
/// The lane's edit base follows the pass, as it does for a response: a
/// candidate the compiler rejected is the base with its diagnostics, a
/// registered one is the base without any.
pub(super) async fn build_through_tool(
    runtime: &'static Runtime,
    tool: &'static str,
    parse: impl FnOnce(&mut StageTimings) -> Result<Candidate>,
) -> std::result::Result<String, RevisionToolError> {
    let ctx = ToolContext::current().ok_or(RevisionToolError::OutsideEvolve)?;
    let t0 = Instant::now();
    let mut stages = StageTimings::default();

    let (candidate, step) = match runtime.parse_and_validate(&mut stages, parse) {
        Err(e) => (e.candidate().map(str::to_owned), Step::Rejected(e)),
        Ok(source) => {
            let step = match ctx.rejected_verdict(&source) {
                Some(verdict) => Step::Repeated(verdict),
                None => match ctx.reserve_build() {
                    Err(budget) => {
                        record_refused(&ctx, tool, source, stages, t0, &budget.to_string());
                        return Err(budget);
                    }
                    Ok(budget) => match runtime
                        .build_and_register(source.clone(), stages.build_mut())
                        .await
                    {
                        // The reservation was for a compile. A candidate
                        // that is byte-identical to a registered revision
                        // as written ran none, so it costs none; one that
                        // matched only after autofixes did run once.
                        Ok(revision) => match stages.build() {
                            Some(BuildRecord::Deduped { autofixes, .. })
                                if autofixes.is_empty() =>
                            {
                                Step::Registered(revision, ctx.refund_build())
                            }
                            _ => Step::Registered(revision, budget),
                        },
                        Err(e) => Step::Rejected(e),
                    },
                },
            };
            (Some(source), step)
        }
    };

    let (outcome, verdict, harness_failure) = match step {
        Step::Registered(revision, budget) => {
            ctx.push_built(revision);
            // The registered source is the candidate the build accepted,
            // autofixes included: that is the text a later edit refers to.
            if let Some(source) = runtime.revision_code(revision) {
                ctx.set_edit_base(Some(EditBase::new(source, Vec::new())));
            }
            let verdict =
                render_registered(revision, stages.build().as_ref(), &ctx.built(), budget);
            info!("Tool `{tool}` registered revision {revision}.");
            (ToolBuildOutcome::Registered { revision }, verdict, None)
        }
        Step::Rejected(e) => {
            let kind = failure_kind_of(&e).to_string();
            let compile_failure = match &e {
                Error::CompilationFailed {
                    code, diagnostics, ..
                } => {
                    ctx.set_edit_base(Some(EditBase::new(code.clone(), diagnostics.clone())));
                    Some(code.clone())
                }
                _ => None,
            };
            let mut verdict = String::new();
            let harness_failure = match runtime.render_nudge(e, &mut verdict).await {
                Ok(_) => None,
                Err(e) => {
                    verdict = e.to_string();
                    Some(verdict.clone())
                }
            };
            if let Some(source) = compile_failure {
                ctx.remember_rejected(source, verdict.clone());
            }
            info!("Tool `{tool}` rejected the candidate ({kind}).");
            (
                ToolBuildOutcome::Rejected {
                    kind,
                    verdict: verdict.clone(),
                },
                verdict,
                harness_failure,
            )
        }
        Step::Repeated(earlier) => {
            info!(
                "Tool `{tool}` saw a candidate the compiler already rejected; answering from memory."
            );
            (ToolBuildOutcome::Repeated, render_repeated(&earlier), None)
        }
    };

    counter!(TOOL_BUILDS, "tool" => tool, "outcome" => outcome_label(&outcome)).increment(1);
    ctx.record(
        ToolBuild::builder()
            .tool(tool)
            .candidate(candidate)
            .stages(stages)
            .outcome(outcome)
            .duration(t0.elapsed())
            .build(),
    );
    match harness_failure {
        Some(message) => Err(RevisionToolError::Harness(message)),
        None => Ok(verdict),
    }
}

/// Record a call the budget refused, so the trace shows it.
fn record_refused(
    ctx: &ToolContext,
    tool: &'static str,
    candidate: String,
    stages: StageTimings,
    t0: Instant,
    verdict: &str,
) {
    counter!(TOOL_BUILDS, "tool" => tool, "outcome" => "budget").increment(1);
    ctx.record(
        ToolBuild::builder()
            .tool(tool)
            .candidate(Some(candidate))
            .stages(stages)
            .outcome(ToolBuildOutcome::Rejected {
                kind: "budget".to_string(),
                verdict: verdict.to_string(),
            })
            .duration(t0.elapsed())
            .build(),
    );
}

/// The `outcome` label of [`TOOL_BUILDS`].
fn outcome_label(outcome: &ToolBuildOutcome) -> &'static str {
    match outcome {
        ToolBuildOutcome::Registered { .. } => "registered",
        ToolBuildOutcome::Rejected { .. } => "rejected",
        ToolBuildOutcome::Repeated => "repeated",
    }
}

/// The answer for a candidate that built: the revision, what the build did
/// to get there, and where the lane stands.
fn render_registered(
    revision: Revision,
    record: Option<&BuildRecord>,
    built: &[Revision],
    budget: BuildBudget,
) -> String {
    let mut out = String::new();
    match record {
        Some(BuildRecord::Deduped { autofixes, .. }) if autofixes.is_empty() => {
            writeln!(
                out,
                "Your code is byte-identical to revision {revision}, which is already registered; \
                 no build was spent."
            )
            .expect(EXPECT_WRITE);
        }
        Some(BuildRecord::Deduped { autofixes, .. }) => {
            writeln!(
                out,
                "With the compiler suggestions below applied, your code is byte-identical to \
                 revision {revision}, which is already registered; no second build was spent."
            )
            .expect(EXPECT_WRITE);
            render_fixes_for_prompt(autofixes, &mut out);
        }
        Some(BuildRecord::Built { autofixes, .. }) if !autofixes.is_empty() => {
            writeln!(
                out,
                "Registered revision {revision}. It compiled after the harness applied compiler \
                 suggestions, so the registered source differs from your code; `revision_source` \
                 shows it."
            )
            .expect(EXPECT_WRITE);
            render_fixes_for_prompt(autofixes, &mut out);
        }
        _ => {
            writeln!(out, "Registered revision {revision}.").expect(EXPECT_WRITE);
        }
    }
    writeln!(
        out,
        "It is not active: the harness activates the revision you choose with `submit_revision`. \
         Revisions built in this lane: {}. Build budget: {} of {} used.",
        revision_list(built),
        budget.used(),
        budget.max()
    )
    .expect(EXPECT_WRITE);
    out
}

/// The answer for a candidate the compiler already rejected in this lane.
fn render_repeated(earlier: &str) -> String {
    format!(
        "You already sent this exact code and the compiler rejected it. It was not built again. \
         Change the code before you send it again. The verdict was:\n{earlier}"
    )
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::AppliedFix;

    fn budget() -> BuildBudget {
        let ctx = ToolContext::new(10);
        ctx.reserve_build().expect("first build");
        ctx.reserve_build().expect("second build")
    }

    #[test]
    fn a_fresh_build_names_the_revision_and_the_budget() {
        let record = BuildRecord::Built {
            slot_wait: Duration::ZERO,
            compile: Duration::from_secs(3),
            load: Duration::ZERO,
            autofixes: Vec::new(),
        };
        let shown = render_registered(
            Revision::new(4),
            Some(&record),
            &[Revision::new(2), Revision::new(4)],
            budget(),
        );
        assert_eq!(
            shown,
            "Registered revision 4.\n\
             It is not active: the harness activates the revision you choose with \
             `submit_revision`. Revisions built in this lane: 2, 4. Build budget: 2 of 10 used.\n"
        );
    }

    #[test]
    fn a_deduplicated_build_says_so() {
        let record = BuildRecord::Deduped {
            slot_wait: Duration::ZERO,
            revision: Revision::new(2),
            autofixes: Vec::new(),
        };
        let shown = render_registered(
            Revision::new(2),
            Some(&record),
            &[Revision::new(2)],
            budget(),
        );
        assert!(
            shown.starts_with("Your code is byte-identical to revision 2"),
            "{shown}"
        );
    }

    #[test]
    fn an_autofixed_build_lists_the_fixes() {
        let record = BuildRecord::Built {
            slot_wait: Duration::ZERO,
            compile: Duration::ZERO,
            load: Duration::ZERO,
            autofixes: vec![AppliedFix {
                code: Some("E0308".to_string()),
                message: "mismatched types".to_string(),
                line: 1,
                before: "2".to_string(),
                after: "2.0".to_string(),
            }],
        };
        let shown = render_registered(
            Revision::new(3),
            Some(&record),
            &[Revision::new(3)],
            budget(),
        );
        assert!(shown.contains("registered source differs"), "{shown}");
        assert!(
            shown.contains("`2` -> `2.0`") || shown.contains("2.0"),
            "{shown}"
        );
    }

    #[test]
    fn a_repeated_candidate_quotes_the_earlier_verdict() {
        let shown = render_repeated("nudge: E1");
        assert!(shown.ends_with("The verdict was:\nnudge: E1"), "{shown}");
    }
}

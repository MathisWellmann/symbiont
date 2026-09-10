// SPDX-License-Identifier: MPL-2.0
//! The `revision_source` tool: the full source of a registered revision, on
//! demand, for the evolution agent.

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
};

/// The error of a `revision_source` query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum RevisionSourceError {
    /// No revision with the requested number is registered.
    #[error(
        "revision {requested} is not registered; registered revisions: 0..={latest}, active: {active}"
    )]
    UnknownRevision {
        /// The number the agent asked for.
        requested: Revision,
        /// The highest registered revision.
        latest: Revision,
        /// The revision the dispatch pointers currently execute.
        active: Revision,
    },
}

/// Make the text of a query error visible to the model.
///
/// The default of [`PortableTool::map_error`] hides the text of the source
/// and shows the model the string "the tool failed". The explicit constructor
/// keeps the message, so the agent reads which revisions exist instead.
fn model_visible(error: RevisionSourceError) -> ToolExecutionError {
    match error {
        RevisionSourceError::UnknownRevision { .. } => {
            ToolExecutionError::not_found(error.to_string()).with_retryable(false)
        }
    }
}

/// The `revision_source` tool: read the full source of one registered
/// revision, as the harness compiled it.
///
/// The source is the candidate the build accepted, autofixes included,
/// without the prelude and the harness glue: the same text
/// [`Runtime::revision_code`] returns. The tool takes no I/O and runs no
/// build; it reads the revision registry.
///
/// Not registered by default: [`crate::agent_builder`] leaves it out, so a
/// host that does not want the agent to read earlier revisions pays nothing.
/// Register it on the builder with `.tool(RevisionSourceTool::new(runtime))`
/// and give the agent room to call it: rig's default `default_max_turns` of
/// `0` aborts the run at the first tool call.
///
/// Revision numbers are process-wide. In an [`Runtime::evolve_batch`], one
/// lane can therefore read what another lane registered.
#[derive(Clone, Copy)]
pub struct RevisionSourceTool {
    runtime: &'static Runtime,
}

impl RevisionSourceTool {
    /// Create the tool over the runtime whose revisions it reads.
    pub fn new(runtime: &'static Runtime) -> Self {
        Self { runtime }
    }

    /// The highest registered revision. The registry is never empty: the
    /// initial build is revision `0`.
    fn latest(&self) -> Revision {
        Revision::new(self.runtime.revision_count().saturating_sub(1))
    }
}

/// The arguments of [`RevisionSourceTool`].
#[derive(Debug, Deserialize)]
pub struct RevisionSourceArgs {
    /// The revision to read. Omit to read the active revision.
    revision: Option<Revision>,
}

impl PortableTool for RevisionSourceTool {
    const NAME: &'static str = "revision_source";
    type Args = RevisionSourceArgs;
    type Output = String;
    type Error = RevisionSourceError;

    fn description(&self) -> String {
        "Read the full source of a registered revision of the evolvable code, as the harness \
         compiled it: no prelude, no harness glue. \
         Omit `revision` to read the active revision. \
         The first line of the answer names the revision, says whether it is active, and gives \
         the range of registered revisions; the code follows after a blank line. \
         Revision 0 is the initial build from the default bodies."
            .to_string()
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
                    "description": "The revision number, for example `0` for the initial build. Omit to read the active revision."
                }
            }
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let active = self.runtime.active_revision();
        let requested = args.revision.unwrap_or(active);
        let source = self.runtime.revision_code(requested).ok_or_else(|| {
            RevisionSourceError::UnknownRevision {
                requested,
                latest: self.latest(),
                active,
            }
        })?;
        Ok(render(requested, active, self.latest(), &source))
    }
}

/// The answer of the tool: one header line, a blank line, then the source.
///
/// The header names the revision, its standing (`active`, the initial build,
/// both, or neither) and the range of registered revisions, so one answer
/// tells the agent what else it can read.
fn render(revision: Revision, active: Revision, latest: Revision, source: &str) -> String {
    let standing = match (revision == active, revision == Revision::INITIAL) {
        (true, true) => " (active; initial build from the default bodies)",
        (true, false) => " (active)",
        (false, true) => " (initial build from the default bodies)",
        (false, false) => "",
    };
    let mut out = String::with_capacity(source.len() + 96);
    writeln!(
        out,
        "Revision {revision}{standing}. Registered revisions: 0..={latest}.\n"
    )
    .expect(EXPECT_WRITE);
    out.push_str(source);
    if !source.ends_with('\n') {
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use rig_core::tool::ToolErrorKind;

    use super::*;

    #[test]
    fn render_names_the_revision_and_appends_the_source() {
        let shown = render(
            Revision::new(2),
            Revision::new(1),
            Revision::new(3),
            "fn f() {}",
        );
        assert_eq!(
            shown,
            "Revision 2. Registered revisions: 0..=3.\n\nfn f() {}\n"
        );
    }

    #[test]
    fn render_marks_the_active_and_the_initial_revision() {
        let header = |revision: u64, active: u64| {
            render(
                Revision::new(revision),
                Revision::new(active),
                Revision::new(3),
                "fn f() {}\n",
            )
            .lines()
            .next()
            .expect("the header is the first line")
            .to_string()
        };
        assert_eq!(
            header(1, 1),
            "Revision 1 (active). Registered revisions: 0..=3."
        );
        assert_eq!(
            header(0, 1),
            "Revision 0 (initial build from the default bodies). Registered revisions: 0..=3."
        );
        assert_eq!(
            header(0, 0),
            "Revision 0 (active; initial build from the default bodies). Registered revisions: 0..=3."
        );
    }

    /// A source that already ends with a newline gains no second one.
    #[test]
    fn render_keeps_a_single_trailing_newline() {
        let shown = render(
            Revision::new(1),
            Revision::new(1),
            Revision::new(1),
            "fn f() {}\n",
        );
        assert!(shown.ends_with("fn f() {}\n"));
        assert!(!shown.ends_with("\n\n"));
    }

    /// The model must read which revisions exist, not the redacted default
    /// of `ToolExecutionError::from_error`.
    #[test]
    fn unknown_revisions_stay_visible_to_the_model() {
        let mapped = model_visible(RevisionSourceError::UnknownRevision {
            requested: Revision::new(7),
            latest: Revision::new(3),
            active: Revision::new(2),
        });
        let feedback = mapped.model_feedback().expect("the feedback is text");
        assert!(feedback.contains("revision 7"), "{feedback}");
        assert!(feedback.contains("0..=3"), "{feedback}");
        assert!(feedback.contains("active: 2"), "{feedback}");
        assert_eq!(mapped.kind(), ToolErrorKind::NotFound);
        assert_eq!(mapped.retryable(), Some(false));
    }

    /// The arguments arrive as the JSON the model wrote: a bare integer for
    /// the revision, or nothing at all.
    #[test]
    fn arguments_deserialize_from_the_schema() {
        let args: RevisionSourceArgs =
            serde_json::from_value(serde_json::json!({ "revision": 2 })).expect("an integer");
        assert_eq!(args.revision, Some(Revision::new(2)));

        let args: RevisionSourceArgs =
            serde_json::from_value(serde_json::json!({})).expect("no arguments");
        assert_eq!(args.revision, None);

        serde_json::from_value::<RevisionSourceArgs>(serde_json::json!({ "revision": -1 }))
            .expect_err("a negative number is not a revision");
    }
}

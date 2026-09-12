// SPDX-License-Identifier: MPL-2.0
//! The `submit_revision` tool: the agent's choice among the revisions it
//! built.

use rig_core::tool::{
    PortableTool,
    ToolExecutionError,
};
use serde::Deserialize;
use tracing::info;

use crate::{
    Revision,
    tools::context::{
        RevisionToolError,
        ToolContext,
    },
};

/// The `submit_revision` tool: choose the revision the lane ends with.
///
/// A run that built revisions through `build_revision` or `edit_revision`
/// ends by naming one of them. The ladder reads the choice after the run,
/// registers nothing more, and publishes the chosen revision exactly as it
/// would publish one a response's code block produced. A choice takes
/// precedence over any code block in the same reply.
///
/// Only a revision this lane built through the tools can be submitted. The
/// active revision, or one another lane built, is not an evolution of this
/// lane. A run that built revisions but neither submitted one nor answered
/// with code gets a nudge that asks for the choice; a run whose tools were
/// withdrawn can still choose with a single line `revision: N`.
///
/// Register it with [`crate::with_revision_tools`].
#[derive(Clone, Copy, Default)]
pub struct SubmitRevisionTool;

/// The arguments of [`SubmitRevisionTool`].
#[derive(Debug, Deserialize)]
pub struct SubmitRevisionArgs {
    /// The revision to activate.
    revision: Revision,
    /// Why this one, for the reader of the transcript. The harness does
    /// not act on it.
    #[serde(default)]
    rationale: Option<String>,
}

impl SubmitRevisionArgs {
    /// Arguments that choose `revision`.
    pub fn new(revision: Revision) -> Self {
        Self {
            revision,
            rationale: None,
        }
    }
}

impl PortableTool for SubmitRevisionTool {
    const NAME: &'static str = "submit_revision";
    type Args = SubmitRevisionArgs;
    type Output = String;
    type Error = RevisionToolError;

    fn description(&self) -> String {
        "Choose the revision the harness activates, among the revisions you built in this task \
         with `build_revision` or `edit_revision`. Call it once you have compared your candidates. \
         After the call, end your reply with a short summary; no code block is needed. \
         The choice overrides any code block in the same reply."
            .to_string()
    }

    fn map_error(&self, error: Self::Error) -> ToolExecutionError {
        error.model_visible()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "revision": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "The revision number a build tool answered with."
                },
                "rationale": {
                    "type": "string",
                    "description": "Optional: why this revision, in one or two sentences."
                }
            },
            "required": ["revision"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let ctx = ToolContext::current().ok_or(RevisionToolError::OutsideEvolve)?;
        ctx.submit(args.revision)?;
        match &args.rationale {
            Some(rationale) => info!("Agent chose revision {}: {rationale}", args.revision),
            None => info!("Agent chose revision {}.", args.revision),
        }
        Ok(format!(
            "Revision {} is chosen. End your reply now with a short summary; do not include code.",
            args.revision
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn outside_an_evolution_the_tool_refuses() {
        let err = PortableTool::call(
            &SubmitRevisionTool,
            SubmitRevisionArgs::new(Revision::new(1)),
        )
        .await
        .expect_err("no lane on this task");
        assert_eq!(err, RevisionToolError::OutsideEvolve);
    }

    #[tokio::test]
    async fn the_choice_lands_in_the_lane_context() {
        let ctx = ToolContext::new(1);
        ctx.push_built(Revision::new(4));
        let answer = ctx
            .scope(async {
                let err = PortableTool::call(
                    &SubmitRevisionTool,
                    SubmitRevisionArgs::new(Revision::new(2)),
                )
                .await
                .expect_err("revision 2 was not built here");
                assert!(
                    matches!(err, RevisionToolError::NotBuiltHere { .. }),
                    "{err}"
                );
                PortableTool::call(
                    &SubmitRevisionTool,
                    SubmitRevisionArgs::new(Revision::new(4)),
                )
                .await
                .expect("revision 4 was built here")
            })
            .await;
        assert!(answer.starts_with("Revision 4 is chosen."), "{answer}");
        assert_eq!(ctx.take_submitted(), Some(Revision::new(4)));
    }

    #[test]
    fn the_rationale_is_optional_on_the_wire() {
        let args: SubmitRevisionArgs =
            serde_json::from_value(serde_json::json!({ "revision": 3 })).expect("deserializes");
        assert_eq!(args.revision, Revision::new(3));
        assert_eq!(args.rationale, None);
    }
}

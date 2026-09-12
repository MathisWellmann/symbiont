// SPDX-License-Identifier: MPL-2.0
//! The `edit_revision` tool: a change to a candidate the agent already
//! has, through the pipeline.

use rig_core::tool::{
    PortableTool,
    ToolExecutionError,
};
use serde::Deserialize;

use crate::{
    Revision,
    Runtime,
    edit::{
        EditBase,
        fences_of_text,
    },
    parser::parse_candidate,
    tools::{
        context::{
            RevisionToolError,
            ToolContext,
        },
        pipeline::build_through_tool,
    },
};

/// The `edit_revision` tool: apply edits to a candidate the agent already
/// has and run the result through the pipeline, as `build_revision` does
/// with a complete candidate.
///
/// The edits are the ones a repair round may write in a response (see
/// [`crate::edit`]): `E<n> => text` anchors on the errors of the last
/// verdict, `<<<<<<< SEARCH` / `>>>>>>> REPLACE` hunks, and items that
/// replace the base's items of the same name. The base is the candidate
/// the lane most recently built or had rejected by the compiler, through a
/// tool or in a response, or any registered revision by number.
///
/// Anchors need the errors of the last verdict, so they apply to the
/// default base only; a registered revision has no errors to anchor on.
/// Text that carries no edit at all is taken as a complete candidate.
///
/// Register it with [`crate::with_revision_tools`].
#[derive(Clone, Copy)]
pub struct EditRevisionTool {
    runtime: &'static Runtime,
}

impl EditRevisionTool {
    /// Create the tool over the runtime that builds the edited candidates.
    pub fn new(runtime: &'static Runtime) -> Self {
        Self { runtime }
    }

    /// The base the edits apply to: `revision` when given, else the lane's
    /// last candidate.
    fn base(
        &self,
        ctx: &ToolContext,
        revision: Option<Revision>,
    ) -> Result<EditBase, RevisionToolError> {
        match revision {
            Some(revision) => {
                let source = self.runtime.revision_code(revision).ok_or_else(|| {
                    RevisionToolError::UnknownRevision {
                        requested: revision,
                        latest: Revision::new(self.runtime.revision_count().saturating_sub(1)),
                    }
                })?;
                Ok(EditBase::new(source, Vec::new()))
            }
            None => ctx.edit_base().ok_or(RevisionToolError::NoEditBase),
        }
    }
}

/// The arguments of [`EditRevisionTool`].
#[derive(Debug, Deserialize)]
pub struct EditRevisionArgs {
    /// The edits, in the forms a response may use. Plain text or fenced.
    edits: String,
    /// The revision to edit. Omit to edit the candidate the lane most
    /// recently built or had rejected.
    #[serde(default)]
    base: Option<Revision>,
}

impl EditRevisionArgs {
    /// Arguments that edit the lane's last candidate.
    pub fn new(edits: impl Into<String>) -> Self {
        Self {
            edits: edits.into(),
            base: None,
        }
    }

    /// Arguments that edit a registered revision.
    pub fn against(revision: Revision, edits: impl Into<String>) -> Self {
        Self {
            edits: edits.into(),
            base: Some(revision),
        }
    }
}

impl PortableTool for EditRevisionTool {
    const NAME: &'static str = "edit_revision";
    type Args = EditRevisionArgs;
    type Output = String;
    type Error = RevisionToolError;

    fn description(&self) -> String {
        "Change the candidate you last built or had rejected, without retyping it, and build the \
         result as `build_revision` would. Three forms of edit, as in a reply: `E<n> => new code` \
         replaces the text that error n of the last verdict underlines; a `<<<<<<< SEARCH` / \
         `=======` / `>>>>>>> REPLACE` hunk replaces one unique match; a function item replaces \
         the function of the same name. Several edits may share the text. \
         Pass `base` to edit a registered revision instead of the last candidate; error anchors \
         do not apply to a registered revision. \
         The answer is the same as for `build_revision`."
            .to_string()
    }

    fn map_error(&self, error: Self::Error) -> ToolExecutionError {
        error.model_visible()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "edits": {
                    "type": "string",
                    "description": "The edits: `E<n> => code` lines, SEARCH/REPLACE hunks, or replacement function items. Plain text, no fences needed."
                },
                "base": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Optional: the registered revision to edit. Omit to edit the candidate you last built or had rejected."
                }
            },
            "required": ["edits"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let ctx = ToolContext::current().ok_or(RevisionToolError::OutsideEvolve)?;
        let base = self.base(&ctx, args.base)?;
        let runtime = self.runtime;
        build_through_tool(runtime, Self::NAME, move |stages| {
            runtime.edited_candidate(&base, &fences_of_text(&args.edits), stages, || {
                parse_candidate(args.edits.clone())
            })
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_base_is_optional_on_the_wire() {
        let args: EditRevisionArgs =
            serde_json::from_value(serde_json::json!({ "edits": "E1 => 2.0" }))
                .expect("deserializes");
        assert_eq!(args.edits, "E1 => 2.0");
        assert_eq!(args.base, None);
        let args: EditRevisionArgs =
            serde_json::from_value(serde_json::json!({ "edits": "fn f() {}", "base": 3 }))
                .expect("deserializes");
        assert_eq!(args.base, Some(Revision::new(3)));
    }
}

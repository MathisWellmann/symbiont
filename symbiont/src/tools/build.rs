// SPDX-License-Identifier: MPL-2.0
//! The `build_revision` tool: a complete candidate through the pipeline,
//! with the verdict as the answer.

use rig_core::tool::{
    PortableTool,
    ToolExecutionError,
};
use serde::Deserialize;

use crate::{
    Runtime,
    parser::parse_candidate,
    tools::{
        context::RevisionToolError,
        pipeline::build_through_tool,
    },
};

/// The `build_revision` tool: parse, validate, compile and register a
/// complete candidate, and answer with the revision it got or with the
/// nudge a rejected response would get.
///
/// This is the response path as a tool call. The difference is where the
/// verdict lands: a rejected response ends the attempt and the ladder sends
/// the nudge as the next prompt, while a rejected tool build hands the same
/// nudge back as the tool's result, inside the same run. The agent can
/// therefore build, repair and compare several candidates before it answers,
/// and answer by choosing one with `submit_revision`.
///
/// The tool registers but never activates: the revision that becomes active
/// is the one the agent submits, published by the ladder as for a response.
/// A candidate the compiler already rejected in this lane is answered from
/// memory without a build. Each candidate that reaches the compiler spends
/// one build of [`Runtime::MAX_TOOL_BUILDS`].
///
/// The tool works inside an evolution run only: it needs the lane it is
/// called from, which [`Runtime::evolve`] attaches to the run's task.
/// Register it with [`crate::with_revision_tools`], which also adds the
/// companion tools and the prompt section that explains them.
#[derive(Clone, Copy)]
pub struct BuildRevisionTool {
    runtime: &'static Runtime,
}

impl BuildRevisionTool {
    /// Create the tool over the runtime that builds the candidates.
    pub fn new(runtime: &'static Runtime) -> Self {
        Self { runtime }
    }
}

/// The arguments of [`BuildRevisionTool`].
#[derive(Debug, Deserialize)]
pub struct BuildRevisionArgs {
    /// The complete Rust source: every evolvable function, as items, no
    /// markdown fences.
    code: String,
}

impl BuildRevisionArgs {
    /// Arguments for `code`.
    pub fn new(code: impl Into<String>) -> Self {
        Self { code: code.into() }
    }
}

impl PortableTool for BuildRevisionTool {
    const NAME: &'static str = "build_revision";
    type Args = BuildRevisionArgs;
    type Output = String;
    type Error = RevisionToolError;

    fn description(&self) -> String {
        "Compile a complete candidate of the evolvable code and register it as a numbered \
         revision, without activating it. Pass the full Rust source with every required \
         function as an item, no markdown fences. \
         The answer names the revision, or gives the reasons the candidate was rejected: the \
         compiler errors numbered [E1], [E2], ... with line numbers into your code, a signature \
         mismatch, or a forbidden construct. Fix a rejected candidate with `edit_revision`. \
         Code the compiler already rejected here is answered from memory and not built again. \
         Choose the revision to activate with `submit_revision`."
            .to_string()
    }

    fn map_error(&self, error: Self::Error) -> ToolExecutionError {
        error.model_visible()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The complete Rust source: every required function as an item. No markdown fences."
                }
            },
            "required": ["code"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        build_through_tool(self.runtime, Self::NAME, |_| {
            parse_candidate(strip_fences(args.code))
        })
        .await
    }
}

/// The code without a markdown fence around it, if the model added one
/// despite the instructions. The text between the fences is what it meant.
fn strip_fences(code: String) -> String {
    let trimmed = code.trim();
    let Some(rest) = trimmed.strip_prefix("```") else {
        return code;
    };
    let Some(body) = rest.strip_suffix("```") else {
        return code;
    };
    // Drop the info string (`rust`) on the opening line.
    let body = body.split_once('\n').map_or("", |(_, body)| body);
    body.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fences_are_stripped_when_the_model_adds_them() {
        assert_eq!(
            strip_fences("```rust\nfn f() {}\n```".to_string()),
            "fn f() {}"
        );
        assert_eq!(strip_fences("```\nfn f() {}\n```".to_string()), "fn f() {}");
        assert_eq!(strip_fences("fn f() {}".to_string()), "fn f() {}");
        assert_eq!(
            strip_fences("fn f() {}\n```".to_string()),
            "fn f() {}\n```",
            "an unbalanced fence is left alone"
        );
    }
}

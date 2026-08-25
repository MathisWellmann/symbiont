// SPDX-License-Identifier: MPL-2.0
//! The `api_index` and `api_doc` tools: on-demand access to the host API
//! documentation for the evolution agent.

use std::sync::Arc;

use rig_core::tool::PortableTool;
use serde::Deserialize;

use crate::doc_index::{
    DocIndex,
    DocIndexError,
};

/// The `api_index` tool: list the public items of a host API module.
///
/// [`crate::agent_builder`] registers this tool when the
/// [`DocMode`](crate::DocMode) includes tools. Register it by hand on a
/// custom agent with `.tool(ApiIndexTool::new(index))`.
#[derive(Clone)]
pub struct ApiIndexTool {
    index: Arc<DocIndex>,
}

impl ApiIndexTool {
    /// Create the tool from a shared doc index.
    pub fn new(index: Arc<DocIndex>) -> Self {
        Self { index }
    }
}

/// The arguments of [`ApiIndexTool`].
#[derive(Debug, Deserialize)]
pub struct ApiIndexArgs {
    /// The `::`-separated path of the module to list. Omit to list the
    /// prelude.
    module: Option<String>,
}

impl PortableTool for ApiIndexTool {
    const NAME: &'static str = "api_index";
    type Args = ApiIndexArgs;
    type Output = String;
    type Error = DocIndexError;

    fn description(&self) -> String {
        "List the public items of a host API module as `kind name` lines. \
         Call without arguments to list the prelude. \
         Call `api_doc` with a path to get the full definition of an item."
            .to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "module": {
                    "type": "string",
                    "description": "The `::`-separated module path, for example `prelude::indicators`. Omit to list the prelude."
                }
            }
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let listing = self.index.render_index(args.module.as_deref())?;
        if listing.is_empty() {
            return Ok("The module exposes no public items.".to_string());
        }
        Ok(listing)
    }
}

/// The `api_doc` tool: get the full definition of one host API item or
/// module.
///
/// [`crate::agent_builder`] registers this tool when the
/// [`DocMode`](crate::DocMode) includes tools. Register it by hand on a
/// custom agent with `.tool(ApiDocTool::new(index))`.
#[derive(Clone)]
pub struct ApiDocTool {
    index: Arc<DocIndex>,
}

impl ApiDocTool {
    /// Create the tool from a shared doc index.
    pub fn new(index: Arc<DocIndex>) -> Self {
        Self { index }
    }
}

/// The arguments of [`ApiDocTool`].
#[derive(Debug, Deserialize)]
pub struct ApiDocArgs {
    /// The `::`-separated path of the item or module.
    path: String,
}

impl PortableTool for ApiDocTool {
    const NAME: &'static str = "api_doc";
    type Args = ApiDocArgs;
    type Output = String;
    type Error = DocIndexError;

    fn description(&self) -> String {
        "Get the full definition of a host API item or module: the declaration, \
         the inherent methods, and the operator impls. \
         Pass a `::`-separated path like `prelude::Order` or a single name from the prelude."
            .to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The `::`-separated path of the item or module, for example `prelude::Order` or `Order`."
                }
            },
            "required": ["path"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.index.render_doc(&args.path).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn api_index_lists_prelude_and_modules() {
        let tool = ApiIndexTool::new(Arc::new(crate::doc_index::tests::fixture_index()));

        let listing = PortableTool::call(&tool, ApiIndexArgs { module: None })
            .await
            .expect("prelude exists");
        assert!(listing.contains("fn submit_order"));

        let listing = PortableTool::call(
            &tool,
            ApiIndexArgs {
                module: Some("prelude::indicators".to_string()),
            },
        )
        .await
        .expect("module exists");
        assert_eq!(listing, "fn zscore\n");

        let err = PortableTool::call(
            &tool,
            ApiIndexArgs {
                module: Some("nope".to_string()),
            },
        )
        .await
        .expect_err("the path does not resolve");
        assert!(matches!(err, DocIndexError::ModuleNotFound(_)));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn api_doc_renders_definitions_and_reports_unknown_paths() {
        let tool = ApiDocTool::new(Arc::new(crate::doc_index::tests::fixture_index()));

        let doc = PortableTool::call(
            &tool,
            ApiDocArgs {
                path: "decimal".to_string(),
            },
        )
        .await
        .expect("re-export resolves");
        assert!(doc.contains("macro_rules! decimal"));

        let err = PortableTool::call(
            &tool,
            ApiDocArgs {
                path: "nope".to_string(),
            },
        )
        .await
        .expect_err("the path does not resolve");
        assert!(matches!(err, DocIndexError::ItemNotFound(_)));
    }
}

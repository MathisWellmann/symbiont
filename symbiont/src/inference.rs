// SPDX-License-Identifier: MPL-2.0
//! Module containing inference related functions.

use std::env::var;

use rig_core::{
    client::CompletionClient,
    providers::openrouter,
};

use crate::Result;

/// Initialize a pre-configured [`crate::AgentBuilder`] using the environment variables.
///
/// The returned builder already has the inference client (from the env vars
/// below) and the symbiont system prompt attached. Customize it with the full
/// `rig` builder API — most notably tool registration — before calling
/// `.build()`:
///
/// ```no_run
/// use rig_core::{
///     completion::ToolDefinition,
///     tool::Tool,
/// };
///
/// #[derive(Debug, thiserror::Error)]
/// #[error("running the tests failed")]
/// struct RunTestsError;
///
/// struct RunTests;
///
/// impl Tool for RunTests {
///     const NAME: &'static str = "run_tests";
///
///     type Error = RunTestsError;
///     type Args = ();
///     type Output = String;
///
///     async fn definition(&self, _prompt: String) -> ToolDefinition {
///         ToolDefinition {
///             name: Self::NAME.to_string(),
///             description: "Run the host crate's test suite and return its output".to_string(),
///             parameters: serde_json::json!({ "type": "object", "properties": {} }),
///         }
///     }
///
///     async fn call(&self, (): Self::Args) -> Result<Self::Output, Self::Error> {
///         Ok("all tests passed".to_string())
///     }
/// }
///
/// # async fn example() -> symbiont::Result<()> {
/// let agent = symbiont::agent_builder(Some("my-crate"))
///     .await?
///     .tool(RunTests)
///     .default_max_turns(5)
///     .build();
/// # Ok(())
/// # }
/// ```
///
/// Rig drives the tool-calling loop internally during [`crate::Runtime::evolve`].
/// When registering tools, also set `.default_max_turns(n)` (`n >= 1`): rig's
/// default of `0` allows only a single tool round-trip and returns
/// `MaxTurnsError` if the model chains tool calls.
///
/// # Arguments:
/// - `opt_crate_name`: If `Some`, then documentation for that crate will be built and included in the system prompt,
///   to inform the agent which methods are available in the dylib.
///   Usually this will be `Some(env!("CARGO_PKG_NAME"))`;
///
/// # Required Env vars:
/// - `API_KEY`: The API key for authenticating the requests, if any. Can be empty
/// - `BASE_URL`: The inference endpoint for `/v1/chat/completions` based requests.
/// - `MODEL`: The model slug.
///
pub async fn agent_builder(opt_crate_name: Option<&str>) -> Result<crate::AgentBuilder> {
    let api_key = var("API_KEY").unwrap_or_default();
    let base_url = var("BASE_URL").unwrap_or_default();
    let model = var("MODEL").unwrap_or_default();

    let client = openrouter::Client::builder()
        .api_key(api_key)
        .base_url(base_url)
        .build()?;

    let system_prompt = crate::system_prompt::system_prompt(opt_crate_name).await?;
    Ok(client.agent(model).preamble(&system_prompt))
}

/// Initialize the agent using the environment variables.
///
/// Convenience wrapper around [`agent_builder`] for agents without tools.
/// To register tools or customize the agent (temperature, max turns, hooks),
/// use [`agent_builder`] instead.
///
/// # Arguments:
/// - `opt_crate_name`: If `Some`, then documentation for that crate will be built and included in the system prompt,
///   to inform the agent which methods are available in the dylib.
///   Usually this will be `Some(env!("CARGO_PKG_NAME"))`;
///
/// # Required Env vars:
/// - `API_KEY`: The API key for authenticating the requests, if any. Can be empty
/// - `BASE_URL`: The inference endpoint for `/v1/chat/completions` based requests.
/// - `MODEL`: The model slug.
///
pub async fn init_agent(opt_crate_name: Option<&str>) -> Result<crate::Agent> {
    Ok(agent_builder(opt_crate_name).await?.build())
}

/* TODO: collect the token usage in the runtime and provide summary stats. This test is used for exploring this path.
#[cfg(test)]
mod tests {
    use rig_core::completion::Prompt;

    use super::*;

    #[tokio::test]
    async fn inference_usage() {
        let agent = init_agent(None).await.unwrap();
        let resp = agent
            .prompt("Hello, whats 1+1?")
            .extended_details()
            .await
            .unwrap();
        dbg!(&resp);
    }
}
*/

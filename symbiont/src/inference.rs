// SPDX-License-Identifier: MPL-2.0
//! Module containing inference related functions.
//!
//! The core constructors ([`agent_builder`], [`init_agent`]) take the
//! inference endpoint, credentials, and model explicitly — the library never
//! reads configuration from the process environment. The `*_from_env`
//! variants are thin conveniences for binaries that follow the
//! `BASE_URL`/`API_KEY` env-var convention.

use std::env::var;

use rig_agent::client::AgentClientExt;
use rig_core::{
    http_client::ReqwestClient,
    providers::openrouter,
};

use crate::{
    MeteredHttpClient,
    Result,
    ThinkingLevel,
};

/// Initialize a pre-configured [`crate::AgentBuilder`] for `model`.
///
/// The returned builder already has the inference client (talking to `model`
/// at `base_url`) and the symbiont system prompt attached. Customize it with
/// the full `rig` builder API — most notably tool registration — before
/// calling `.build()`:
///
/// ```no_run
/// use rig_core::tool::PortableTool;
/// use symbiont::ThinkingLevel;
///
/// #[derive(Debug, thiserror::Error)]
/// #[error("running the tests failed")]
/// struct RunTestsError;
///
/// struct RunTests;
///
/// impl PortableTool for RunTests {
///     const NAME: &'static str = "run_tests";
///
///     type Error = RunTestsError;
///     type Args = ();
///     type Output = String;
///
///     fn description(&self) -> String {
///         "Run the host crate's test suite and return its output".to_string()
///     }
///
///     fn parameters(&self) -> serde_json::Value {
///         serde_json::json!({ "type": "object", "properties": {} })
///     }
///
///     async fn call(&self, (): Self::Args) -> Result<Self::Output, Self::Error> {
///         Ok("all tests passed".to_string())
///     }
/// }
///
/// # async fn example() -> symbiont::Result<()> {
/// let agent = symbiont::agent_builder(
///     Some("my-crate"),
///     "http://127.0.0.1:8321/v1",
///     "",
///     "qwen3.6",
///     ThinkingLevel::Disabled,
/// )
/// .await?
/// .tool(RunTests)
/// .default_max_turns(5)
/// .build();
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
/// - `base_url`: The inference endpoint for `/v1/chat/completions` based requests.
/// - `api_key`: The API key for authenticating the requests, if any. Can be empty.
/// - `model`: The model slug served at `base_url`.
/// - `thinking`: The [`ThinkingLevel`] configuring reasoning effort across inference providers
///   (vLLM, llama-server, OpenRouter, OpenAI, etc.). Can also be passed as `bool` (`false` -> `Disabled`, `true` -> `Medium`).
///   Keep this [`ThinkingLevel::Disabled`] for thinking models (e.g. Qwen3, DeepSeek) in latency-sensitive loops.
///
pub async fn agent_builder(
    opt_crate_name: Option<&str>,
    base_url: &str,
    api_key: &str,
    model: &str,
    thinking: impl Into<ThinkingLevel>,
) -> Result<crate::AgentBuilder> {
    let client = openrouter::Client::builder()
        .api_key(api_key)
        .base_url(base_url)
        // Replaces rig's default backend with the same `reqwest::Client`,
        // wrapped so every outbound prompt payload is measured
        // (`observability::REQUEST_BODY_BYTES`).
        .http_client(MeteredHttpClient::new(ReqwestClient::default()))
        .build()?;

    let system_prompt = crate::system_prompt::system_prompt(opt_crate_name).await?;
    let thinking_level: ThinkingLevel = thinking.into();

    Ok(client
        .agent(model)
        .preamble(&system_prompt)
        .additional_params(thinking_level.to_additional_params()))
}

/// [`agent_builder`] with the endpoint and credentials read from the
/// environment: `BASE_URL` and `API_KEY` (both may be absent or empty).
pub async fn agent_builder_from_env(
    opt_crate_name: Option<&str>,
    model: &str,
    thinking: impl Into<ThinkingLevel>,
) -> Result<crate::AgentBuilder> {
    let base_url = var("BASE_URL").unwrap_or_default();
    let api_key = var("API_KEY").unwrap_or_default();
    agent_builder(opt_crate_name, &base_url, &api_key, model, thinking).await
}

/// Initialize the agent for `model`.
///
/// Convenience wrapper around [`agent_builder`] for agents without tools.
/// To register tools or customize the agent (temperature, max turns, hooks),
/// use [`agent_builder`] instead.
///
/// # Arguments:
/// - `opt_crate_name`: If `Some`, then documentation for that crate will be built and included in the system prompt,
///   to inform the agent which methods are available in the dylib.
///   Usually this will be `Some(env!("CARGO_PKG_NAME"))`;
/// - `base_url`: The inference endpoint for `/v1/chat/completions` based requests.
/// - `api_key`: The API key for authenticating the requests, if any. Can be empty.
/// - `model`: The model slug served at `base_url`.
/// - `thinking`: The [`ThinkingLevel`] configuring reasoning effort. See [`agent_builder`].
///
pub async fn init_agent(
    opt_crate_name: Option<&str>,
    base_url: &str,
    api_key: &str,
    model: &str,
    thinking: impl Into<ThinkingLevel>,
) -> Result<crate::Agent> {
    Ok(
        agent_builder(opt_crate_name, base_url, api_key, model, thinking)
            .await?
            .build(),
    )
}

/// [`init_agent`] with the endpoint and credentials read from the
/// environment: `BASE_URL` and `API_KEY` (both may be absent or empty).
pub async fn init_agent_from_env(
    opt_crate_name: Option<&str>,
    model: &str,
    thinking: impl Into<ThinkingLevel>,
) -> Result<crate::Agent> {
    Ok(agent_builder_from_env(opt_crate_name, model, thinking)
        .await?
        .build())
}

/* TODO: collect the token usage in the runtime and provide summary stats. This test is used for exploring this path.
#[cfg(test)]
mod tests {
    use rig_agent::completion::Prompt;

    use super::*;

    #[tokio::test]
    async fn inference_usage() {
        let agent = init_agent_from_env(None, "test-model").await.unwrap();
        let resp = agent
            .prompt("Hello, whats 1+1?")
            .extended_details()
            .await
            .unwrap();
        dbg!(&resp);
    }
}
*/

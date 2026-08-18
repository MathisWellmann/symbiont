// SPDX-License-Identifier: MPL-2.0
//! Module containing inference related functions.
//!
//! The core constructors ([`agent_builder`], [`init_agent`]) take the
//! inference endpoint, credentials, and model explicitly — the library never
//! reads configuration from the process environment. The `*_from_env`
//! variants are thin conveniences for binaries that follow the
//! `BASE_URL`/`API_KEY` env-var convention.

use std::env::var;

use rig_core::{
    client::CompletionClient,
    http_client::ReqwestClient,
    providers::openrouter,
};

use crate::{
    MeteredHttpClient,
    Result,
};

/// Initialize a pre-configured [`crate::AgentBuilder`] for `model`.
///
/// The returned builder already has the inference client (talking to `model`
/// at `base_url`) and the symbiont system prompt attached. Customize it with
/// the full `rig` builder API — most notably tool registration — before
/// calling `.build()`:
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
/// let agent = symbiont::agent_builder(
///     Some("my-crate"),
///     "http://127.0.0.1:8321/v1",
///     "",
///     "qwen3.6",
///     false,
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
/// - `enable_thinking`: Whether the model may emit reasoning ("thinking") tokens
///   before answering. Configures `chat_template_kwargs` (`enable_thinking` and `thinking`)
///   for vLLM/llama-server Jinja templates, standard OpenAI/vLLM `reasoning_effort`,
///   and OpenRouter `reasoning` settings.
///   Keep this `false` for thinking models (e.g. Qwen3, DeepSeek) in latency-sensitive loops.
///
pub async fn agent_builder(
    opt_crate_name: Option<&str>,
    base_url: &str,
    api_key: &str,
    model: &str,
    enable_thinking: bool,
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
    let reasoning_params = if enable_thinking {
        serde_json::json!({
            "enable_thinking": true,
            "chat_template_kwargs": {
                "enable_thinking": true,
                "thinking": true,
            },
            "reasoning": {
                "effort": "medium",
            },
        })
    } else {
        serde_json::json!({
            "enable_thinking": false,
            "reasoning_effort": "none",
            "chat_template_kwargs": {
                "enable_thinking": false,
                "thinking": false,
            },
            "reasoning": {
                "effort": "none",
                "max_tokens": 0,
            },
        })
    };

    Ok(client
        .agent(model)
        .preamble(&system_prompt)
        .additional_params(reasoning_params))
}

/// [`agent_builder`] with the endpoint and credentials read from the
/// environment: `BASE_URL` and `API_KEY` (both may be absent or empty).
pub async fn agent_builder_from_env(
    opt_crate_name: Option<&str>,
    model: &str,
    enable_thinking: bool,
) -> Result<crate::AgentBuilder> {
    let base_url = var("BASE_URL").unwrap_or_default();
    let api_key = var("API_KEY").unwrap_or_default();
    agent_builder(opt_crate_name, &base_url, &api_key, model, enable_thinking).await
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
/// - `enable_thinking`: Whether the model may emit reasoning ("thinking") tokens
///   before answering. See [`agent_builder`].
///
pub async fn init_agent(
    opt_crate_name: Option<&str>,
    base_url: &str,
    api_key: &str,
    model: &str,
    enable_thinking: bool,
) -> Result<crate::Agent> {
    Ok(
        agent_builder(opt_crate_name, base_url, api_key, model, enable_thinking)
            .await?
            .build(),
    )
}

/// [`init_agent`] with the endpoint and credentials read from the
/// environment: `BASE_URL` and `API_KEY` (both may be absent or empty).
pub async fn init_agent_from_env(
    opt_crate_name: Option<&str>,
    model: &str,
    enable_thinking: bool,
) -> Result<crate::Agent> {
    Ok(
        agent_builder_from_env(opt_crate_name, model, enable_thinking)
            .await?
            .build(),
    )
}

/* TODO: collect the token usage in the runtime and provide summary stats. This test is used for exploring this path.
#[cfg(test)]
mod tests {
    use rig_core::completion::Prompt;

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

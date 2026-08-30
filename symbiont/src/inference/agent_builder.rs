// SPDX-License-Identifier: MPL-2.0
//! Module containing inference related functions.
//!
//! The core constructors ([`agent_builder`], [`init_agent`]) take the
//! inference endpoint, credentials, and model explicitly — the library never
//! reads configuration from the process environment. The `*_from_env`
//! variants are thin conveniences for binaries that follow the
//! `BASE_URL`/`API_KEY` env-var convention.

use std::{
    env::var,
    sync::Arc,
    time::Duration,
};

use rig_agent::client::AgentClientExt;
use rig_core::providers::openrouter;
use rig_reqwest::{
    ReqwestClient,
    reqwest,
};

use crate::{
    Agent,
    ApiDocTool,
    ApiIndexTool,
    DocIndex,
    DocMode,
    MeteredHttpClient,
    Result,
    ThinkingLevel,
};

/// Total time budget for one completion request, from send until the
/// response body is fully read.
///
/// A hung endpoint (accepted connection, no response) must not stall an
/// evolution lane forever. Past the deadline the request fails with a
/// connection-level error, which
/// [`Runtime::MAX_TRANSIENT_RETRIES`](crate::Runtime::MAX_TRANSIENT_RETRIES)
/// classifies as transient: it is retried with backoff and does *not* count
/// against the lane's self-healing attempt budget. A permanently hung
/// endpoint therefore costs at most this timeout times the transient retry
/// budget before the lane gives up.
pub const INFERENCE_REQUEST_TIMEOUT: Duration = Duration::from_secs(15 * 60);

/// The turn budget that [`agent_builder`] sets when the [`DocMode`] registers
/// the documentation tools.
///
/// One attempt usually needs zero to two documentation calls before the
/// final answer. The budget leaves room for chained calls and for retries of
/// invalid tool calls. A host that registers more tools must set
/// `.default_max_turns(n)` with room for both.
pub const DOC_TOOLS_MAX_TURNS: usize = 8;

/// Initialize a pre-configured [`crate::AgentBuilder`] for `model`.
///
/// The returned builder already has the inference client (talking to `model`
/// at `base_url`) and the symbiont system prompt attached. Customize it with
/// the full `rig` builder API — most notably tool registration — before
/// calling `.build()`:
///
/// ```no_run
/// use rig_core::tool::PortableTool;
/// use symbiont::{DocMode, ThinkingLevel};
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
///     DocMode::default(),
///     "http://127.0.0.1:8321/v1",
///     "",
///     "qwen3.6",
///     ThinkingLevel::Disabled,
/// )
/// .await?
/// .tool(RunTests)
/// .default_max_turns(10)
/// .build();
/// # Ok(())
/// # }
/// ```
///
/// Rig drives the tool-calling loop internally during [`crate::Runtime::evolve`].
/// When `doc_mode` registers the documentation tools, the builder gets a
/// `default_max_turns` of [`DOC_TOOLS_MAX_TURNS`]. When you register your own
/// tools, set `.default_max_turns(n)` with room for both: rig's default of
/// `0` allows only a single tool round-trip and returns `MaxTurnsError` if
/// the model chains tool calls.
///
/// The builder builds a bare `rig_agent::Agent`. Before handing it to the
/// runtime, wrap it with [`crate::Agent::new`], passing the same `base_url`.
///
/// # Arguments:
/// - `opt_crate_name`: The crate whose API the evolved code can use, usually
///   `Some(env!("CARGO_PKG_NAME"))`. With `None`, no host API is documented
///   and `doc_mode` has no effect.
/// - `doc_mode`: How the agent gets the host API documentation: inline in
///   the system prompt, or on demand through the `api_index` and `api_doc`
///   tools.
/// - `base_url`: The inference endpoint for `/v1/chat/completions` based requests.
/// - `api_key`: The API key for authenticating the requests, if any. Can be empty.
/// - `model`: The model slug served at `base_url`.
/// - `thinking`: The [`ThinkingLevel`] configuring reasoning effort across inference providers
///   (vLLM, llama-server, OpenRouter, OpenAI, etc.). Can also be passed as `bool` (`false` -> `Disabled`, `true` -> `Medium`).
///   Keep this [`ThinkingLevel::Disabled`] for thinking models (e.g. Qwen3, DeepSeek) in latency-sensitive loops.
///
pub async fn agent_builder(
    opt_crate_name: Option<&str>,
    doc_mode: DocMode,
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
        .http_client(MeteredHttpClient::new(ReqwestClient(
            reqwest::Client::builder()
                .timeout(INFERENCE_REQUEST_TIMEOUT)
                .build()
                .map_err(std::io::Error::other)?,
        )))
        .build()?;

    let system_prompt = crate::system_prompt::system_prompt(opt_crate_name, doc_mode).await?;
    let thinking_level: ThinkingLevel = thinking.into();

    let builder = client
        .agent(model)
        .preamble(&system_prompt)
        .additional_params(thinking_level.to_additional_params());

    // The builder always transitions to the tool state so the return type
    // does not depend on the doc mode.
    let builder = match (opt_crate_name, doc_mode.uses_tools()) {
        (Some(crate_name), true) => {
            let index = DocIndex::host(crate_name).await?;
            builder
                .tool(ApiIndexTool::new(Arc::clone(&index)))
                .tool(ApiDocTool::new(index))
                .default_max_turns(DOC_TOOLS_MAX_TURNS)
        }
        _ => builder.dynamic_tools(Vec::new()),
    };
    Ok(builder)
}

/// Create an `Agent` with the endpoint and credentials read from the
/// environment: `BASE_URL` and `API_KEY` (both may be absent or empty).
pub async fn agent_from_env(
    opt_crate_name: Option<&str>,
    doc_mode: DocMode,
    model: &str,
    thinking: impl Into<ThinkingLevel>,
) -> Result<Agent> {
    let base_url = var("BASE_URL").unwrap_or_default();
    let api_key = var("API_KEY").unwrap_or_default();
    agent_builder(
        opt_crate_name,
        doc_mode,
        &base_url,
        &api_key,
        model,
        thinking,
    )
    .await
    .map(|v| Agent::new(v.build(), base_url, model))
}

/// Initialize the agent for `model`.
///
/// Convenience wrapper around [`agent_builder`] for agents without tools.
/// To register tools or customize the agent (temperature, max turns, hooks),
/// use [`agent_builder`] instead.
///
/// # Arguments:
/// - `opt_crate_name`: The crate whose API the evolved code can use, usually
///   `Some(env!("CARGO_PKG_NAME"))`. With `None`, no host API is documented
///   and `doc_mode` has no effect.
/// - `doc_mode`: How the agent gets the host API documentation. See [`agent_builder`].
/// - `base_url`: The inference endpoint for `/v1/chat/completions` based requests.
/// - `api_key`: The API key for authenticating the requests, if any. Can be empty.
/// - `model`: The model slug served at `base_url`.
/// - `thinking`: The [`ThinkingLevel`] configuring reasoning effort. See [`agent_builder`].
///
pub async fn init_agent(
    opt_crate_name: Option<&str>,
    doc_mode: DocMode,
    base_url: &str,
    api_key: &str,
    model: &str,
    thinking: impl Into<ThinkingLevel>,
) -> Result<Agent> {
    let agent = agent_builder(opt_crate_name, doc_mode, base_url, api_key, model, thinking)
        .await?
        .build();
    Ok(Agent::new(agent, base_url, model))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn init_agent_carries_the_base_url_as_provider() {
        let agent = init_agent(
            None,
            DocMode::default(),
            "http://127.0.0.1:8321/v1",
            "",
            "model",
            ThinkingLevel::Disabled,
        )
        .await
        .expect("building a local agent needs no network");
        assert_eq!(agent.provider(), "http://127.0.0.1:8321/v1");
    }
}

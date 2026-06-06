// SPDX-License-Identifier: MPL-2.0
//! Module containing inference related functions.

use std::env::var;

use rig_core::{
    client::CompletionClient,
    providers::openrouter,
};

use crate::Result;

/// Initialize the agent using the environment variables.
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
    let api_key = var("API_KEY").unwrap_or_default();
    let base_url = var("BASE_URL").unwrap_or_default();
    let model = var("MODEL").unwrap_or_default();

    let client = openrouter::Client::builder()
        .api_key(api_key)
        .base_url(base_url)
        .build()?;

    let system_prompt = crate::system_prompt::system_prompt(opt_crate_name).await?;
    Ok(client.agent(model).preamble(&system_prompt).build())
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

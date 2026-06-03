// SPDX-License-Identifier: MPL-2.0
//! Module containing inference related functions.

use std::env::var;

use rig_core::{
    agent::Agent,
    client::CompletionClient,
    providers::{
        openrouter,
        openrouter::completion::CompletionModel,
    },
};

use crate::Result;

/// Initialize the agent using the environment variables.
pub async fn init_agent(crate_name: &str) -> Result<Agent<CompletionModel>> {
    let api_key = var("API_KEY").unwrap_or_default();
    let base_url = var("BASE_URL").unwrap_or_default();
    let model = var("MODEL").unwrap_or_default();

    let client = openrouter::Client::builder()
        .api_key(api_key)
        .base_url(base_url)
        .build()?;

    let system_prompt = crate::system_prompt::system_prompt(crate_name).await?;
    Ok(client.agent(model).preamble(&system_prompt).build())
}

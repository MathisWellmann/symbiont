// SPDX-License-Identifier: MPL-2.0
//! Module containing inference related functions.

use std::env::var;

use rig::{
    agent::Agent,
    client::CompletionClient,
    providers::{
        openai,
        openai::completion::CompletionModel,
    },
};

/// Initialize the agent using the environment variables.
pub fn init_agent() -> crate::Result<Agent<CompletionModel>> {
    let api_key = var("API_KEY").unwrap_or_default();
    let base_url = var("BASE_URL").unwrap_or_default();
    let model = var("MODEL").unwrap_or_default();

    let client = openai::Client::builder()
        .api_key(api_key)
        .base_url(base_url)
        .build()?
        .completions_api(); // Use Chat Completions API instead of Responses API

    // Create agent with a single context prompt
    Ok(client
        .agent(model)
        .preamble("You are a Rust Software Engineer, specialized in function body implementations.")
        .build())
}

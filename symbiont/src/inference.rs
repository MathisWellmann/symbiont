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
use tracing::debug;

use crate::doc_string::write_prelude_doc_string;

/// Initialize the agent using the environment variables.
pub async fn init_agent(crate_name: &str) -> crate::Result<Agent<CompletionModel>> {
    let api_key = var("API_KEY").unwrap_or_default();
    let base_url = var("BASE_URL").unwrap_or_default();
    let model = var("MODEL").unwrap_or_default();

    let client = openai::Client::builder()
        .api_key(api_key)
        .base_url(base_url)
        .build()?
        .completions_api(); // Use Chat Completions API instead of Responses API

    let mut system_prompt = "
    # Purpose

    You are a Rust Coding Agent in the `symbiont` agent harness,
    which parses your generated output, checks it against the existing function signatures,
    which must match exactly and then compiles your code as a dynamic library.
    If the code does not compile or does not match the function signatures required,
    then the error is fed back into the next prompt and a correction is demanded.
    The harness is running in the host binary and compiles your function implementations as a dylib,
    swaps out the atomic pointer to the functions in the host binary, runs the function under some evaluation conditions
    and feeds the results back into the prompt for the next iteration in order to improve the implementation.
    The evaluation in the host binary could include a test suite for correctness, a performance benchmark,
    a black box function search or auto-research style hyperparameter tuning, among other things.
    The user prompt will include the concrete goal.

    The following section contains the `cargo doc` generated documentation as markdown, showing the available methods
    which you can call, if any. If the section is empty then you can only use the rust builtin standard library
    and no other dependencies. Here it is:

    "
    .to_string();
    write_prelude_doc_string(&mut system_prompt, crate_name).await?;
    debug!("system_prompt: {system_prompt}");

    Ok(client.agent(model).preamble(&system_prompt).build())
}

mod error;
mod tests;

use error::Result;
use tracing::info;
use tracing_subscriber::EnvFilter;
mod function_parser;

use std::env::var;

use rig::{client::CompletionClient, completion::Prompt, providers::openai};

use crate::function_parser::parse_functions;

// The value of `dylib = "..."` should be the library containing the hot-reloadable functions
// It should normally be the crate name of your sub-crate.
#[hot_lib_reloader::hot_module(
    dylib = "symbiont_lib",
    lib_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../target/debug")
)]
mod hot_lib {
    // Reads public no_mangle functions from lib.rs and  generates hot-reloadable
    // wrapper functions with the same signature inside this module.
    // Note that this path relative to the project root (or absolute)
    hot_functions_from_file!("symbiont-lib/src/lib.rs");

    // Because we generate functions with the exact same signatures,
    // we need to import types used
    // pub use symbiont_lib::State;
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_line_number(true)
        .init();

    let fn_sigs = parse_functions()?;
    info!("fn_sigs: {fn_sigs:?}");
    assert_eq!(
        fn_sigs.len(),
        1,
        "Only 1 public function is supported for now"
    );

    let api_key = var("API_KEY").unwrap_or_default();
    let base_url = var("BASE_URL").unwrap_or_default();
    let model = var("MODEL").unwrap_or_default();

    let client = openai::Client::builder()
        .api_key(api_key)
        .base_url(base_url)
        .build()?
        .completions_api(); // Use Chat Completions API instead of Responses API

    // Create agent with a single context prompt
    let agent = client
        .agent(model)
        .preamble("You are a Rust Software Engineer, specialized in function body implementations.")
        .build();

    // Prompt the agent and print the response
    let prompt = format!(
        "Give a concise implementation for this function signature: ```{}```. Code Only",
        fn_sigs[0]
    );
    info!("prompt: {prompt}");
    let response = agent.prompt(prompt).await?;
    info!("{response}");

    // let mut state = hot_lib::State { counter: 0 };
    let mut counter = 1;
    // Running in a loop so you can modify the code and see the effects
    loop {
        hot_lib::step(&mut counter);
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}

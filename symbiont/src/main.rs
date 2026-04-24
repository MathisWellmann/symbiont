mod error;
mod inference;
mod tests;
mod utils;
mod validation;

use error::Result;
use tracing::info;
use tracing_subscriber::EnvFilter;
mod function_parser;
mod parser;

use rig::completion::Prompt;

use crate::{
    error::Error, function_parser::parse_functions, inference::init_agent, parser::parse_rust_code,
    validation::validate_generated_ast,
};

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

    let agent = init_agent()?;

    // Prompt the agent and print the response
    let prompt = format!(
        "Give a concise implementation for this function signature: ```{}```. Code Only. Function must have `pub` visibility and `#[unsafe(no_mangle)]` annotation",
        fn_sigs[0]
    );
    info!("prompt: {prompt}");
    let response = agent.prompt(prompt).await?;
    info!("{response}");

    let ast = parse_rust_code(&response).map_err(|_| Error::CouldNotParseRust)?;
    validate_generated_ast(&ast, &fn_sigs)?;

    // TODO: overwrite the existing `lib.rs` file with new code
    // TODO: compile rust code
    // TODO: run new rust code.

    // let mut state = hot_lib::State { counter: 0 };
    let mut counter = 1;
    // Running in a loop so you can modify the code and see the effects
    loop {
        hot_lib::step(&mut counter);
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}

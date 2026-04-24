mod compiler;
mod error;
mod inference;
mod tests;
mod utils;
mod validation;
mod writer;

use error::Result;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;
mod function_parser;
mod parser;

use rig::{agent::Agent, completion::Prompt, providers::openai::completion::CompletionModel};
use std::time::Duration;

use crate::{
    compiler::compile_lib,
    error::Error,
    function_parser::{FuncSig, parse_functions},
    inference::init_agent,
    parser::parse_rust_code,
    validation::validate_generated_ast,
    writer::write_generated_lib,
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

    #[lib_change_subscription]
    pub fn subscribe() -> hot_lib_reloader::LibReloadObserver {}

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

    // let mut state = hot_lib::State { counter: 0 };
    let mut counter = 1;
    // Running in a loop so you can modify the code and see the effects
    loop {
        hot_lib::step(&mut counter);
        println!("counter: {counter}");
        std::thread::sleep(std::time::Duration::from_secs(1));

        if counter % 10 == 0 {
            // Prompt the agent and print the response
            let base_prompt = format!(
                "Give a concise implementation for this function signature: ```{}```, \
                that increments the counter by a constant in the range (5..20). \
                Code Only. Function must have `pub` visibility.",
                fn_sigs[0]
            );
            let mut prompt = base_prompt.clone();

            while let Err(e) = evolve(&agent, &prompt, &fn_sigs).await {
                info!("Function evolution error: {e}");

                // Restore the original base prompt, then add steering to allow self-healing LLM generation (constrained generation).
                prompt = base_prompt.clone();

                use Error::*;
                match e {
                    NoRustCode => prompt.push_str(
                        "Your response did not contain a rust code block. Please try again.",
                    ),
                    CouldNotParseRust => prompt.push_str(
                        "Your response did not contain valid Rust code. Please try again",
                    ),
                    WriteLib(_) => todo!(),
                    SignatureMismatch{ name: _, expected, got } => prompt.push_str(&format!("Generated function signature miss-match. Expected ```{expected}```, Got ```{got}```")),
                    CompilationFailed(ref stderr) => prompt.push_str(&format!("The generated code failed to compile. Compiler output:\n```\n{stderr}\n```\nPlease fix the compilation errors.")),
                    _ => warn!("Unhandled error"),
                }
            }
            info!("Successfully evolved the function");
        }
    }
}

async fn evolve(agent: &Agent<CompletionModel>, prompt: &str, fn_sigs: &[FuncSig]) -> Result<()> {
    info!("prompt: {prompt}");
    let response = agent.prompt(prompt).await?;
    info!("{response}");

    let mut ast = parse_rust_code(&response).map_err(|_| Error::CouldNotParseRust)?;
    validate_generated_ast(&mut ast, &fn_sigs)?;

    // Subscribe to reload events before triggering any changes,
    // so we don't miss the notification.
    let observer = hot_lib::subscribe();

    // Write the validated AST to lib.rs
    write_generated_lib(&ast)?;

    // Compile the library (replaces external `cargo watch`)
    compile_lib().await?;

    // Wait for hot_lib_reloader to pick up the new .so and reload it.
    const RELOAD_TIMEOUT: Duration = Duration::from_secs(10);
    if observer.wait_for_reload_timeout(RELOAD_TIMEOUT) {
        info!("Library hot-reloaded successfully");
    } else {
        warn!("Timed out waiting for library hot-reload");
    }

    Ok(())
}

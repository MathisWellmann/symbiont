//! The example shows a basic function which gets agenticly evolved with a user-defined prompt
//! and the compiled dylib gets hot-swapped in the hot-loop, achieving bare-metal performance.

use std::time::Duration;

use rig::{
    agent::Agent,
    completion::Prompt,
    providers::openai::completion::CompletionModel,
};
use symbiont::{
    Error,
    Runtime,
};
use tracing::{
    info,
    warn,
};
use tracing_subscriber::EnvFilter;

// The starting function definition, used during constrained generation,
// where the LLM model will implement the function body.
// The body can be empty too. The Agent will only see the function signature.
symbiont::evolvable! {
    fn step(counter: &mut usize) {
        *counter += 1;
        println!("doing stuff in iteration {}", counter);
    }
}

// TODO: the example should be more minimal.
#[tokio::main]
async fn main() -> symbiont::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_line_number(true)
        .init();

    let runtime = Runtime::init(SYMBIONT_DECLS).await?;
    let fn_sigs = runtime.fn_sigs();
    info!("fn_sigs: {fn_sigs:?}");

    let agent = symbiont::inference::init_agent()?;

    let mut counter = 1;
    // Running in a loop so you can modify the code and see the effects
    loop {
        step(&mut counter);
        println!("counter: {counter}");
        std::thread::sleep(Duration::from_secs(1));

        if counter % 10 == 0 {
            let base_prompt = format!(
                "Give a concise implementation for this function signature: ```{}```, \
                that increments the counter by a constant in the range (5..20). \
                Code Only",
                fn_sigs[0]
            );
            let mut prompt = base_prompt.clone();

            while let Err(e) = evolve_step(runtime, &agent, &prompt).await {
                info!("Function evolution error: {e}");

                prompt = base_prompt.clone();

                // TODO: the example should not have to have the back-pressure here.
                use Error::*;
                match e {
                    NoRustCode => prompt.push_str(
                        "Your response did not contain a rust code block. Please try again and make sure its wrapped like this: ```CODE```",
                    ),
                    CouldNotParseRust => prompt.push_str(
                        "Your response did not contain valid Rust code. Please try again",
                    ),
                    WriteLib(_) => todo!(),
                    SignatureMismatch {
                        name: _,
                        expected,
                        got,
                    } => prompt.push_str(&format!(
                        "Generated function signature miss-match. Expected ```{expected}```, Got ```{got}```"
                    )),
                    CompilationFailed(ref stderr) => prompt.push_str(&format!(
                        "The generated code failed to compile. Compiler output:\n```\n{stderr}\n```\nPlease fix the compilation errors."
                    )),
                    _ => warn!("Unhandled error"),
                }
            }
            info!("Successfully evolved the function");
        }
    }
}

async fn evolve_step(
    runtime: &Runtime,
    agent: &Agent<CompletionModel>,
    prompt: &str,
) -> symbiont::Result<()> {
    info!("prompt: {prompt}");
    let response = agent.prompt(prompt).await?;
    info!("{response}");
    runtime.evolve(&response).await
}

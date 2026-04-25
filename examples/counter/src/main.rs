//! The example shows a basic function which gets agenticly evolved with a user-defined prompt
//! and the compiled dylib gets hot-swapped in the hot-loop, achieving bare-metal performance.

use std::time::Duration;

use symbiont::Runtime;
use tracing::info;
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
            info!("base_prompt: {base_prompt}");

            runtime
                .evolve_with_backpressure(&agent, &base_prompt)
                .await
                .expect("Can successfully evolve");
            info!(
                "Successfully evolved the function, which is now hot-reloaded in-place. Next call to `step` will run the newly compiled Agent code."
            );
        }
    }
}

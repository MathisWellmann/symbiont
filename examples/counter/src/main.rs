// SPDX-License-Identifier: MPL-2.0
//! The example shows a basic counter function where the Agent evolves the implementation,
//! based on a user-defined prompt.
//! The compiled dylib (of the function) gets hot-swapped in the evaluation loop, achieving bare-metal performance.
//! This is agentic code mode in action.
//! The harness provides constrained generation and nudges the LLM prompt if necessary.

use std::time::Duration;

use symbiont::Runtime;
use tracing::info;

// The starting function definition, used during constrained generation,
// where the LLM model will implement the function body.
// The body can be empty too.
// If prompting with `fn_sigs`, then the Agent will only see the function signature. This is used here.
// If prompting with `fn_full_sources`, then the Agent will see the entire function, including docs and default body. (Not shown here.)
symbiont::evolvable! {
    /// Should increment the counter by a value in the range 5..20
    fn step(counter: &mut usize) {
        *counter += 1;
        println!("doing stuff in iteration {}", counter);
    }
}

#[tokio::main]
async fn main() -> symbiont::Result<()> {
    symbiont::init_tracing();

    let runtime = Runtime::init(SYMBIONT_DECLS, symbiont::Profile::Debug).await?;
    let fn_sigs = runtime.fn_sigs(); // Alternatively, `fn_full_sources` can be used to also show doc string and default function body.
    info!("fn_sigs: {fn_sigs:?}");

    let agent = symbiont::inference::init_agent()?;

    let base_prompt = format!(
        "Give a concise implementation for this function signature: ```{}```, \
        that increments the counter by a constant in the range (5..20). \
        Code Only",
        fn_sigs[0]
    );

    let mut counter = 1;
    let mut last_evolution = std::time::Instant::now();
    let evolution_interval = Duration::from_secs(5);

    loop {
        step(&mut counter);
        println!("counter: {counter}");
        std::thread::sleep(Duration::from_secs(1));

        if last_evolution.elapsed() >= evolution_interval {
            runtime
                .evolve(&agent, &base_prompt)
                .await
                .expect("Can successfully evolve");
            info!(
                "Successfully evolved the function, which is now hot-reloaded in-place. Next call to `step` will run the newly compiled Agent code."
            );
            last_evolution = std::time::Instant::now();
        }
    }
}

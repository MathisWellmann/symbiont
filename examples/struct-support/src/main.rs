// SPDX-License-Identifier: MPL-2.0
//! The example shows support for functions taking in custom structs from the surrounding scope.

use std::time::Duration;

use symbiont::Runtime;
use tracing::info;

/// A 2D game state, with just the x and y coordinates.
#[derive(Debug, Clone)]
#[allow(dead_code, reason = "Just debug impl is used.")]
struct GameState {
    /// The x coodinate. Range of 0..100
    x: usize,
    /// The y coordinate. Range of 0..250
    y: usize,
}

symbiont::evolvable! {
    /// Implement some different logic in here, while respecting the bounds laid out in the docs.
    fn step(state: &mut GameState) {
        if state.x < 100 {
            state.x += 1;
        }
        if state.y < 250 {
            state.x += 1;
        }
    }
}

#[tokio::main]
async fn main() -> symbiont::Result<()> {
    symbiont::init_tracing();

    info!("SYMBIONT_DECLS: {SYMBIONT_DECLS:#?}");
    let runtime = Runtime::init(SYMBIONT_DECLS, symbiont::Profile::Debug).await?;
    let fn_source = runtime.fn_full_sources();
    info!("fn_source: {fn_source:?}");

    let agent = symbiont::inference::init_agent()?;

    let base_prompt = format!(
        "Give an implementation for this function: ```{}```, \
        Give Rust Code Only.",
        fn_source[0]
    );

    let mut last_evolution = std::time::Instant::now();
    let evolution_interval = Duration::from_secs(5);

    let mut state = GameState { x: 0, y: 0 };

    loop {
        step(&mut state);
        println!("state: {state:?}");
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

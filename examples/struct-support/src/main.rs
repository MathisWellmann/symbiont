// SPDX-License-Identifier: MPL-2.0
//! The example shows support for functions taking in custom structs from the surrounding scope.

use std::time::Duration;

use symbiont::Runtime;
use tracing::info;

/// A 2D game state, with just the x and y coordinates.
///
/// Annotated with `#[symbiont::shared]` so the macro records the type's
/// source code into a hidden `__SYMBIONT_SHARED_GameState` constant that
/// `evolvable!` can pull into the dylib via `shared GameState;`.
#[symbiont::shared]
#[derive(Default, Debug, Clone, PartialEq, Eq)]
#[allow(dead_code, reason = "Just debug impl is used.")]
struct GameState {
    /// The x coordinate. Range of 0..100
    x: usize,
    /// The y coordinate. Range of 0..250
    y: usize,
}

symbiont::evolvable! {
    // Bring the externally-defined `GameState` into the dylib's source so the
    // evolved function below can reference it.
    shared GameState;

    /// Implement some different logic in here, while respecting the bounds laid out in the docs.
    fn step(state: &mut GameState);
}

#[tokio::main]
async fn main() -> symbiont::Result<()> {
    symbiont::init_tracing();

    info!("SYMBIONT_DECLS: {SYMBIONT_DECLS:#?}");
    let runtime = Runtime::init(SYMBIONT_DECLS, SYMBIONT_PRELUDE, symbiont::Profile::Debug).await?;
    let fn_prelude = runtime.fn_prelude();
    let fn_source = runtime.fn_full_sources();
    info!("fn_prelude: {fn_prelude:#?}, fn_source: {fn_source:#?}");

    let agent = symbiont::inference::init_agent()?;

    let base_prompt = format!(
        "Give an implementation for this evolvable function:\n
```
{:#?}
{:#?}\n
```.\n",
        fn_source[0], fn_prelude[0],
    );
    runtime
        .evolve(&agent, &base_prompt)
        .await
        .expect("Can successfully evolve");

    let mut last_evolution = std::time::Instant::now();
    let evolution_interval = Duration::from_secs(2);

    let mut state = GameState::default();

    for _ in 0..10 {
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
    assert_ne!(state, GameState::default(), "Game state must have evolved.");

    Ok(())
}

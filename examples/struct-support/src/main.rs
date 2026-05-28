// SPDX-License-Identifier: MPL-2.0
//! The example shows support for functions taking in custom structs from the surrounding scope.

use struct_support_example::prelude::*;
use symbiont::{
    DylibConfig,
    Runtime,
};
use tracing::info;

symbiont::evolvable! {
    /// Implement some different logic in here, while respecting the bounds laid out in the docs.
    fn step(state: &mut GameState);
}

#[tokio::main]
async fn main() -> symbiont::Result<()> {
    symbiont::init_tracing();

    info!("SYMBIONT_DECLS: {SYMBIONT_DECLS:#?}");
    let runtime = Runtime::init(
        SYMBIONT_DECLS,
        SYMBIONT_PRELUDE,
        DylibConfig::host_package(
            symbiont::Profile::Debug,
            env!("CARGO_PKG_NAME"),
            env!("CARGO_MANIFEST_DIR"),
        ),
    )
    .await?;
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
    info!(
        "Successfully evolved the function, which is now hot-reloaded in-place. Next call to `step` will run the newly compiled Agent code."
    );

    let mut state = GameState::default();

    step(&mut state);
    println!("state: {state:?}");
    assert_ne!(state, GameState::default(), "Game state must have evolved.");

    Ok(())
}

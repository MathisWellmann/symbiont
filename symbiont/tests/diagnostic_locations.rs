// SPDX-License-Identifier: MPL-2.0
//! The candidate is the byte-for-byte prefix of the generated `lib.rs`, so
//! every location in a compiler diagnostic is a location in the code block
//! the agent wrote: line 3 of the diagnostic is line 3 of the block.
//!
//! One test per binary: [`symbiont::Runtime`] is a process-wide singleton.
#![expect(
    unused_crate_dependencies,
    reason = "Integration tests don't use them all"
)]

mod common;

use common::{
    ScriptedAgent,
    Turn,
};
use symbiont::{
    Profile,
    Runtime,
};

#[tokio::test]
#[cfg_attr(
    miri,
    ignore = "compiles and dlopens dylibs, which Miri does not support"
)]
async fn compiler_locations_point_into_the_agents_own_code_block() {
    symbiont::evolvable! {
        fn diag_step(counter: &mut usize) {
            *counter += 1;
        }
    };
    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init runtime");

    // The type error sits on line 3, column 24 of the block as the agent
    // wrote it. A private `fn` without any attribute is enough.
    let broken = "fn diag_step(counter: &mut usize) {\n\
                  \x20   let scale = 2;\n\
                  \x20   let wrong: usize = \"not a usize\";\n\
                  \x20   *counter += wrong * scale;\n\
                  }";
    let agent = ScriptedAgent::new([
        Turn::reply(&format!("```rust\n{broken}\n```")),
        Turn::reply("```rust\nfn diag_step(counter: &mut usize) { *counter += 9; }\n```"),
    ]);

    rt.evolve(&agent, "Implement the function. Code only.")
        .await
        .expect("evolution should succeed after one self-healing retry");

    let retry_prompt = agent.prompt(1);
    assert!(
        retry_prompt.contains("--> line 3:24"),
        "the diagnostic must locate the error on the agent's own line 3, got: {retry_prompt}"
    );
    assert!(
        !retry_prompt.contains("src/lib.rs"),
        "the file name means nothing to the agent, got: {retry_prompt}"
    );
    assert!(
        retry_prompt.contains("[E1]\nerror[E0308]"),
        "errors are numbered for reference, got: {retry_prompt}"
    );
    assert!(
        !retry_prompt.contains("__symbiont"),
        "harness glue must not leak into the diagnostics, got: {retry_prompt}"
    );
    assert!(
        !retry_prompt.contains("aborting due to") && !retry_prompt.contains("could not compile"),
        "cargo's summary lines carry nothing to act on, got: {retry_prompt}"
    );

    // The registered source is the block as written, byte for byte.
    assert_eq!(
        rt.current_code(),
        "fn diag_step(counter: &mut usize) { *counter += 9; }"
    );
    let mut counter = 0;
    diag_step(&mut counter);
    assert_eq!(counter, 9);
}

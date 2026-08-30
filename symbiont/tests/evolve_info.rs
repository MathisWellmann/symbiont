// SPDX-License-Identifier: MPL-2.0
//! `EvolveInfo` reports the token usage of every LLM run an evolve call made.
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
use rig_core::completion::Usage;
use symbiont::{
    Profile,
    Runtime,
};

const BASE_PROMPT: &str = "Implement the function. Code only.";

fn usage(input: u64, output: u64) -> Usage {
    let mut usage = Usage::new();
    usage.input_tokens = input;
    usage.output_tokens = output;
    usage.total_tokens = input + output;
    usage
}

#[tokio::test]
#[cfg_attr(
    miri,
    ignore = "compiles and dlopens dylibs, which Miri does not support"
)]
async fn evolve_reports_usage_including_rejected_attempts() {
    symbiont::evolvable! {
        fn ei_step(counter: &mut usize) {
            *counter += 1;
        }
    };
    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init runtime");

    let agent = ScriptedAgent::new([
        // Attempt 1: no code block — rejected, but its tokens are spent.
        Turn::reply_with_usage("I will write the function shortly.", usage(1_000, 50)),
        // Attempt 2: valid code — the run that wins.
        Turn::reply_with_usage(
            "```rust\npub fn ei_step(counter: &mut usize) { *counter += 2; }\n```",
            usage(1_500, 200),
        ),
    ]);

    let info = rt
        .evolve(&agent, BASE_PROMPT)
        .await
        .expect("evolution should succeed after one self-healing retry");

    assert_eq!(agent.calls(), 2, "one rejected attempt before the winner");
    assert_eq!(
        info.usage(),
        usage(2_500, 250),
        "usage must cover the rejected attempt as well as the winner"
    );
}

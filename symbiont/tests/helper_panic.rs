#![expect(
    unused_crate_dependencies,
    reason = "Integration tests don't use them all"
)]

//! Only exported entry points are wrapped in `catch_unwind`, so a panic in a
//! generated *helper* has to unwind up to the exported function before it is
//! caught. Verify that it is still reported through the host protocol, that
//! the aborted call yields the `Default` placeholder, and that helpers may
//! return types that do not implement `Default`.

use rig_core::{
    completion::{
        PromptError,
        Usage,
    },
    message::Message,
};
use symbiont::{
    AgentRun,
    EvolutionAgent,
    Profile,
    Runtime,
};

#[tokio::test]
#[cfg_attr(
    miri,
    ignore = "compiles and dlopens dylibs, which Miri does not support"
)]
async fn panic_in_helper_is_caught_at_the_entry_point() {
    symbiont::evolvable! {
        /// Pick the larger element, or 0 when out of range.
        fn pick(data: &[usize], idx: usize) -> usize {
            let _ = idx;
            data.len()
        }
    };
    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init");

    let agent = MockAgent;
    rt.evolve(&agent, "irrelevant").await.expect("Can evolve");

    // In range: the helper's non-`Default` return type compiles and works.
    assert_eq!(pick(&[7, 3], 0), 7);
    assert_eq!(rt.take_panic(), None);

    // Out of range: the helper panics, the unwind reaches the exported
    // function's wrapper, and the call returns the placeholder.
    assert_eq!(pick(&[7, 3], 9), 0);
    let msg = rt.take_panic().expect("panic is reported to the host");
    assert!(msg.contains("index out of bounds"), "message: {msg}");
}

struct MockAgent;

/// The helper returns `Ordering`, which has no `Default` impl: wrapping it in
/// `catch_unwind` would not have compiled.
const MOCK_LLM_REPLY: &str = "```rust
pub fn pick(data: &[usize], idx: usize) -> usize {
    match order(data, idx) {
        std::cmp::Ordering::Less => data[idx],
        _ => data[0],
    }
}

fn order(data: &[usize], idx: usize) -> std::cmp::Ordering {
    data[0].cmp(&data[idx])
}
```";

impl EvolutionAgent for MockAgent {
    async fn run(&self, prompt: &str, _history: Vec<Message>) -> Result<AgentRun, PromptError> {
        Ok(AgentRun {
            output: MOCK_LLM_REPLY.to_string(),
            new_messages: vec![Message::user(prompt), Message::assistant(MOCK_LLM_REPLY)],
            usage: Usage::new(),
        })
    }
}

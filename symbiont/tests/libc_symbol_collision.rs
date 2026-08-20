#![expect(
    unused_crate_dependencies,
    reason = "Integration tests don't use them all"
)]

//! Generated helper functions must never be exported from the dylib.
//!
//! An exported (`#[unsafe(no_mangle)]`) symbol in an ELF dylib has default
//! visibility and is preemptible, so the dylib's *own* calls to it go through
//! the GOT and the loader resolves them against the global scope — the host
//! executable and everything loaded with it, libc included — before the dylib
//! itself. A generated helper named `qsort` would therefore hijack the call to
//! libc's `qsort`, which reinterprets the arguments as
//! `(base, nmemb, size, compar)` and jumps to `compar` — a segfault that takes
//! the whole host process down.
//!
//! This test evolves exactly that shape of code and calls it.

use rig_agent::completion::PromptError;
use rig_core::{
    completion::Usage,
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
async fn helper_named_like_libc_symbol_does_not_hijack_the_call() {
    symbiont::evolvable! {
        /// Sort the first `len` elements ascending.
        fn sort(data: &mut [f64], len: usize) {
            let _ = (data, len);
        }
    };
    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init");

    let agent = MockAgent;
    rt.evolve(&agent, "irrelevant").await.expect("Can evolve");

    let mut data = [3.0, 1.0, 2.0];
    sort(&mut data, 3);
    assert_eq!(rt.take_panic(), None);
    assert_eq!(data, [1.0, 2.0, 3.0]);
}

struct MockAgent;

/// A canned reply whose helper collides with libc's `qsort`. The argument
/// layout matches libc's `qsort(base, nmemb, size, compar)` closely enough
/// that a hijacked call jumps to `hi` as a function pointer.
const MOCK_LLM_REPLY: &str = "```rust
pub fn sort(data: &mut [f64], len: usize) {
    qsort(data, 0, len);
}

pub fn qsort(data: &mut [f64], lo: usize, hi: usize) {
    for i in lo..hi {
        for j in lo..hi - 1 - (i - lo) {
            if data[j] > data[j + 1] {
                data.swap(j, j + 1);
            }
        }
    }
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

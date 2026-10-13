// SPDX-License-Identifier: MPL-2.0
//! `type State`: a value the dylib defines and the host owns, that lives
//! between calls, belongs to one revision, and pins that revision.
//!
//! One test per binary: [`symbiont::Runtime`] is a process-wide singleton.

mod common;

use common::{
    ScriptedAgent,
    Turn,
};
use symbiont::{
    Error,
    Profile,
    Revision,
    Runtime,
    Unloaded,
};

/// A revision whose state counts calls and remembers the last input.
const COUNTING: &str = "```rust
#[derive(Default)]
struct State {
    calls: usize,
    seen: Vec<usize>,
}

impl Drop for State {
    fn drop(&mut self) {
        // Exercise the dylib's drop glue with heap data.
        self.seen.clear();
    }
}

fn state_step(state: &mut State, input: usize) -> usize {
    state.calls += 1;
    state.seen.push(input);
    state.calls * 100 + state.seen.iter().sum::<usize>()
}

fn state_peek(state: &State) -> usize {
    state.calls
}
```";

/// A revision whose `State` has another layout and a panicking `Default`.
const PANICKING_DEFAULT: &str = "```rust
struct State {
    big: [u64; 8],
}

impl Default for State {
    fn default() -> Self {
        panic!(\"no default today\")
    }
}

fn state_step(state: &mut State, input: usize) -> usize {
    state.big[0] += 1;
    input
}

fn state_peek(state: &State) -> usize {
    usize::try_from(state.big[0]).unwrap_or_default()
}
```";

#[tokio::test]
#[cfg_attr(
    miri,
    ignore = "compiles and dlopens dylibs, which Miri does not support"
)]
#[tracing_test::traced_test]
async fn state_lives_between_calls_and_pins_its_revision() {
    symbiont::evolvable! {
        type State: Default + Send + 'static;

        /// The initial state: a counter only.
        #[derive(Default)]
        struct State {
            calls: usize,
        }

        fn state_step(state: &mut State, input: usize) -> usize {
            state.calls += 1;
            state.calls + input
        }

        fn state_peek(state: &State) -> usize {
            state.calls
        }
    };

    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init");

    // -- The initial revision carries the macro's `State` ----------------------

    let initial_source = rt.state_source().expect("the block declares a state");
    assert_eq!(
        initial_source.to_string(),
        "/// The initial state: a counter only.\n#[derive(Default)]\nstruct State {\n    calls: usize,\n}\n"
    );
    let code = rt.current_code();
    assert!(
        code.starts_with(
            "/// The initial state: a counter only.\n#[derive(Default)]\nstruct State {"
        )
    );
    assert!(code.contains("fn state_step(state: &mut State, input: usize) -> usize"));

    let mut initial = new_state(Revision::INITIAL).expect("initial state");
    assert_eq!(initial.revision(), Revision::INITIAL);
    assert_eq!(state_step(&mut initial, 10), 11);
    assert_eq!(state_step(&mut initial, 10), 12);
    assert_eq!(state_peek(&initial), 2);
    assert!(rt.take_panic().is_none());

    // Two states of one revision are independent.
    let mut other = rt.new_state(Revision::INITIAL).expect("second state");
    assert_eq!(state_step(&mut other, 0), 1);
    assert_eq!(state_peek(&initial), 2);

    // -- An evolved revision defines its own `State` ---------------------------

    let agent = ScriptedAgent::new([Turn::reply(COUNTING), Turn::reply(PANICKING_DEFAULT)]);
    let counting = rt
        .evolve(&agent, "count")
        .await
        .expect("Can evolve")
        .revision();
    assert_eq!(counting, Revision::new(1));

    let mut state = new_state(counting).expect("counting state");
    let step = state_step_fn(counting).expect("loaded").get();
    let peek = state_peek_fn(counting).expect("loaded").get();
    assert_eq!(step(&mut state, 1), 101);
    assert_eq!(step(&mut state, 2), 203);
    assert_eq!(step(&mut state, 3), 306);
    assert_eq!(peek(&state), 3);
    // Dispatch reaches the same code while `counting` is active.
    assert_eq!(state_step(&mut state, 4), 410);

    // The initial revision's state still works through its own handle.
    let initial_step = state_step_fn(Revision::INITIAL).expect("loaded").get();
    assert_eq!(initial_step(&mut initial, 0), 3);

    // States move to other threads.
    let state = std::thread::spawn(move || {
        assert_eq!(step(&mut state, 0), 510);
        state
    })
    .join()
    .expect("thread does not panic");

    // -- A panicking `State::default()` is an error, not a crash ---------------

    let panicking = rt
        .evolve(&agent, "panic")
        .await
        .expect("Can evolve")
        .revision();
    let err = rt.new_state(panicking).expect_err("Default panics");
    assert!(
        matches!(&err, Error::StateDefaultPanicked { revision, message }
            if *revision == panicking && message.contains("no default today")),
        "{err:?}"
    );

    // -- Unknown revisions -----------------------------------------------------

    let err = rt.new_state(Revision::new(99)).expect_err("unknown");
    assert!(matches!(err, Error::UnknownRevision { .. }), "{err:?}");

    // -- A state pins its revision -----------------------------------------------

    rt.activate_revision(Revision::INITIAL)
        .expect("Can activate");
    drop(other);
    // SAFETY: no dispatch call is in flight, and no bare pointer of
    // `counting` is called after this point.
    let outcome = unsafe { rt.unload_revision(counting) }.expect("Can unload");
    assert_eq!(
        outcome,
        Unloaded::Pinned { handles: 1 },
        "the state pins it"
    );
    assert!(!rt.is_loaded(counting));
    assert!(matches!(
        rt.new_state(counting),
        Err(Error::RevisionUnloaded { .. })
    ));
    // The pinned dylib still runs the state's drop glue, then unmaps.
    drop(state);

    let outcome = unsafe { rt.unload_revision(panicking) }.expect("Can unload");
    assert_eq!(outcome, Unloaded::Now);

    assert_eq!(state_step(&mut initial, 0), 4);
    drop(initial);
}

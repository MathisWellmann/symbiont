// SPDX-License-Identifier: MPL-2.0
// The state protocol, injected verbatim into every generated dylib whose
// `evolvable!` block declares `type State` (see `crate::state`, which pulls
// this file in via `include_str!`).
//
// A standalone source file rather than a string literal for the same reason
// as `panic_preamble.rs`: the tests `include!` the exact code that ships
// inside generated dylibs and run it under Miri.
//
// The dylib owns the agent's `State` value: it allocates it, it frees it, and
// its code is the only code that knows its layout. The host holds a
// `symbiont::State`, whose `#[repr(C)]` prefix is `Handle` below. The host
// never touches the value; it only passes the handle back in.
//
// NOTE: keep this file free of inner attributes and `//!` doc comments;
// `include!` does not accept them.

/// The prefix of the host's `symbiont::State` that the dylib reads.
///
/// Field order and types must match `symbiont::State` exactly; a test in
/// `crate::state` checks the offsets.
#[repr(C)]
pub struct Handle {
    /// The `Box<State>` that `new` leaked, as a thin pointer.
    data: *mut (),
    /// The `tag()` of the dylib that created `data`.
    tag: usize,
    /// The revision id of that dylib, for the mismatch message only.
    revision: u64,
}

/// Its address identifies this copy of the dylib: every mapped dylib has its
/// own instance, and a live `symbiont::State` pins the dylib that created it,
/// so two values with the same tag always come from the same dylib.
static __SYMBIONT_STATE_TAG: u8 = 0;

/// The tag of this dylib. See `__SYMBIONT_STATE_TAG`.
pub(crate) fn tag() -> usize {
    (&raw const __SYMBIONT_STATE_TAG).addr()
}

/// A fresh `S::default()` on this dylib's heap, as a thin pointer.
pub(crate) fn new<S: Default>() -> *mut () {
    Box::into_raw(Box::new(S::default())).cast::<()>()
}

/// Free a value that `new::<S>` returned.
///
/// # Safety
///
/// `data` must come from `new::<S>` of this dylib, and must not be used
/// again afterwards.
pub(crate) unsafe fn drop_state<S>(data: *mut ()) {
    debug_assert!(!data.is_null());
    // SAFETY: `data` is a leaked `Box<S>` of this dylib (caller contract), so
    // this allocator freed it and `S` is its type.
    drop(unsafe { Box::from_raw(data.cast::<S>()) });
}

/// Whether `handle` holds a value of this dylib.
pub(crate) fn is_own(handle: &Handle) -> bool {
    handle.tag == tag()
}

/// The value behind `handle`, borrowed mutably.
///
/// Aborts the process if `handle` holds the state of another revision: its
/// type, and therefore its layout, may differ, and any access would be
/// undefined behaviour. That is a bug in the host, not in the agent's code,
/// so it is not reported as a panic of the agent.
///
/// # Safety
///
/// `handle` must be the prefix of a live `symbiont::State`, and `S` must be
/// the type `new` was called with in the dylib whose tag `handle` carries.
pub(crate) unsafe fn state_mut<'a, S>(handle: &'a mut Handle, function: &str) -> &'a mut S {
    if !is_own(handle) {
        mismatch(handle, function);
    }
    debug_assert!(!handle.data.is_null());
    // SAFETY: the tag matches, so `data` is a live `Box<S>` of this dylib
    // (caller contract). The value is owned by the handle, which is
    // borrowed exclusively for `'a`.
    unsafe { &mut *handle.data.cast::<S>() }
}

/// The value behind `handle`, borrowed shared. See `state_mut`.
///
/// # Safety
///
/// As `state_mut`.
pub(crate) unsafe fn state_ref<'a, S>(handle: &'a Handle, function: &str) -> &'a S {
    if !is_own(handle) {
        mismatch(handle, function);
    }
    debug_assert!(!handle.data.is_null());
    // SAFETY: as in `state_mut`; the handle is borrowed shared for `'a`, so
    // nothing mutates the value meanwhile.
    unsafe { &*handle.data.cast::<S>() }
}

#[cold]
#[inline(never)]
fn mismatch(handle: &Handle, function: &str) -> ! {
    eprintln!(
        "symbiont: the State of revision {} was passed to `{function}` of another revision. \
         A State only works with the revision that created it. Aborting.",
        handle.revision
    );
    std::process::abort()
}

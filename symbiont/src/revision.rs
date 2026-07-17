// SPDX-License-Identifier: MPL-2.0
//! The revision registry types: every dylib that was successfully compiled,
//! loaded, and hot-swapped is retained for the lifetime of the process
//! (keep-all), so earlier evolutions stay callable later without parsing or
//! compiling anything again.

use std::{
    ffi::CString,
    fmt,
    sync::{
        Arc,
        atomic::Ordering,
    },
};

use libloading::{
    Library,
    Symbol,
};

use crate::{
    Error,
    EvolvableDecl,
    Result,
    runtime::TAKE_PANIC_PTR,
};

/// Identifier of a successfully compiled, loaded, and registered dylib revision.
///
/// Revision ids are dense: [`Revision::INITIAL`] (id `0`) is the initial build
/// compiled from the `evolvable!` default bodies, and every successful
/// [`crate::Runtime::evolve`] registers the next id. All registered revisions
/// stay loaded for the lifetime of the process, so any of them can be pointed
/// at again later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Revision(u64);

impl Revision {
    /// The initial revision, compiled from the default function bodies
    /// declared in `evolvable!` when [`crate::Runtime::new`] initializes.
    pub const INITIAL: Self = Self(0);

    /// Create a revision id: `0` is the initial build, `n` is the `n`-th
    /// successful evolution.
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// The raw numeric id.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for Revision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

/// A retained, loaded dylib revision: the library handle, its resolved
/// symbols, and the clean source it was compiled from.
pub(crate) struct RevisionEntry {
    /// Keeps the mapped library alive for the lifetime of the runtime
    /// (keep-all policy). Never unloading means the resolved pointers below
    /// stay valid forever, and a call racing a swap executes old but
    /// still-mapped code instead of unmapped pages.
    _library: Library,
    /// Resolved function pointers, parallel to the runtime's `decls` slice.
    fn_ptrs: Box<[*const ()]>,
    /// This revision's `__symbiont_take_panic` symbol.
    take_panic: *const (),
    /// The clean generated source (without panic wrappers or preamble),
    /// suitable for prompts and display.
    source: String,
}

// SAFETY: The raw pointers are symbol addresses inside `_library`, which is
// owned by this entry and never unloaded. They are only ever read and called
// through the fn signatures that were validated at generation time.
unsafe impl Send for RevisionEntry {}
unsafe impl Sync for RevisionEntry {}

impl RevisionEntry {
    /// Resolve all declared symbols plus `__symbiont_take_panic` in `library`
    /// and retain the library together with the resolved pointers.
    ///
    /// # Safety
    ///
    /// The caller must ensure `library` was compiled from source whose
    /// function signatures match `decls`. The runtime guarantees this by
    /// validating generated code against the declarations before compiling.
    pub(crate) unsafe fn resolve(
        library: Library,
        decls: &[EvolvableDecl],
        source: String,
    ) -> Result<Self> {
        let (fn_ptrs, take_panic) = {
            let mut fn_ptrs = Vec::with_capacity(decls.len());
            for decl in decls {
                let c_name = CString::new(decl.name)
                    .expect("function name must not contain interior NUL bytes");
                let sym: Symbol<'_, *const ()> = unsafe {
                    library.get(c_name.as_bytes_with_nul()).map_err(|e| {
                        Error::DylibLoad(format!("symbol '{}' not found: {e}", decl.name))
                    })?
                };
                fn_ptrs.push(*sym);
            }

            let take_panic: Symbol<'_, *const ()> = unsafe {
                library.get(b"__symbiont_take_panic\0").map_err(|e| {
                    Error::DylibLoad(format!("symbol '__symbiont_take_panic' not found: {e}"))
                })?
            };
            (fn_ptrs.into_boxed_slice(), *take_panic)
        };

        Ok(Self {
            _library: library,
            fn_ptrs,
            take_panic,
            source,
        })
    }

    /// Publish this revision's pointers into the per-function dispatch
    /// atomics and the global panic-retrieval pointer, making it the revision
    /// all `evolvable!` wrappers call from now on (atomic stores with
    /// `Release` ordering).
    pub(crate) fn publish(&self, decls: &[EvolvableDecl]) {
        debug_assert_eq!(
            decls.len(),
            self.fn_ptrs.len(),
            "revision was resolved against a different declaration set"
        );
        for (decl, fn_ptr) in decls.iter().zip(&self.fn_ptrs) {
            decl.fn_ptr.store(fn_ptr.cast_mut(), Ordering::Release);
        }
        TAKE_PANIC_PTR.store(self.take_panic.cast_mut(), Ordering::Release);
    }

    /// The clean generated source this revision was compiled from.
    pub(crate) fn source(&self) -> &str {
        &self.source
    }

    /// The resolved pointer of the `idx`-th declared function.
    /// Indices are positions in the runtime's `decls` slice.
    pub(crate) fn fn_ptr_at(&self, idx: usize) -> *const () {
        self.fn_ptrs[idx]
    }

    /// Retrieve and clear the last panic message stored in this revision's
    /// dylib-local panic buffer.
    pub(crate) fn take_panic(&self) -> Option<String> {
        // SAFETY: `take_panic` was resolved from this entry's library, which
        // is still loaded, and matches the exported protocol.
        unsafe { read_panic_buffer(self.take_panic) }
    }
}

/// Call a dylib's `__symbiont_take_panic` through `ptr` and decode the result.
///
/// # Safety
///
/// `ptr` must be null or point to a function with the exported protocol
/// signature `unsafe fn(*mut u8, usize) -> usize`, inside a library that is
/// still loaded.
pub(crate) unsafe fn read_panic_buffer(ptr: *const ()) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    let take_panic: unsafe fn(*mut u8, usize) -> usize = unsafe { std::mem::transmute(ptr) };
    let mut buf = [0u8; 512];
    let len = unsafe { take_panic(buf.as_mut_ptr(), buf.len()) };
    if len == 0 {
        None
    } else {
        Some(String::from_utf8_lossy(&buf[..len]).into_owned())
    }
}

/// A typed handle to one evolvable function of one retained revision.
///
/// Created by the `<name>_fn` accessors that `evolvable!` generates (e.g.
/// `decide_fn(rev)` for an evolvable `fn decide(..)`). The handle pins its
/// revision's dylib via reference counting, so calls through it stay valid for
/// as long as the handle (or a clone of it) lives — independent of which
/// revision is currently active and of any further evolutions.
///
/// Calls through a handle never read the swappable dispatch pointers, so they
/// are exempt from the feedback-loop contract: they may safely run
/// concurrently with [`crate::Runtime::evolve`] and
/// [`crate::Runtime::activate_revision`], and several handles from different
/// revisions may run at the same time (ensembles, tournaments, A/B
/// comparisons).
///
/// # Hot loops
///
/// Fetching a handle takes a registry read plus an `Arc` clone; hoist the
/// bare function pointer once with [`RevisionFn::get`] and the per-call cost
/// is a plain indirect call — no atomic load, lock, or refcount touch:
///
/// ```rust,ignore
/// let handle = decide_fn(best_rev).expect("revision is retained");
/// let f = handle.get();
/// for window in windows {
///     let action = f(window, &state);
/// }
/// if let Some(msg) = handle.take_panic() {
///     // the panic of a handle call lands in ITS revision's buffer,
///     // not in `runtime.take_panic()` (which reads the active revision).
/// }
/// ```
pub struct RevisionFn<F> {
    /// The revision this handle points into.
    revision: Revision,
    /// The typed function pointer resolved from the retained dylib.
    f: F,
    /// Pins the revision entry (and thus the mapped library) alive.
    entry: Arc<RevisionEntry>,
}

impl<F: Clone> Clone for RevisionFn<F> {
    fn clone(&self) -> Self {
        Self {
            revision: self.revision,
            f: self.f.clone(),
            entry: Arc::clone(&self.entry),
        }
    }
}

impl<F> fmt::Debug for RevisionFn<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RevisionFn")
            .field("revision", &self.revision)
            .finish_non_exhaustive()
    }
}

impl<F> RevisionFn<F> {
    /// The revision this handle executes.
    pub fn revision(&self) -> Revision {
        self.revision
    }

    /// Retrieve and clear the last panic message of **this revision's**
    /// dylib-local panic buffer.
    ///
    /// Panics raised by calls through this handle are stored here — not in
    /// [`crate::Runtime::take_panic`], which reads the *active* revision's
    /// buffer. The buffer is shared by all functions of the revision and
    /// holds only the most recent message, so concurrent panicking calls
    /// into the same revision overwrite each other.
    pub fn take_panic(&self) -> Option<String> {
        self.entry.take_panic()
    }
}

impl<F: Copy> RevisionFn<F> {
    /// The bare typed function pointer.
    ///
    /// Hoist it out of hot loops: the returned pointer is a plain `fn` whose
    /// calls carry no dispatch overhead. It is valid for the lifetime of the
    /// process — the registry retains every revision (keep-all), and this
    /// handle additionally pins it.
    pub fn get(&self) -> F {
        self.f
    }
}

impl RevisionFn<*const ()> {
    /// Construct an untyped handle. The caller (the runtime's lookup) must
    /// pass the pointer of the declaration the accessor was generated for.
    pub(crate) fn new_untyped(revision: Revision, f: *const (), entry: Arc<RevisionEntry>) -> Self {
        Self { revision, f, entry }
    }

    /// Cast the untyped symbol pointer to the concrete `fn` type.
    ///
    /// Not part of the public API — only the `evolvable!` expansion calls
    /// this, with the exact signature the runtime validated before compiling
    /// the revision.
    ///
    /// # Safety
    ///
    /// `G` must be a `fn` pointer type that is ABI-compatible with the symbol
    /// this handle was resolved from.
    #[doc(hidden)]
    pub unsafe fn cast<G: Copy>(self) -> RevisionFn<G> {
        debug_assert_eq!(
            size_of::<G>(),
            size_of::<*const ()>(),
            "cast target must be a plain fn pointer"
        );
        RevisionFn {
            revision: self.revision,
            // SAFETY: same size per the assertion; validity is the caller's
            // contract (`G` matches the symbol's real signature).
            f: unsafe { std::mem::transmute_copy::<*const (), G>(&self.f) },
            entry: self.entry,
        }
    }
}

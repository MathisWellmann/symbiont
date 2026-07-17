// SPDX-License-Identifier: MPL-2.0
//! The revision registry types: every dylib that was successfully compiled,
//! loaded, and hot-swapped is retained for the lifetime of the process
//! (keep-all), so earlier evolutions stay callable later without parsing or
//! compiling anything again.

use std::{
    ffi::CString,
    fmt,
    sync::atomic::Ordering,
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
}

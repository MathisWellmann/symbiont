use std::{
    ffi::CString,
    sync::atomic::Ordering,
};

use libloading::{
    Library,
    Symbol,
};

use crate::{
    Error,
    EvolvableDecl,
    runtime::TAKE_PANIC_PTR,
};

/// Look up all declared symbols in `lib` and store their addresses
/// in the corresponding `AtomicPtr` fields of each declaration.
///
/// # Safety
///
/// The caller must ensure `lib` is a valid loaded library containing
/// symbols with signatures matching the de
pub(crate) unsafe fn update_fn_ptrs(lib: &Library, decls: &[EvolvableDecl]) -> crate::Result<()> {
    for decl in decls {
        let c_name =
            CString::new(decl.name).expect("function name must not contain interior NUL bytes");
        let sym: Symbol<*const ()> = unsafe {
            lib.get(c_name.as_bytes_with_nul())
                .map_err(|e| Error::DylibLoad(format!("symbol '{}' not found: {e}", decl.name)))?
        };
        decl.fn_ptr.store(*sym as *mut (), Ordering::Release);
    }

    // Resolve the panic-retrieval symbol.
    let panic_sym: Symbol<*const ()> = unsafe {
        lib.get(b"__symbiont_take_panic\0").map_err(|e| {
            Error::DylibLoad(format!("symbol '__symbiont_take_panic' not found: {e}"))
        })?
    };
    TAKE_PANIC_PTR.store(*panic_sym as *mut (), Ordering::Release);

    Ok(())
}

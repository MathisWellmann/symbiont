// SPDX-License-Identifier: MPL-2.0
// Preamble injected verbatim into every generated dylib (see
// `crate::unwind::PANIC_PREAMBLE`, which pulls this file in via
// `include_str!`).
//
// This is a standalone source file rather than a string literal so the
// exact code that ships inside generated dylibs can also be compiled into
// the test binary (via `include!`) and executed under Miri, giving UB
// coverage for the unsafe panic-buffer protocol below.
//
// NOTE: keep this file free of inner attributes and `//!` doc comments;
// `include!` does not accept them.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

/// Lock-free "a panic message is stored" flag.
///
/// `__symbiont_take_panic` reads it as a fast path so the hot no-panic case
/// never touches the Mutex: hosts poll for panics after every evolvable
/// call, potentially from many threads at once,
/// and a shared Mutex lock per call serializes all of them on one cache
/// line. A read-mostly atomic flag scales instead.
static __SYMBIONT_PANICKED: AtomicBool = AtomicBool::new(false);

/// Fixed-size buffer for the last panic message (512 bytes max).
/// Layout: (panicked: bool, message: [u8; 512], length: usize)
static __SYMBIONT_PANIC: Mutex<(bool, [u8; 512], usize)> =
    Mutex::new((false, [0u8; 512], 0));

pub(crate) fn __symbiont_store_panic(msg: &str) {
    if let Ok(mut guard) = __SYMBIONT_PANIC.lock() {
        let len = msg.len().min(512);
        guard.0 = true;
        guard.1[..len].copy_from_slice(&msg.as_bytes()[..len]);
        guard.2 = len;
        __SYMBIONT_PANICKED.store(true, Ordering::Release);
    }
}

/// Store `msg` only when no message is currently stored.
///
/// Fallback used by the `catch_unwind` wrapper: the panic hook has already
/// recorded the message together with its source location, which the
/// location-less `catch_unwind` payload must not overwrite.
pub(crate) fn __symbiont_store_panic_fallback(msg: &str) {
    if let Ok(mut guard) = __SYMBIONT_PANIC.lock() {
        if guard.0 {
            return;
        }
        let len = msg.len().min(512);
        guard.0 = true;
        guard.1[..len].copy_from_slice(&msg.as_bytes()[..len]);
        guard.2 = len;
        __SYMBIONT_PANICKED.store(true, Ordering::Release);
    }
}

/// Ensures the location-capturing panic hook is installed exactly once.
static __SYMBIONT_HOOK: std::sync::Once = std::sync::Once::new();

/// Install a panic hook that records the panic message together with its
/// source location, then delegates to the previously installed hook.
///
/// The hook runs at panic time, before unwinding reaches `catch_unwind`;
/// it is the only point where `std::panic::Location` is available. The
/// dylib links its own copy of `std`, so this hook only observes panics
/// raised by code compiled into this dylib, never panics of the host.
pub(crate) fn __symbiont_install_panic_hook() {
    __SYMBIONT_HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let msg = if let Some(s) = info.payload().downcast_ref::<&str>() {
                *s
            } else if let Some(s) = info.payload().downcast_ref::<String>() {
                s.as_str()
            } else {
                "unknown panic"
            };
            match info.location() {
                Some(location) => __symbiont_store_panic(&format!("{msg} at {location}")),
                None => __symbiont_store_panic(msg),
            }
            previous(info);
        }));
    });
}

/// Copy the last panic message into `buf` and return its length.
/// Returns 0 if no panic occurred. Clears the stored message.
///
/// The no-panic case is a single atomic load; the Mutex is only locked when
/// a panic message is actually stored, so concurrent hot-loop polling from
/// many threads does not contend.
///
/// # Safety
///
/// `buf` must point to at least `buf_len` writable bytes.
#[unsafe(no_mangle)]
pub unsafe fn __symbiont_take_panic(buf: *mut u8, buf_len: usize) -> usize {
    if !__SYMBIONT_PANICKED.load(Ordering::Acquire) {
        return 0;
    }
    if let Ok(mut guard) = __SYMBIONT_PANIC.lock() {
        if !guard.0 {
            return 0;
        }
        let len = guard.2.min(buf_len);
        unsafe { core::ptr::copy_nonoverlapping(guard.1.as_ptr(), buf, len) };
        guard.0 = false;
        guard.2 = 0;
        __SYMBIONT_PANICKED.store(false, Ordering::Release);
        len
    } else {
        0
    }
}

// SPDX-License-Identifier: MPL-2.0
//! The in-dylib panic protocol.
//!
//! The host binary and the dynamically loaded dylib have separate panic
//! runtimes: a panic that unwinds out of the dylib reaches the host as a
//! foreign exception and aborts the process. Every exported entry point of
//! the dylib therefore runs the agent's code under `catch_unwind` (see
//! [`crate::layout`]) and the panic message travels to the host through the
//! exported `__symbiont_take_panic` symbol defined below.

/// The module of the generated crate that holds [`PANIC_PREAMBLE`].
pub(crate) const PANIC_MODULE: &str = "__symbiont_panic";

/// Preamble injected into every generated dylib.
///
/// Provides a fixed-size panic buffer and an exported `__symbiont_take_panic`
/// symbol so the host can retrieve panic messages without heap allocation
/// crossing the dylib boundary.
///
/// The source lives in `panic_preamble.rs` (as a file rather than a string
/// literal) so the tests below can `include!` the exact same code and run it
/// under Miri to check the unsafe buffer protocol for undefined behaviour.
pub(crate) const PANIC_PREAMBLE: &str = include_str!("panic_preamble.rs");

#[cfg(test)]
mod tests {
    use super::*;

    /// The export wrappers in `layout.rs` call these two functions by name
    /// and the host resolves the third as a symbol. Renaming one in the
    /// preamble must fail here, not in every generated dylib.
    #[test]
    fn preamble_defines_the_functions_the_wrappers_and_the_host_use() {
        for needle in [
            "pub(crate) fn __symbiont_install_panic_hook()",
            "pub(crate) fn __symbiont_store_panic_fallback(msg: &str)",
            "pub unsafe fn __symbiont_take_panic(buf: *mut u8, buf_len: usize) -> usize",
        ] {
            assert!(PANIC_PREAMBLE.contains(needle), "preamble lost `{needle}`");
        }
        assert!(PANIC_MODULE.starts_with("__symbiont_"));
    }

    /// The exact preamble source that ships inside every generated dylib,
    /// compiled into this test binary so the unsafe panic-buffer protocol
    /// can be executed directly — in particular under Miri, which flags
    /// undefined behaviour in it.
    #[allow(
        unused,
        unreachable_pub,
        reason = "the preamble is compiled verbatim; in a dylib crate root its `pub` items are reachable"
    )]
    mod preamble {
        include!("panic_preamble.rs");
    }

    /// Serializes the protocol tests: the preamble's panic buffer and
    /// "panicked" flag are process-global statics shared by all tests in
    /// this binary.
    static PROTOCOL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn protocol_lock() -> std::sync::MutexGuard<'static, ()> {
        PROTOCOL_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Install a silent base hook (so intentional panics don't spam test
    /// output), then the preamble's location-capturing hook on top of it.
    fn install_hooks_once() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            std::panic::set_hook(Box::new(|_| {}));
            preamble::__symbiont_install_panic_hook();
        });
    }

    /// Clear any message left behind by another test.
    fn drain_panic_buffer() {
        let mut buf = [0u8; 512];
        // SAFETY: `buf` is 512 writable bytes, matching `buf_len`.
        unsafe { preamble::__symbiont_take_panic(buf.as_mut_ptr(), buf.len()) };
    }

    /// `__symbiont_take_panic` cast to the erased pointer type the host
    /// stores in the dispatch atomics and passes to `read_panic_buffer`.
    fn take_panic_ptr() -> *const () {
        preamble::__symbiont_take_panic as unsafe fn(*mut u8, usize) -> usize as *const ()
    }

    #[test]
    fn take_panic_returns_zero_without_panic() {
        let _guard = protocol_lock();
        drain_panic_buffer();
        let mut buf = [0u8; 64];
        // SAFETY: `buf` is 64 writable bytes, matching `buf_len`.
        let len = unsafe { preamble::__symbiont_take_panic(buf.as_mut_ptr(), buf.len()) };
        assert_eq!(len, 0);
    }

    #[test]
    fn panic_message_roundtrips_through_host_protocol() {
        let _guard = protocol_lock();
        install_hooks_once();
        drain_panic_buffer();

        let _ = std::panic::catch_unwind(|| panic!("boom {}", 42));

        // Decode through the host-side path, exercising the fn-pointer
        // transmute, the uninitialized buffer, and the raw-parts slice.
        // SAFETY: the pointer refers to a function with the exported
        // protocol signature.
        let msg = unsafe { crate::revision::read_panic_buffer(take_panic_ptr()) }
            .expect("panic message must be stored by the hook");
        assert!(msg.contains("boom 42"), "message: {msg}");
        assert!(msg.contains("unwind.rs"), "location missing: {msg}");

        // Taking the message clears the buffer.
        // SAFETY: same as above.
        assert!(unsafe { crate::revision::read_panic_buffer(take_panic_ptr()) }.is_none());
    }

    #[test]
    fn long_panic_messages_truncate_at_buffer_size() {
        let _guard = protocol_lock();
        install_hooks_once();
        drain_panic_buffer();

        let long = "x".repeat(600);
        let _ = std::panic::catch_unwind(|| std::panic::panic_any(long));

        // SAFETY: the pointer refers to a function with the exported
        // protocol signature.
        let msg = unsafe { crate::revision::read_panic_buffer(take_panic_ptr()) }
            .expect("panic message must be stored by the hook");
        assert_eq!(msg.len(), 512);
        assert!(msg.bytes().all(|b| b == b'x'));
    }

    #[test]
    fn take_panic_clamps_to_small_caller_buffer() {
        let _guard = protocol_lock();
        drain_panic_buffer();

        preamble::__symbiont_store_panic("this is a longer message");
        let mut buf = [0u8; 8];
        // SAFETY: `buf` is 8 writable bytes, matching `buf_len`; Miri
        // verifies no out-of-bounds write occurs.
        let len = unsafe { preamble::__symbiont_take_panic(buf.as_mut_ptr(), buf.len()) };
        assert_eq!(len, 8);
        assert_eq!(&buf, b"this is ");
    }

    #[test]
    fn fallback_does_not_overwrite_hook_message() {
        let _guard = protocol_lock();
        drain_panic_buffer();

        preamble::__symbiont_store_panic("primary");
        preamble::__symbiont_store_panic_fallback("secondary");
        let mut buf = [0u8; 512];
        // SAFETY: `buf` is 512 writable bytes, matching `buf_len`.
        let len = unsafe { preamble::__symbiont_take_panic(buf.as_mut_ptr(), buf.len()) };
        assert_eq!(&buf[..len], b"primary");

        // With the buffer empty, the fallback does store.
        preamble::__symbiont_store_panic_fallback("secondary");
        // SAFETY: same as above.
        let len = unsafe { preamble::__symbiont_take_panic(buf.as_mut_ptr(), buf.len()) };
        assert_eq!(&buf[..len], b"secondary");
    }

    #[test]
    fn read_panic_buffer_null_ptr_is_none() {
        // SAFETY: `read_panic_buffer` explicitly permits a null pointer.
        assert!(unsafe { crate::revision::read_panic_buffer(std::ptr::null()) }.is_none());
    }
}

// SPDX-License-Identifier: MPL-2.0

/// Wrap each function body in `catch_unwind` so panics are caught inside the
/// dylib and never unwind across the `dlopen` boundary.
///
/// For a function like:
/// ```ignore
/// pub fn sort(data: &mut [f64], len: usize) { /* body */ }
/// ```
/// this produces:
/// ```ignore
/// pub fn sort(data: &mut [f64], len: usize) {
///     __symbiont_install_panic_hook();
///     match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| { /* body */ })) {
///         Ok(v) => v,
///         Err(e) => {
///             let msg = if let Some(s) = e.downcast_ref::<&str>() { *s }
///                       else if let Some(s) = e.downcast_ref::<String>() { s.as_str() }
///                       else { "unknown panic" };
///             __symbiont_store_panic_fallback(msg);
///             Default::default()
///         }
///     }
/// }
/// ```
///
/// The `Err` arm substitutes `Default::default()` as a placeholder return
/// value — safe for every type, unlike a zeroed value, which is undefined
/// behaviour for types like `String` or `&T`. The `evolvable!` macro
/// enforces at declaration time that every return type implements
/// [`Default`], so the wrapped code always compiles; hosts detect the
/// panic via `Runtime::take_panic` and discard the placeholder.
///
/// The panic *message with its source location* is recorded by the panic hook
/// installed via `__symbiont_install_panic_hook` — the hook runs at panic
/// time, which is the only point where `std::panic::Location` is available.
/// The `Err` arm only stores the location-less payload as a fallback for
/// panics that somehow bypassed the hook.
///
/// The Mutex is only touched when a panic actually fires. On the happy
/// path `catch_unwind` is zero-cost (DWARF-based landing pads).
pub(crate) fn wrap_bodies_in_catch_unwind(file: &mut syn::File) {
    for item in &mut file.items {
        if let syn::Item::Fn(item_fn) = item {
            let original_body = &item_fn.block;
            let wrapped: syn::Block = syn::parse_quote!({
                __symbiont_install_panic_hook();
                match ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(
                    || #original_body,
                )) {
                    Ok(__symbiont_val) => __symbiont_val,
                    Err(__symbiont_err) => {
                        let __symbiont_msg =
                            if let Some(s) = __symbiont_err.downcast_ref::<&str>() {
                                *s
                            } else if let Some(s) = __symbiont_err.downcast_ref::<String>() {
                                s.as_str()
                            } else {
                                "unknown panic"
                            };
                        __symbiont_store_panic_fallback(__symbiont_msg);
                        ::core::default::Default::default()
                    }
                }
            });
            *item_fn.block = wrapped;
        }
    }
}

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

    #[test]
    fn test_wrap_bodies_in_catch_unwind() {
        let mut file: syn::File = syn::parse_str(
            "
            fn step(counter: &mut usize) {
                panic!()
            }
            ",
        )
        .expect("Can parse");
        wrap_bodies_in_catch_unwind(&mut file);
        assert_eq!(
            &prettyplease::unparse(&file),
            r#"fn step(counter: &mut usize) {
    __symbiont_install_panic_hook();
    match ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| { panic!() })) {
        Ok(__symbiont_val) => __symbiont_val,
        Err(__symbiont_err) => {
            let __symbiont_msg = if let Some(s) = __symbiont_err.downcast_ref::<&str>() {
                *s
            } else if let Some(s) = __symbiont_err.downcast_ref::<String>() {
                s.as_str()
            } else {
                "unknown panic"
            };
            __symbiont_store_panic_fallback(__symbiont_msg);
            ::core::default::Default::default()
        }
    }
}
"#
        );
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

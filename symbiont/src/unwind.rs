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
///     match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| { /* body */ })) {
///         Ok(v) => v,
///         Err(e) => {
///             let msg = if let Some(s) = e.downcast_ref::<&str>() { *s }
///                       else if let Some(s) = e.downcast_ref::<String>() { s.as_str() }
///                       else { "unknown panic" };
///             __symbiont_store_panic(msg);
///             unsafe { core::mem::zeroed() }
///         }
///     }
/// }
/// ```
///
/// The Mutex is only touched when a panic actually fires. On the happy
/// path `catch_unwind` is zero-cost (DWARF-based landing pads).
pub(crate) fn wrap_bodies_in_catch_unwind(file: &mut syn::File) {
    for item in &mut file.items {
        if let syn::Item::Fn(item_fn) = item {
            let original_body = &item_fn.block;
            let wrapped: syn::Block = syn::parse_quote!({
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
                        __symbiont_store_panic(__symbiont_msg);
                        unsafe { ::core::mem::zeroed() }
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
#[allow(clippy::needless_raw_strings, reason = "contains #[unsafe(no_mangle)]")]
pub(crate) const PANIC_PREAMBLE: &str = r#"
use std::sync::Mutex;

/// Fixed-size buffer for the last panic message (512 bytes max).
/// Layout: (panicked: bool, message: [u8; 512], length: usize)
static __SYMBIONT_PANIC: Mutex<(bool, [u8; 512], usize)> =
    Mutex::new((false, [0u8; 512], 0));

fn __symbiont_store_panic(msg: &str) {
    if let Ok(mut guard) = __SYMBIONT_PANIC.lock() {
        let len = msg.len().min(512);
        guard.0 = true;
        guard.1[..len].copy_from_slice(&msg.as_bytes()[..len]);
        guard.2 = len;
    }
}

/// Copy the last panic message into `buf` and return its length.
/// Returns 0 if no panic occurred. Clears the stored message.
///
/// # Safety
///
/// `buf` must point to at least `buf_len` writable bytes.
#[unsafe(no_mangle)]
pub unsafe fn __symbiont_take_panic(buf: *mut u8, buf_len: usize) -> usize {
    if let Ok(mut guard) = __SYMBIONT_PANIC.lock() {
        if !guard.0 {
            return 0;
        }
        let len = guard.2.min(buf_len);
        unsafe { core::ptr::copy_nonoverlapping(guard.1.as_ptr(), buf, len) };
        guard.0 = false;
        guard.2 = 0;
        len
    } else {
        0
    }
}
"#;

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
            __symbiont_store_panic(__symbiont_msg);
            unsafe { ::core::mem::zeroed() }
        }
    }
}
"#
        );
    }
}

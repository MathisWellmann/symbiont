// ---------- debug-mode call counter ----------
//
// Tracks in-flight evolvable function calls so that `evolve()` can assert
// none are running. Compiled away entirely in release builds — zero overhead.

use std::sync::atomic::Ordering;

pub(crate) static IN_FLIGHT_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// RAII guard that decrements the in-flight call counter on drop.
/// Only exists in debug builds.
pub struct CallGuard;

#[cfg(debug_assertions)]
impl Drop for CallGuard {
    fn drop(&mut self) {
        IN_FLIGHT_CALLS.fetch_sub(1, Ordering::Release);
    }
}

/// Increment the in-flight call counter and return a guard that decrements
/// it on drop (including during unwind). Debug builds only.
pub fn enter_call() -> CallGuard {
    IN_FLIGHT_CALLS.fetch_add(1, Ordering::Acquire);
    CallGuard
}

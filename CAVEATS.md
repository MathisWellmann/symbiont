# Caveats

This document describes the caveats and limitations of
symbiont's hot-reloading dylib approach.

The harness compiles LLM-generated code into a dynamic library
(`.so` / `.dylib` / `.dll`), loads it via `libloading`, and swaps
function pointers at runtime. Because Rust was not designed for
dynamic loading, this introduces strict limitations.

## Static variables reset on reload

Any `static` variable inside the reloaded dylib is re-initialized
on every reload. If the evolvable function relies on persistent
state across calls, that state is lost when the function evolves.
The harness forbids this by design: all state is owned by the
host binary and passed into evolvable functions via arguments.

## Dangling pointers across reloads

If the host holds a reference or pointer to data allocated inside
the dylib, reloading unmaps the old code and data pages, leaving
the pointer dangling. The harness prevents this by requiring that
evolvable function signatures use only caller-owned memory
(`&mut [f64]`, `&mut usize`, etc.) — no allocations escape the
dylib boundary.

## Compile times

Each evolution round compiles a Rust dylib. The generated crate
has zero dependencies (only `std`), so incremental builds are
fast (~100-200 ms). Adding dependencies to the generated crate
would increase compilation time and break the fast feedback loop.
Keep evolvable function bodies self-contained.
Might change in the future see [TODO.md](TODO.md)

## Destructors bypassed on unload

When the OS unloads the `.so`, Rust destructors are not run. Any
resources created inside the dylib (open files, background
threads, heap allocations) will leak. In practice this is not an
issue because the generated dylib only contains pure functions
operating on caller-provided memory. Avoid spawning threads or
opening files inside evolvable functions.

## Type layout and ABI

Rust has no stable ABI. The host binary and dylib must be compiled
with the same `rustc` version to guarantee matching calling
conventions and memory layouts. The harness ensures this by
compiling the dylib on the same machine with the same toolchain.

Evolvable function signatures are limited to primitive and `std`
types (`usize`, `f64`, `&mut [f64]`, etc.). Custom structs across
the boundary would require matching `#[repr(C)]` layouts and are
not currently supported.

## String types

`String` and `&str` rely on Rust's internal allocator and fat
pointers. Passing them across a dynamic library boundary is
fragile if the two sides use different allocator instances. The
harness avoids this by compiling both sides with the exact same
compiler and linking against the same `std`.

## Dependencies

By default the generated dylib has no dependencies beyond `std`.
External crate support may be added in the future, but introduces
additional constraints:

- The host and dylib compile separate copies of any shared
  dependency. Types from a dependency used on both sides must be
  the exact same version, compiled with the same features and
  compiler, or their memory layouts may diverge silently.
- Heap allocations made by a dependency inside the dylib use the
  dylib's allocator instance. Passing owned types (`Vec`,
  `String`, `Box`) across the boundary is only safe when both
  sides share the same allocator — guaranteed when compiled with
  the same toolchain, but fragile under any mismatch.
- Evolvable function signatures should still prefer primitive and
  `std` types at the boundary. Use dependency types internally
  within the function body, not in the signature.

## Infinite loops in generated code

LLM-generated code may contain infinite loops (e.g. a sort with
a buggy termination condition). The harness catches panics inside
the dylib via `catch_unwind`, but an infinite loop never panics —
it hangs the calling thread indefinitely.

The harness does **not** detect this automatically. It is the
caller's responsibility to implement timeout detection, for
example by running the evolvable function in a separate thread
with `recv_timeout`. If a timeout fires, the abandoned thread
continues executing in the background — the old dylib must be
kept alive (not dropped on reload) to avoid unmapping code pages
that the thread is still running. Callers should account for this
when designing their evaluation loops.

## Panic runtime isolation

The host binary and the dynamically loaded dylib have separate
panic runtimes. A panic originating inside the dylib cannot be
caught by `std::panic::catch_unwind` in the host — the host sees
it as a "foreign exception" and aborts. The harness handles this
by wrapping every evolvable function body in `catch_unwind`
*inside the dylib* and exposing the panic message through an
exported symbol (`__symbiont_take_panic`). Use
`symbiont::catch_panic` to retrieve panic messages after each
call.

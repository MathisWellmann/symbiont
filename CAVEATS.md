# Caveats
This document describes the caveats and limitations of the hot-reloading dylib approach,
which was previously described in this excellent blog post: https://robert.kra.hn/posts/hot-reloading-rust/

The `hot-lib-reloader` crate works by compiling a sub-crate as a dynamic library `.dylib` and loading it via `libloading`,
swappinig function pointers in the process. Because Rust was not originally designed for dynamic loading, this
introduces strict limitations.

## Memory Safety & State
- Any static variable inside the reloaded library crate will be reset to its initial state upon every reload
If the main app relies on a static inside the lib (like a global configuration or a counter), it will suddenly point to a completely different memory address or reset to zero.
This is forbidden and enforced by the harness.
- Dangling Pointers Across Reloads:
If your main application holds a reference or pointer to data allocated inside the hot-reloaded library,
and the library is reloaded, that pointer becomes dangling.
Using it will cause a segfault or memory corruption.
The Agent harness ensures this cannot happen by explictly passing in mutable memory where required.

## Compile times:
- If the hot-reloadable library depends on heavy crates (like a game engine or a parser),
compilation may take tens of seconds, ruining the "instant feedback" loop.
Agent code should be as lightweight as possible. (TODO: can we provide `sscache` like service to reduce this time further? Measure it in any case.)

## Drop:
- Problem: When the OS unloads the .dll/.so, Rust destructors are bypassed. Open files, sockets, or background threads will leak or panic.
- Solution: Create a designated #[hot_function] (e.g., fn save_and_cleanup(&mut State)). The main app must explicitly call this function right before it triggers the reload, allowing the library to manually drop its resources.

## Type layout and ABI limitations:
- Problem: Adding or removing a field in a struct changes its memory size. If the main app expects a 12-byte struct and the newly reloaded library sends a 16-byte struct, it causes immediate memory corruption.
- Solution: The harness treats structs that cross the boundary as an immutable "API contract."
Internal structs that never cross the boundary are safe to change.
Also always use `#[repr(c)]` for deterministic in-memory layout.

## String Types:
- Problem: `String` and `&str` rely on Rust's internal allocator and fat pointers. Passing them across a dynamic library boundary is fragile.
- Solution: Harness and lib compiles with the exact same rust compiler, avoiding the problem.

## Dependencies:
- Problem: Both crates compile their own separate copies of dependencies.
 Solution: The harness never passes third-party types across the boundary. Serialize data can be passed just like any API boundary.

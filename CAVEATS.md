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
state across calls, that state is lost when the function evolves
— and every retained revision has its own instance. The harness
forbids this by design: all state is owned by the host binary and
passed into evolvable functions via arguments. Validation
enforces the rule by rejecting `static` items and `thread_local!`
in LLM-generated code before compilation.

## Dangling pointers across reloads

If the host held a reference or pointer to data allocated inside
the dylib, reloading used to unmap the old code and data pages,
leaving the pointer dangling. The keep-all revision registry now
retains every loaded dylib for the lifetime of the process, which
removes the unmap hazard — but the design rule stands: evolvable
function signatures use only caller-owned memory (`&mut [f64]`,
`&mut usize`, etc.). Each revision has its own instance of any
static data, so a pointer into dylib-owned memory would silently
refer to an inactive revision's instance after a swap.

## Compile times

Each evolution round compiles a Rust dylib. The generated crate
has zero dependencies (only `std`), so incremental builds are
fast (~100-200 ms). Adding dependencies to the generated crate
would increase compilation time and break the fast feedback loop.
Keep evolvable function bodies self-contained.
Might change in the future see [TODO.md](TODO.md)

Builds are serialized. The generated crate directory, the `.so`
cargo writes, and the dense revision id are all process-wide, and
cargo takes an exclusive lock on its build directory anyway, so
`Runtime::evolve_batch` runs its lanes' inference concurrently but
their compilations one at a time. That is the right trade while
inference dominates a lane by an order of magnitude. Watch
`symbiont_build_slot_wait_seconds`: if it starts to rival
`symbiont_pipeline_stage_duration_seconds{stage="llm"}`, the batch
has become build-bound and the crate directory needs splitting per
lane — which costs a separate target directory, and therefore a
separate dependency build, for each.

## Destructors bypassed on unload

Loaded dylibs are retained by the revision registry and only
unmapped when the process exits — at which point Rust destructors
are not run. Any resources created inside the dylib (open files,
background threads, heap allocations) will leak for the lifetime
of the process. In practice this is not an issue because the
generated dylib only contains pure functions operating on
caller-provided memory. Avoid spawning threads or opening files
inside evolvable functions.

## Type layout and ABI

Rust has no stable ABI. The host binary and dylib must be compiled
with the same `rustc` version to guarantee matching calling
conventions and memory layouts. The harness ensures this by
compiling the dylib on the same machine with the same toolchain.

Evolvable function signatures work best with primitive and `std`
types (`usize`, `f64`, `&mut [f64]`, etc.). Custom structs across
the boundary are supported when both the host and generated dylib
compile against the same shared API crate, but Rust still has no
stable ABI: layout and calling-convention compatibility remain an
unsafe invariant of the hot-loading boundary.

## String types

`String` and `&str` rely on Rust's internal allocator and fat
pointers. Passing them across a dynamic library boundary is
fragile if the two sides use different allocator instances. The
harness avoids this by compiling both sides with the exact same
compiler and linking against the same `std`.

## Dependencies

By default the generated dylib has no dependencies beyond `std`.
Use [`DylibConfig`](symbiont/src/decl.rs) to add path or registry
dependencies to the generated crate. This enables shared API crates
and upstream dependency types in evolvable signatures, but introduces
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
  `std` types at the boundary when possible. When dependency types
  are used in the signature, expose them through a small shared API
  crate/prelude that both sides compile against with matching
  versions and features.

## Keep-all revision registry memory

Every successfully compiled revision stays loaded so it can be
re-activated at any time. A retained revision maps roughly 1.2 MiB
(debug profile; the generated dylib links `std` statically), and
its versioned `.so` file remains in the temp crate directory on
disk. Long searches with thousands of evolutions should account
for this growth; a pruning API can be added once a real workload
needs it.

## Infinite loops in generated code

LLM-generated code may contain infinite loops (e.g. a sort with
a buggy termination condition). The harness catches panics inside
the dylib via `catch_unwind`, but an infinite loop never panics —
it hangs the calling thread indefinitely.

The harness does **not** detect this automatically. It is the
caller's responsibility to implement timeout detection, for
example by running the evolvable function in a separate thread
with `recv_timeout`. If a timeout fires, the abandoned thread
continues executing in the background. The keep-all revision
registry keeps every loaded dylib mapped, so such a thread keeps
running valid code even after further evolutions — it still burns
a CPU core, so callers should bound how many abandoned threads
they tolerate.

## Panic runtime isolation

The host binary and the dynamically loaded dylib have separate
panic runtimes. A panic originating inside the dylib cannot be
caught by `std::panic::catch_unwind` in the host — the host sees
it as a "foreign exception" and aborts. The harness handles this
with a generated wrapper per declared function *inside the dylib*:
the wrapper is the exported symbol, it calls the agent's function
under `catch_unwind`, and the panic message travels to the host
through a second exported symbol (`__symbiont_take_panic`). Use
`Runtime::take_panic` to retrieve panic messages after each
call. The agent's code itself is compiled as written; the wrappers
live in a `mod __symbiont_exports` appended after it (see
`symbiont/src/layout.rs`).

Only these entry points catch panics — the agent's helpers do not.
A panic in a helper unwinds normally within the dylib until the
wrapper catches it, so nothing escapes, while recursive helpers
avoid a `catch_unwind` frame (and hook install) per call, which
costs speed and stack depth in exactly the hot code this harness
optimizes. It also means helpers may return types that do not
implement `Default`.

When an implementation panics, the wrapped call returns
`Default::default()` as a safe placeholder value — check
`Runtime::take_panic` to distinguish it from a real result.
Every evolvable return type must therefore implement `Default`;
the `evolvable!` macro enforces this with a compile error at the
declaration site, so generated dylibs always compile.

Each revision has its own panic buffer. `Runtime::take_panic`
reads the **active** revision's buffer; panics from calls through
a `RevisionFn` handle land in that handle's revision — read them
with `RevisionFn::take_panic`. A buffer holds only the most
recent message, so concurrent panicking calls into the same
revision overwrite each other.

## Symbol exports and libc collisions

Only the *declared* evolvable functions are exported from the
dylib, through the generated wrappers (`#[unsafe(export_name =
"<name>")]`). Nothing in the agent's code is exported: helper
functions keep their mangled Rust names, and validation rejects
any `#[no_mangle]`, `#[unsafe(no_mangle)]` or `#[export_name]`
attribute the model adds on its own.

That is not cosmetic. An exported symbol in an ELF dylib has
default visibility and is therefore *preemptible*: the compiler
routes even the dylib's own calls to it through the GOT, and the
loader resolves those against the global scope — the host
executable and everything loaded with it, libc included — before
the dylib itself. A helper named like a libc function (`qsort`,
`random`, `div`, `remove`, …) would then hijack the call to libc's
version, which reinterprets the arguments and typically segfaults
the host process. Generated dylibs are additionally linked with
`-Wl,-Bsymbolic-functions` on Linux, which pre-binds intra-dylib
calls to local definitions and closes the same hole for declared
function names.

## Edits to the previous candidate

After a compile failure the agent may answer with an edit to its
previous code instead of the whole code: `E<n> => text` for the
span the `n`-th reported error underlines, a `SEARCH`/`REPLACE`
hunk matched on tokens, or a code block holding only the items
that change. The harness resolves every edit against the
unmodified previous candidate, rejects overlapping edits, and
runs the result through the same parse, validation and build as a
whole response. The base is the candidate that *compiled with
errors*; a response that failed to parse or validate, or an edit
that did not apply, leaves it unchanged. A context or repeat
reset drops the base, because the agent no longer sees the code
an edit would refer to.

Token matching means a `SEARCH` text must be lexically valid Rust
on its own: an unbalanced `{` cannot be matched. Anchors replace
exactly what rustc underlines, which is often a single token; the
nudge quotes that text so the agent sizes its replacement right.

Machine-applicable compiler suggestions are applied once, before
the agent is asked. The patched text is what gets registered and
what a still-failing build's diagnostics refer to. Suggestions
rustc marks `MaybeIncorrect` (an import among several candidates)
are never applied.

## Undefined behaviour and Miri

The generated code itself is barred from introducing new unsafety:
validation rejects any `unsafe` construct in LLM-generated code at
the AST level before compiling — `unsafe` blocks, `unsafe fn`,
`unsafe impl`/`trait`, `extern` blocks, unsafe attributes
(including `#[unsafe(no_mangle)]`), and `unsafe` tokens smuggled
through macros. The offending construct is fed
back to the agent as backpressure.

Beyond `unsafe`, validation also rejects constructs that break the
harness's contracts or reach for process capabilities: `static`
items and `thread_local!` (dylib state resets on reload),
`macro_rules!` definitions, allocator/panic-handler/entry
overrides, tampering with the panic hook, and — by default —
references to `std::process`, `std::thread`, `std::fs`,
`std::net`, `std::env`, `std::os`, and `std::io::stdin` (matched
through `use` aliases and inside macro tokens; glob imports of
denied modules are rejected outright). Hosts widen or narrow the
capability surface with `DylibConfig::with_allowed_path` /
`with_denied_path`. Note this bounds what evolvable code can
*name*; it is *not* a security sandbox — safe Rust reached through
host-provided APIs still runs with the host's privileges.

The pointer-swapping dispatch, the panic-buffer protocol, and the
fn-pointer transmutes are all `unsafe` code. The test suite runs
under [Miri](https://github.com/rust-lang/miri) to detect
undefined behaviour in them:

```sh
MIRIFLAGS="-Zmiri-disable-isolation" cargo miri test -p symbiont --lib
```

Miri cannot spawn processes or `dlopen` libraries, so tests that
compile and load dylibs are `#[cfg_attr(miri, ignore)]`d. The
panic-buffer preamble that ships inside every generated dylib is
still covered: it lives in `symbiont/src/panic_preamble.rs` and is
compiled directly into the test binary (see the tests in
`symbiont/src/unwind.rs`), where Miri executes both sides of the
protocol — the dylib-side buffer writes and the host-side
`read_panic_buffer` decode.

Miri cannot check what it cannot execute: the actual `dlopen`
boundary and cross-dylib calls through swapped pointers remain
outside its reach.

use owo_colors::OwoColorize;
use tracing::debug;

use crate::{
    DocIndex,
    DocIndexError,
    Error,
    Result,
    doc_string::write_prelude_doc_string,
};

const BASE_PROMPT: &str = "
# Role

You are a Rust coding agent running inside the `symbiont` function-evolution harness.

Your job is to generate Rust implementations for one or more evolvable functions.
The harness parses your response, validates the required function signatures,
compiles the code as a temporary dynamic library, hot-swaps the compiled functions
into the host process, evaluates them, and feeds results/errors back to you on
later iterations.

# Output contract

Respond with exactly one fenced Rust code block:

```rust
// code here
```

Do not write prose, explanations, markdown tables, or additional code blocks
outside the Rust block.

Emit complete Rust function item(s), not just function bodies.

Preserve every ABI-relevant part of each required function signature:
- same function name
- parameter names may differ
- same parameter types
- same return type
- same parameter order
- no added or removed parameters
- no changed lifetimes or generics

Prefer emitting only the required top-level evolvable function(s).
If helper logic is needed, prefer local helper functions, closures, constants,
or inline code inside the required function.
Avoid extra top-level generic helper functions.

Every required function must be genuinely implemented. A body that is empty,
only a `todo!()`/`unimplemented!()`/`unreachable!()` placeholder, or a
verbatim copy of the declared default body counts as unimplemented, and a
candidate in which every required function is unimplemented is rejected
before compilation. When you improve only some required functions, the
others may keep their default bodies.

Do not emit `main`, tests, Cargo metadata, modules, or unrelated items unless
the user explicitly asks.

Do not add `#[no_mangle]`, `#[unsafe(no_mangle)]`, `#[export_name]`, or
`extern` attributes; they are rejected before compilation. The harness
handles dynamic-library exports. Visibility does not matter either: a plain
`fn` is fine.

Unsafe code is forbidden and rejected before compilation: never use `unsafe`
blocks, `unsafe fn`, `unsafe impl`, `unsafe trait`, `extern` blocks, or unsafe
attributes.

Also rejected before compilation: `static` items and `thread_local!` (dylib
state resets on every reload — keep state host-owned and passed via arguments;
use `const` for constants), `macro_rules!` definitions, allocator or
panic-handler overrides, tampering with the panic hook, and (by default)
access to `std::process`, `std::thread`, `std::fs`, `std::net`, `std::env`,
`std::os`, and `std::io::stdin`.

# Repairing a compile error

When the harness reports compiler errors for your previous code block, the
line numbers refer to that block, and the errors are numbered `[E1]`, `[E2]`,
... Each header says which text the error underlines, for example
`[E1] replaces `len / 2` on line 3`.

Do not retype code that is already correct. Describe the change instead, in
one `rust-edit` block, using any mix of these two forms:

```rust-edit
E1 => 1
<<<<<<< SEARCH
let mid = len / 2
=======
let mid = len / 2;
>>>>>>> REPLACE
```

- `E<n> => text` replaces exactly the underlined text of error `n` with
  `text`. Replace only that text: if the compiler underlines the `*` in
  `1.5 * 2`, then `E1 => as u64 *` yields `1.5 as u64 * 2`. When the
  replacement spans several lines, put it on the lines after `E<n> =>` and
  end it with a blank line.
- `SEARCH`/`REPLACE` replaces the one place in your previous code whose
  tokens match the `SEARCH` text. Whitespace and line breaks do not have to
  match. Include enough context that the text occurs exactly once.

If a whole function changes, you may instead send a ```rust block that
contains only that function (or only the helpers that change); the harness
replaces the functions of the same name and keeps the rest of your previous
code. A ```rust block that contains every required function replaces the
whole previous code.

The harness applies the compiler's own mechanical fixes (a missing `&` or
`*`, `2` where `2.0` is due) before it asks you; it tells you which ones it
applied. An edit that does not apply is reported and your previous code
stays as it was; answer with a corrected edit or the complete code.

# Compilation environment

The generated crate uses Rust edition 2024.

You may use:
- Rust `std`
- items, types, and methods from the host API documentation
- items already imported by the harness prelude, if any

Do not invent imports or dependencies. Emit no `use` item for a prelude that
the harness already injects.

When host APIs are documented, the generated crate can depend on `host`
without depending directly on crates named in the documentation. Dependency
API sections describe the origin and API of host-re-exported items; they do
not make `dependency_name::...` paths available. Unless the task explicitly
says a crate is a direct dylib dependency, use only unqualified names imported
by `host::prelude::*` (or an explicit `host::...` path). Never add a
dependency import merely because that dependency has a documentation section.

Treat the synopsis literally: call only documented public methods on the exact
receiver type and use documented enum variants and constructors. Do not infer
fields, methods, or variants from similarly named APIs. For arithmetic or
conversions between documented types, use only the operators listed in the
type's `// Operator and conversion impls:` section (`impl OP<Rhs> for Type`);
if no impl is listed for an operand combination, that operation does not
exist — convert operands through documented constructors first. When a
documented type is generic (for example over an id, currency, or state
parameter), unify its generic parameters with the concrete types required by
the evolvable function signature instead of treating them as incompatible.
If several documented constructors exist for the same type, pick the one whose
generic parameters produce the required concrete type (e.g. a `new_with_...`
constructor that accepts the required field directly) rather than concluding
the goal is unachievable. Only if the documented inputs truly expose no API
needed for an idea, choose a simpler implementation or do nothing instead of
inventing one.

# Runtime constraints

Generated code runs inside a hot-reloaded dynamic library. Keep functions
self-contained.

Avoid:
- panics
- infinite loops
- unbounded recursion (the call stack is shared with the host process and
  overflowing it aborts the host — prefer iterative solutions, or bound the
  recursion depth so it stays small for large inputs)
- out-of-bounds indexing
- leaking allocations across the dynamic-library boundary
- spawning threads
- file or network I/O
- printing or logging in hot paths
- global mutable state or persistent static state

Static state inside the dynamic library is reset on every reload and should
not be relied on.

Respect explicit `len` arguments. Usually process only the first `len`
elements and guard against `len > slice.len()` when appropriate.

# Optimization policy

First satisfy correctness and safety.
If feedback reports compiler errors, signature mismatches, panics, invalid
outputs, failed tests, or invalid moves, fix those before optimizing.

When correctness is satisfied and benchmark/evaluation data is provided,
optimize for the concrete metric requested by the user.
Use the previous implementation and evaluation feedback to target the worst
cases first.

Prefer deterministic, simple, robust code.
For performance-sensitive functions, avoid unnecessary heap allocation,
formatting, dynamic dispatch, excessive bounds checks, and avoidable cloning.

";

/// The documentation section for [`DocMode::Inline`]. The full synopsis
/// follows this header.
const INLINE_DOC_SECTION: &str = "# Host API documentation

The following section contains generated documentation for host APIs
available to the evolved code. If empty, only `std` is available.

";

/// The documentation section for the modes that register the `api_index` and
/// `api_doc` tools.
const TOOL_DOC_SECTION: &str = "# Host API documentation
The host API is documented on demand. Two tools give access to it:

- `api_index` lists the public items of a module as `kind name` lines. Call
  it without arguments to list the prelude. Pass a `mod` name from a listing
  to list that module.
- `api_doc` shows the full definition of one item or module: the declaration,
  the inherent methods, and the operator impls.

Both tools accept a single name from a listing, for example `Order`, or a
`::`-separated path, for example `prelude::Order`. A line that ends with
`(re-exported from `crate`)` names the crate that declares the item. The name
is still in scope unqualified. Do not add an import for that crate.

If you do not know the exact signature of an item, call `api_doc` before you
use the item. You have a limited number of documentation tool calls available (50).

";

/// The note for a host crate without a `prelude` module.
const NO_PRELUDE_NOTE: &str = "The host crate does not expose a `prelude` module, so `use host::prelude::*;` imports nothing. No host API is available beyond explicit `host::...` paths.\n";

/// How the agent gets the host API documentation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DocMode {
    /// The system prompt contains the full API synopsis. The prompt grows
    /// with the size of the host API, and every inference request sends it
    /// again.
    #[default]
    Inline,
    /// The system prompt contains the compact index of the prelude. The
    /// `api_index` and `api_doc` tools give the agent the full definitions
    /// on demand.
    IndexAndTools,
    /// The system prompt contains no API content. The agent explores the API
    /// with the `api_index` and `api_doc` tools.
    Tools,
}

impl DocMode {
    /// Does this mode register the documentation tools on the agent?
    pub(crate) fn uses_tools(self) -> bool {
        !matches!(self, Self::Inline)
    }
}

/// Build the system prompt (the agent preamble) that symbiont sends with
/// every inference request.
///
/// This is the exact string that [`crate::agent_builder`] installs as the
/// preamble of the agent. A host can thus reproduce what the agent was told
/// without a copy of its own.
///
/// [`EvolutionTrace`](crate::EvolutionTrace) omits the preamble by design.
/// The preamble is the same for every attempt of a lane and every lane of a
/// batch. With [`DocMode::Inline`] it embeds the full host API documentation
/// and is the largest single string in the process. The other modes keep it
/// small.
///
/// # Limits
///
/// The result is the same as the string that an agent received only under
/// three conditions. The caller must pass the same `opt_crate_name`. The
/// caller must pass the same `doc_mode`. The caller must also not replace
/// the preamble on its own builder.
///
/// With `Some(crate_name)`, this function builds the rustdoc JSON of the
/// host crate and of every crate that the host facade re-exports, which is
/// slow. Call it one time and cache the result. A mode with tools pays this
/// cost one time here, so that a later tool call needs no I/O.
///
/// # Arguments
///
/// - `opt_crate_name`: The crate to document, usually
///   `Some(env!("CARGO_PKG_NAME"))`. With `None`, no host API is documented
///   and `doc_mode` must be `DocMode::Inline`.
/// - `doc_mode`: How the prompt carries the host API documentation.
///
/// # Errors
///
/// Returns [`Error::InvalidDocMode`] if `doc_mode` registers tools but
/// `opt_crate_name` is `None`. Otherwise, returns an error if the runtime
/// cannot build or parse the documentation of the host crate.
pub async fn system_prompt(opt_crate_name: Option<&str>, doc_mode: DocMode) -> Result<String> {
    let mut prompt = BASE_PROMPT.to_string();
    match (opt_crate_name, doc_mode) {
        (Some(crate_name), DocMode::Inline) => {
            prompt.push_str(INLINE_DOC_SECTION);
            write_prelude_doc_string(&mut prompt, crate_name).await?;
        }
        (Some(crate_name), DocMode::IndexAndTools) => {
            prompt.push_str(TOOL_DOC_SECTION);
            let index = DocIndex::host(crate_name).await?;
            match index.render_index(None) {
                Ok(listing) if !listing.trim().is_empty() => {
                    push_prelude_index(&mut prompt, &listing);
                }
                Ok(_) => prompt.push_str("The prelude imports no names.\n"),
                Err(DocIndexError::NoPrelude) => prompt.push_str(NO_PRELUDE_NOTE),
                Err(err) => return Err(err.into()),
            }
        }
        (Some(_), DocMode::Tools) => prompt.push_str(TOOL_DOC_SECTION),
        (None, DocMode::Inline) => prompt.push_str(INLINE_DOC_SECTION),
        (None, _) => return Err(Error::InvalidDocMode),
    }
    debug!("system_prompt: {}", prompt.green());

    Ok(prompt)
}

/// Append the compact prelude index to the prompt.
fn push_prelude_index(prompt: &mut String, listing: &str) {
    prompt.push_str(
        "The harness injects `use host::prelude::*;`. The following list is complete: these names are in scope and no others. Call `api_doc` with a name or a path to get the full definition.\n\n```\n",
    );
    prompt.push_str(listing);
    prompt.push_str("```\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn system_prompt_without_crate_requires_inline() {
        let prompt = system_prompt(None, DocMode::Inline)
            .await
            .expect("no docs to build");
        assert!(prompt.contains("# Host API documentation"));
        assert!(prompt.contains("only `std` is available"));
        assert!(!prompt.contains("api_doc"));
        for doc_mode in [DocMode::IndexAndTools, DocMode::Tools] {
            assert!(
                matches!(
                    system_prompt(None, doc_mode).await,
                    Err(Error::InvalidDocMode)
                ),
                "tool doc modes need a crate, got doc_mode {doc_mode:?}"
            );
        }
    }

    /// The prompt teaches the same edit syntax the parser accepts: the
    /// fence tag, the hunk markers and the anchor arrow.
    #[test]
    fn base_prompt_documents_the_edit_contract() {
        for needle in [
            "```rust-edit",
            "<<<<<<< SEARCH",
            "=======",
            ">>>>>>> REPLACE",
            "E1 => 1",
            "[E1]",
            "Do not retype code that is already correct.",
        ] {
            assert!(BASE_PROMPT.contains(needle), "prompt lost `{needle}`");
        }
    }

    #[test]
    fn tool_doc_section_names_both_tools() {
        assert!(TOOL_DOC_SECTION.contains("api_index"));
        assert!(TOOL_DOC_SECTION.contains("api_doc"));
    }

    #[test]
    fn prelude_index_section_wraps_the_listing() {
        let mut prompt = String::new();
        push_prelude_index(&mut prompt, "struct Order\nfn submit_order\n");
        assert!(prompt.contains("use host::prelude::*;"));
        assert!(prompt.contains("```\nstruct Order\nfn submit_order\n```\n"));
    }
}

// SPDX-License-Identifier: MPL-2.0
//! Module containing inference related functions.

use std::env::var;

use owo_colors::OwoColorize;
use rig::{
    agent::Agent,
    client::CompletionClient,
    providers::{
        openai,
        openai::completion::CompletionModel,
    },
};
use tracing::info;

use crate::doc_string::write_prelude_doc_string;

/// Initialize the agent using the environment variables.
pub async fn init_agent(crate_name: &str) -> crate::Result<Agent<CompletionModel>> {
    let api_key = var("API_KEY").unwrap_or_default();
    let base_url = var("BASE_URL").unwrap_or_default();
    let model = var("MODEL").unwrap_or_default();

    let client = openai::Client::builder()
        .api_key(api_key)
        .base_url(base_url)
        .build()?
        .completions_api(); // Use Chat Completions API instead of Responses API

    let mut system_prompt = r#"# Role

You are a Rust coding agent running inside the `symbiont` function-evolution harness.

Your job is to generate Rust implementations for one or more evolvable functions.
The harness parses your response, validates the required function signatures,
compiles the code as a temporary dynamic library, hot-swaps the compiled functions into the host process,
evaluates them, and feeds results/errors back to you on later iterations.

# Output contract

Always respond with exactly one fenced Rust code block:

```rust
// code here
```

Do not write prose, explanations, markdown tables, or additional code blocks outside the Rust block.

Emit complete Rust function item(s), not just function bodies.

Preserve every required function signature exactly:
- same function name
- same parameter names
- same parameter types
- same return type
- same parameter order
- no added or removed parameters
- no changed lifetimes or generics

Prefer emitting only the required top-level evolvable function(s).
If helper logic is needed, prefer local helper functions, closures, constants, or inline code inside the required function.
Avoid extra top-level generic helper functions.

Do not emit `main`, tests, Cargo metadata, modules, or unrelated items unless the user explicitly asks.

Do not add `#[no_mangle]`, `#[unsafe(no_mangle)]`, or `extern` attributes. The harness handles dynamic-library exports.

# Compilation environment

The generated crate uses Rust edition 2024.

You may use:
- Rust `std`
- items, types, and methods documented in the host API section below
- items already imported by the harness prelude, if any

Do not use external crates unless they are explicitly documented or available. Do not invent imports or dependencies.

If host crate APIs are documented below, assume the relevant prelude/imports may already be injected by the harness. Use only the documented public API.

# Runtime constraints

Generated code runs inside a hot-reloaded dynamic library. Keep functions self-contained.

Avoid:
- panics
- infinite loops
- out-of-bounds indexing
- leaking allocations across the dynamic-library boundary
- spawning threads
- file or network I/O
- printing or logging in hot paths
- global mutable state or persistent static state

Static state inside the dynamic library is reset on every reload and should not be relied on.

Respect explicit `len` arguments. Usually process only the first `len` elements and guard against `len > slice.len()` when appropriate.

# Optimization policy

First satisfy correctness and safety.
If feedback reports compiler errors, signature mismatches, panics, invalid outputs, failed tests, or invalid moves, fix those before optimizing.

When correctness is satisfied and benchmark/evaluation data is provided,
optimize for the concrete metric requested by the user.
Use the previous implementation and evaluation feedback to target the worst cases first.

Prefer deterministic, simple, robust code.
For performance-sensitive functions, avoid unnecessary heap allocation, formatting, dynamic dispatch, excessive bounds checks, and avoidable cloning.

# Host API documentation

The following section contains generated documentation for host APIs available to the evolved code. If empty, only `std` is available.

"#
    .to_string();
    write_prelude_doc_string(&mut system_prompt, crate_name).await?;
    info!("system_prompt: {}", system_prompt.green());

    Ok(client.agent(model).preamble(&system_prompt).build())
}

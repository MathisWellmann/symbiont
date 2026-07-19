// SPDX-License-Identifier: MPL-2.0
//! Parse Rust code from markdown-fenced code blocks.
//!
//! Handles strings like:
//! ```text
//! "```rust
//! fn step(counter: &mut usize) {
//!     *counter += 1;
//! }
//! ```"
//!

use syn::parse_file;

use crate::{
    Result,
    error::Error,
};

/// Extract the inner Rust source code from a markdown-fenced code block.
///
/// Handles the common pattern where an LLM response wraps code in
/// ```rust ... ``` fences. Returns the first code block found, or `None`.
///
/// Fences only count when they open a line (ignoring leading whitespace),
/// per CommonMark. This keeps fences embedded in doc comments, such as
/// `/// ```ignore` examples the LLM re-emits from the function's docs,
/// from being mistaken for the closing fence and truncating the code.
pub(crate) fn extract_rust_code(input: &str) -> Option<String> {
    // Prefer an explicit ```rust fence, then fall back to any ``` fence.
    extract_fenced(input, "```rust").or_else(|| extract_fenced(input, "```"))
}

/// Extract the contents of the first line-anchored fenced block opened by
/// `start_marker` and closed by a line-anchored ``` fence.
fn extract_fenced(input: &str, start_marker: &str) -> Option<String> {
    let start = find_line_anchored_fence(input, start_marker, 0)?;
    // Skip the rest of the opening fence line (language tag, whitespace).
    let code_start = start + input[start..].find('\n')? + 1;
    let end = find_line_anchored_fence(input, "```", code_start)?;
    Some(input[code_start..end].trim().to_string())
}

/// Byte offset of the first occurrence of `marker` at or after `from` that
/// is preceded on its line only by whitespace.
fn find_line_anchored_fence(input: &str, marker: &str, from: usize) -> Option<usize> {
    let mut search_from = from;
    while let Some(rel) = input[search_from..].find(marker) {
        let pos = search_from + rel;
        let line_start = input[..pos].rfind('\n').map(|i| i + 1).unwrap_or(0);
        if input[line_start..pos].chars().all(char::is_whitespace) {
            return Some(pos);
        }
        search_from = pos + marker.len();
    }
    None
}

/// Parse Rust code from a markdown-fenced code block into a `syn::File` AST.
///
/// This is the public entry point used by main.rs — callers pass the raw
/// LLM response and this function handles fence extraction + parsing.
///
/// On parse failure the returned [`Error::CouldNotParseRust`] carries the
/// offending code and syn's diagnostic (with line/column), so the evolve
/// loop can feed a precise nudge back to the LLM.
pub(crate) fn parse_rust_code(input: &str) -> Result<syn::File> {
    let code = extract_rust_code(input).ok_or(Error::NoRustCode)?;
    let file = parse_file(&code).map_err(|e| {
        let start: proc_macro2::LineColumn = e.span().start();
        Error::CouldNotParseRust {
            err: format!("{e} (line {}, column {})", start.line, start.column),
            code,
        }
    })?;
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_rust_code_simple_fence() {
        let input = "```rust
fn step(counter: &mut usize) {
    *counter += 1;
}
```";
        let code = extract_rust_code(input).expect("Can parse");
        assert_eq!(
            code.trim(),
            "fn step(counter: &mut usize) {\n    *counter += 1;\n}"
        );
    }

    #[test]
    fn test_extract_rust_code_with_text_around() {
        let input = "Here is the implementation:
```rust
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
```
Hope that helps!";
        let code = extract_rust_code(input).expect("can extract");
        assert_eq!(
            code.trim(),
            "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}"
        );
    }

    #[test]
    fn test_extract_rust_code_no_fence() {
        let input = "fn bare_function(x: i32) -> i32 { x }";
        assert!(extract_rust_code(input).is_none());
    }

    #[test]
    fn test_extract_rust_code_generic_fence() {
        let input = "```
fn no_lang_marker(x: i32) -> i32 { x }
```";
        let code = extract_rust_code(input).expect("can extract");
        assert_eq!(code.trim(), "fn no_lang_marker(x: i32) -> i32 { x }");
    }

    #[test]
    fn test_extract_rust_code_with_prefix_and_extra_whitespace() {
        // Prefix text ensures `start > 0` and extra whitespace after the fence
        // ensures the whitespace count is `> 1`, so that `code_start + count`
        // differs from `code_start * count` in a way `trim()` cannot recover.
        let prefix = "Here is the code you requested:\n";
        let input = format!("{prefix}```rust\n\n  fn foo() -> i32 {{ 42 }}\n```");
        let code = extract_rust_code(&input).expect("can extract");
        assert_eq!(code, "fn foo() -> i32 { 42 }");
    }

    #[test]
    fn test_extract_rust_code_generic_fence_with_prefix() {
        // Prefix ensures `start > 0` for the generic-fence branch so that
        // mutations of `+ start` to `- start` or `* start` produce a wrong
        // (or panicking) result.
        let input = "Some explanation here:\n```\nfn no_lang(x: i32) -> i32 { x }\n```";
        let code = extract_rust_code(input).expect("can extract");
        assert_eq!(code, "fn no_lang(x: i32) -> i32 { x }");
    }

    #[test]
    fn test_parse_rust_code_from_block() {
        let input = "```rust
#[unsafe(no_mangle)]
pub fn step(state: &mut usize) {
    *state += 1;
}
```";
        let file = parse_rust_code(input).expect("can parse");
        assert_eq!(file.items.len(), 1);
    }

    /// Regression test: a fence embedded in a doc comment (`/// ```ignore`)
    /// must not terminate the outer ```rust block. This previously truncated
    /// the extracted code to the doc-comment prefix, producing the
    /// "unexpected end of input (line 1, column 0)" parse error when the LLM
    /// re-emitted the evolvable function's documentation.
    #[test]
    fn test_extract_rust_code_fence_inside_doc_comment() {
        let input = "```rust
/// Construct commands, e.g. for cancellation:
///
/// ```ignore
/// if let Ok(command) = Command::limit_order(Side::Buy, price, qty, 7) {
///     commands[0] = command;
/// }
/// ```
///
/// `Command::market_order(...)` submits a market order.
pub fn action(commands: &mut [u32]) {
    commands[0] = 1;
}
```
Hope that helps!";
        let code = extract_rust_code(input).expect("can extract");
        assert!(
            code.contains("pub fn action"),
            "must not stop at the doc-comment fence: {code}"
        );
        assert!(code.ends_with('}'), "must extract the full block: {code}");
        assert!(!code.contains("Hope that helps"));
    }

    /// The doc-comment regression above must also parse end-to-end.
    #[test]
    fn test_parse_rust_code_with_doc_comment_fence() {
        let input = "```rust
/// Example usage:
///
/// ```ignore
/// let x = step(1);
/// ```
pub fn step(x: i32) -> i32 {
    x + 1
}
```";
        let file = parse_rust_code(input).expect("can parse");
        assert_eq!(file.items.len(), 1);
    }

    /// A ```rust fence inside prose (e.g. quoted mid-line) must not be taken
    /// as the opening fence; only line-anchored fences count.
    #[test]
    fn test_extract_rust_code_ignores_inline_fence_mentions() {
        let input = "Wrap your code like ```rust ... ``` as requested:
```rust
fn real() -> i32 { 1 }
```";
        let code = extract_rust_code(input).expect("can extract");
        assert_eq!(code, "fn real() -> i32 { 1 }");
    }

    /// An indented fence (whitespace-only prefix) still opens/closes a block.
    #[test]
    fn test_extract_rust_code_indented_fence() {
        let input = "  ```rust\n  fn indented() -> i32 { 2 }\n  ```";
        let code = extract_rust_code(input).expect("can extract");
        assert_eq!(code, "fn indented() -> i32 { 2 }");
    }

    /// An opening fence with no newline after it has no code block.
    #[test]
    fn test_extract_rust_code_unterminated_fence_line() {
        assert!(extract_rust_code("```rust fn oneliner() {}").is_none());
    }

    /// Regression test for the cast-then-shift grammar pitfall LLMs run into:
    /// `r as u8 << 16` is invalid Rust because `<<` after a cast type is
    /// interpreted as the start of generic arguments (`u8<...`), not a shift.
    /// The returned error must carry the code and a located diagnostic.
    #[test]
    fn test_parse_error_carries_code_and_location() {
        let input = "```rust
pub fn shade(x: f64, y: f64, t: f64) -> u32 {
    let r = (x * 255.0) as u32;
    (r as u8 << 16) as u32
}
```";
        let err = match parse_rust_code(input) {
            Err(e) => e,
            Ok(_) => panic!("cast followed by shift must fail to parse"),
        };
        match err {
            Error::CouldNotParseRust { code, err } => {
                assert!(
                    code.contains("r as u8 << 16"),
                    "code must be echoed: {code}"
                );
                assert!(err.contains("line "), "error must carry a location: {err}");
            }
            other => panic!("expected CouldNotParseRust, got: {other}"),
        }
    }
}

//! Parse Rust code from markdown-fenced code blocks.
//!
//! Handles strings like:
//! ```
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
/// ```rust ... ``` fences. Returns the first code block found, or `None`
pub(crate) fn extract_rust_code(input: &str) -> Option<String> {
    // Match ```rust ... ``` with optional whitespace/newlines
    let start_marker = "```rust";
    let end_marker = "```";

    if let Some(start) = input.find(start_marker) {
        let code_start = start + start_marker.len();
        // Skip any whitespace after the language tag
        let code_start = input[code_start..]
            .chars()
            .take_while(|c| c.is_whitespace())
            .count()
            + code_start;

        if let Some(end) = input[code_start..].find(end_marker) {
            let code_end = code_start + end;
            let mut code = input[code_start..code_end].to_string();
            // Strip leading/trailing whitespace (including the newline after the fence)
            code = code.trim().to_string();
            return Some(code);
        }
    }

    // Fallback: if no ```rust fence found, try any ``` fence
    if let Some(start) = input.find("```") {
        let after_fence = &input[start + 3..];
        // Skip language tag and optional whitespace
        let lang_end = after_fence
            .find(|c: char| c == '\n')
            .unwrap_or(after_fence.len());
        let code_start = lang_end
            + start
            + 3
            + after_fence[lang_end..]
                .chars()
                .take_while(|c| c.is_whitespace())
                .count();

        if let Some(end) = input[code_start..].find("```") {
            let code_end = code_start + end;
            let mut code = input[code_start..code_end].to_string();
            code = code.trim().to_string();
            return Some(code);
        }
    }

    // No fence detected.
    None
}

/// Parse Rust code from a markdown-fenced code block into a `syn::File` AST.
///
/// This is the public entry point used by main.rs — callers pass the raw
/// LLM response and this function handles fence extraction + parsing.
pub(crate) fn parse_rust_code(input: &str) -> Result<syn::File> {
    let code = extract_rust_code(input).ok_or(Error::NoRustCode)?;
    let file = parse_file(&code)?;
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_rust_code_simple_fence() {
        let input = r#"```rust
fn step(counter: &mut usize) {
    *counter += 1;
}
```"#;
        let code = extract_rust_code(input).expect("Can parse");
        assert_eq!(
            code.trim(),
            "fn step(counter: &mut usize) {\n    *counter += 1;\n}"
        );
    }

    #[test]
    fn test_extract_rust_code_with_text_around() {
        let input = r#"Here is the implementation:
```rust
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
```
Hope that helps!"#;
        let code = extract_rust_code(input).unwrap();
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
        let input = r#"```
fn no_lang_marker(x: i32) -> i32 { x }
```"#;
        let code = extract_rust_code(input).unwrap();
        assert_eq!(code.trim(), "fn no_lang_marker(x: i32) -> i32 { x }");
    }

    #[test]
    fn test_parse_rust_code_from_block() {
        let input = r#"```rust
#[unsafe(no_mangle)]
pub fn step(state: &mut usize) {
    *state += 1;
}
```"#;
        let file = parse_rust_code(input).unwrap();
        assert_eq!(file.items.len(), 1);
    }
}

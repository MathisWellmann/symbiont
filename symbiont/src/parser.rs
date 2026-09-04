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

use syn::{
    parse_file,
    visit::{
        self,
        Visit,
    },
};

use crate::{
    Result,
    error::Error,
};

/// One fenced block of a markdown response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Fence {
    /// The first word of the info string after the opening fence, e.g.
    /// `rust` for ```` ```rust ````. Empty for a bare ```` ``` ````.
    tag: String,
    /// The block's content with outer whitespace trimmed.
    body: String,
}

impl Fence {
    /// The block's content with outer whitespace trimmed.
    pub(crate) fn body(&self) -> &str {
        &self.body
    }

    /// Is this a block of Rust code, tagged as such?
    pub(crate) fn is_tagged_rust(&self) -> bool {
        matches!(self.tag.as_str(), "rust" | "rs")
    }

    /// Is this a block of edits (see [`crate::edit`])? Tagged `rust-edit`
    /// or `edit`, or any block whose text opens with the `<<<<<<<` marker:
    /// a model that tags its edits `rust` still means them as edits.
    pub(crate) fn is_edit(&self) -> bool {
        matches!(self.tag.as_str(), "rust-edit" | "edit")
            || self.body.trim_start().starts_with("<<<<<<<")
    }
}

/// Every line-anchored fenced block of `input`, in source order.
///
/// Fences only count when they open a line (ignoring leading whitespace),
/// per CommonMark. This keeps fences embedded in doc comments, such as
/// `/// ```ignore` examples the LLM re-emits from the function's docs,
/// from being mistaken for the closing fence and truncating the code.
pub(crate) fn fences(input: &str) -> Vec<Fence> {
    let mut blocks = Vec::new();
    let mut from = 0;
    while let Some(start) = find_line_anchored_fence(input, "```", from) {
        // The rest of the opening fence line is the info string.
        let Some(rel_newline) = input[start..].find('\n') else {
            break;
        };
        let info = &input[start + "```".len()..start + rel_newline];
        let code_start = start + rel_newline + 1;
        let Some(end) = find_line_anchored_fence(input, "```", code_start) else {
            break;
        };
        blocks.push(Fence {
            tag: info
                .split(|c: char| c.is_whitespace() || c == ',')
                .next()
                .unwrap_or_default()
                .to_string(),
            body: input[code_start..end].trim().to_string(),
        });
        from = end + "```".len();
    }
    blocks
}

/// The code blocks among `fences`, in source order.
///
/// Handles the common pattern where an LLM response wraps code in
/// ```rust ... ``` fences. An explicit ```rust fence wins outright when one
/// is present: a response that tags its code also tags its answer, so the
/// untagged blocks around it are prose or program output. Only when no
/// tagged fence exists does an untagged ``` fence count. Edit blocks are
/// never code blocks.
pub(crate) fn code_blocks(fences: &[Fence]) -> Vec<&Fence> {
    let tagged: Vec<&Fence> = fences
        .iter()
        .filter(|fence| fence.is_tagged_rust() && !fence.is_edit())
        .collect();
    if tagged.is_empty() {
        fences
            .iter()
            .filter(|fence| fence.tag.is_empty() && !fence.is_edit())
            .collect()
    } else {
        tagged
    }
}

/// Extract every candidate Rust source block from a markdown response, in
/// source order. See [`code_blocks`].
fn extract_rust_code_blocks(input: &str) -> Vec<String> {
    code_blocks(&fences(input))
        .into_iter()
        .map(|fence| fence.body.clone())
        .collect()
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

/// The agent's answer: the text of the chosen code block and its AST.
///
/// `source` is the block as the agent wrote it (outer whitespace trimmed).
/// It is what the dylib compiles, what the compiler's line numbers refer to
/// and what the harness quotes back, so it is never re-rendered.
pub(crate) struct Candidate {
    /// The chosen code block, verbatim.
    source: String,
    /// `source`, parsed.
    ast: syn::File,
}

impl Candidate {
    /// The parsed block.
    pub(crate) fn ast(&self) -> &syn::File {
        &self.ast
    }

    /// Does the block hold at least one function item?
    fn has_fn(&self) -> bool {
        self.ast
            .items
            .iter()
            .any(|item| matches!(item, syn::Item::Fn(_)))
    }

    /// Give up the AST and keep the text.
    pub(crate) fn into_source(self) -> String {
        self.source
    }

    /// Split into the text and its AST.
    #[cfg(test)]
    pub(crate) fn into_parts(self) -> (String, syn::File) {
        (self.source, self.ast)
    }
}

/// Parse Rust code from a markdown-fenced code block.
///
/// Callers pass the raw LLM response and this function handles fence
/// extraction + parsing.
///
/// A response often carries more than one fenced block: reasoning models
/// quote scratch snippets while thinking themselves towards an answer, and
/// some models append example usage or expected output after it. The answer
/// is the *last* block holding real items, so candidates are considered from
/// the end and the first one that parses into at least one function wins. A
/// parseable but function-less block (a lone `use`, a snippet of output) only
/// serves as a fallback.
///
/// A block that parses but carries an item `syn` only kept as raw tokens (see
/// [`find_verbatim`]) is rejected like a parse error: `prettyplease` panics on
/// most of those instead of printing them.
///
/// On parse failure the returned [`Error::CouldNotParseRust`] carries the
/// offending code and syn's diagnostic (with line/column), so the evolve
/// loop can feed a precise nudge back to the LLM. The diagnostic describes
/// the last block, which is the one the agent meant as its answer — quoting
/// an earlier scratch snippet back at it only derails the next attempt.
pub(crate) fn parse_rust_code(input: &str) -> Result<Candidate> {
    let blocks = extract_rust_code_blocks(input);
    if blocks.is_empty() {
        return Err(Error::NoRustCode);
    }

    let mut function_less: Option<Candidate> = None;
    let mut last_err: Option<Error> = None;
    for code in blocks.into_iter().rev() {
        match parse_candidate(code) {
            // Reverse iteration means the first candidate to set any of
            // these is the latest one, so `or` keeps the block closest to
            // the end of the response.
            Ok(candidate) if candidate.has_fn() => return Ok(candidate),
            Ok(candidate) => function_less = function_less.or(Some(candidate)),
            Err(e) => last_err = last_err.or(Some(e)),
        }
    }

    function_less
        .ok_or_else(|| last_err.expect("a block that neither parses nor errors is impossible"))
}

/// Parse one block of Rust source into a [`Candidate`].
///
/// Rejects a block that `syn` only keeps as raw tokens (see
/// [`find_verbatim`]) like a parse error, with [`Error::CouldNotParseRust`]
/// carrying the code and a located diagnostic either way.
pub(crate) fn parse_candidate(source: String) -> Result<Candidate> {
    match parse_file(&source) {
        Ok(ast) => match find_verbatim(&ast) {
            Some(tokens) => Err(unprintable_item(&source, &tokens)),
            None => Ok(Candidate { source, ast }),
        },
        Err(e) => Err(could_not_parse(&source, &e)),
    }
}

/// Build the [`Error::CouldNotParseRust`] backpressure payload for `code`.
fn could_not_parse(code: &str, e: &syn::Error) -> Error {
    let start: proc_macro2::LineColumn = e.span().start();
    let err = format!("{e} (line {}, column {})", start.line, start.column);
    Error::CouldNotParseRust {
        err: with_offending_line(err, code, start),
        code: code.to_string(),
    }
}

/// Build the [`Error::CouldNotParseRust`] backpressure payload for an item
/// `syn` kept as raw tokens (see [`find_verbatim`]).
fn unprintable_item(code: &str, tokens: &proc_macro2::TokenStream) -> Error {
    let start: proc_macro2::LineColumn = syn::spanned::Spanned::span(tokens).start();
    let err = format!(
        "incomplete item `{tokens}` (line {}, column {}): an item without a body or \
         initializer is only valid inside a trait definition",
        start.line, start.column,
    );
    Error::CouldNotParseRust {
        err: with_offending_line(err, code, start),
        code: code.to_string(),
    }
}

/// Quote the offending source line with a caret marker so the agent does not
/// have to count lines to locate the error.
fn with_offending_line(mut err: String, code: &str, start: proc_macro2::LineColumn) -> String {
    if let Some(line) = code.lines().nth(start.line.saturating_sub(1)) {
        use std::fmt::Write;
        write!(
            err,
            "\nOffending line:\n{line}\n{caret_pad}^ error is here",
            caret_pad = " ".repeat(start.column)
        )
        .expect("Can write to String");
    }
    err
}

/// The tokens of the first item `syn` parked in a `Verbatim` variant, if any.
///
/// `syn` is deliberately lenient about a handful of malformed items: rather
/// than failing, it stashes their raw tokens in `Item::Verbatim` (and the
/// `ImplItem` / `TraitItem` / `ForeignItem` equivalents). `const NAME: Type;`
/// and `fn f();` — bodyless forms that are only legal inside a trait — are the
/// ones LLMs emit, typically as a half-finished edit.
///
/// `prettyplease` has no printer for most verbatim forms and answers them with
/// `unimplemented!`, so leaving one in the AST turns a bad generation into a
/// panic in the evolve loop. Rejecting the block here keeps every downstream
/// `unparse` total — including the ones inside the validation error payloads,
/// which would otherwise panic while *reporting* the problem.
fn find_verbatim(file: &syn::File) -> Option<proc_macro2::TokenStream> {
    let mut scan = VerbatimScan(None);
    scan.visit_file(file);
    scan.0
}

/// Records the first non-empty verbatim token stream found in a file.
/// An empty one is fine: `prettyplease` prints it as a blank line.
struct VerbatimScan(Option<proc_macro2::TokenStream>);

impl VerbatimScan {
    fn record(&mut self, tokens: &proc_macro2::TokenStream) {
        if self.0.is_none() && !tokens.is_empty() {
            self.0 = Some(tokens.clone());
        }
    }
}

impl<'ast> Visit<'ast> for VerbatimScan {
    fn visit_item(&mut self, node: &'ast syn::Item) {
        if let syn::Item::Verbatim(tokens) = node {
            self.record(tokens);
        }
        visit::visit_item(self, node);
    }

    fn visit_impl_item(&mut self, node: &'ast syn::ImplItem) {
        if let syn::ImplItem::Verbatim(tokens) = node {
            self.record(tokens);
        }
        visit::visit_impl_item(self, node);
    }

    fn visit_trait_item(&mut self, node: &'ast syn::TraitItem) {
        if let syn::TraitItem::Verbatim(tokens) = node {
            self.record(tokens);
        }
        visit::visit_trait_item(self, node);
    }

    fn visit_foreign_item(&mut self, node: &'ast syn::ForeignItem) {
        if let syn::ForeignItem::Verbatim(tokens) = node {
            self.record(tokens);
        }
        visit::visit_foreign_item(self, node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The single block of a well-behaved response, i.e. the one
    /// [`parse_rust_code`] settles on when there is nothing else to choose
    /// between. Multi-block selection is covered separately below.
    fn extract_rust_code(input: &str) -> Option<String> {
        extract_rust_code_blocks(input).pop()
    }

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
pub fn step(state: &mut usize) {
    *state += 1;
}
```";
        let (source, ast) = parse_rust_code(input).expect("can parse").into_parts();
        assert_eq!(ast.items.len(), 1);
        assert_eq!(
            source, "pub fn step(state: &mut usize) {\n    *state += 1;\n}",
            "the source is the block as written, not a re-rendering"
        );
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
        let candidate = parse_rust_code(input).expect("can parse");
        assert_eq!(candidate.ast().items.len(), 1);
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
                assert!(
                    err.contains("Offending line:\n    (r as u8 << 16) as u32"),
                    "error must quote the offending source line: {err}"
                );
                assert!(
                    err.contains("^ error is here"),
                    "error must carry a caret marker: {err}"
                );
            }
            other => panic!("expected CouldNotParseRust, got: {other}"),
        }
    }

    /// Regression test for the `prettyplease` panic path: `syn` accepts
    /// `const NAME: Type;` (no initializer) by parking it in `Item::Verbatim`,
    /// and `prettyplease::unparse` then hits `unimplemented!("Item::Verbatim
    /// ..")`. Such a block must be rejected as unparseable *before* it reaches
    /// any `unparse`, so the evolve loop nudges the model instead of panicking.
    #[test]
    fn test_parse_rust_code_rejects_bodyless_item() {
        let input = "```rust
pub fn action(state: &mut usize) {
    *state += 1;
}
const SELL_ID: User200;
```";
        let err = match parse_rust_code(input) {
            Err(e) => e,
            Ok(_) => panic!("a bodyless const must not reach prettyplease"),
        };
        match err {
            Error::CouldNotParseRust { code, err } => {
                assert!(
                    code.contains("const SELL_ID"),
                    "code must be echoed: {code}"
                );
                assert!(
                    err.contains("incomplete item"),
                    "error must name the problem: {err}"
                );
                assert!(
                    err.contains("Offending line:\nconst SELL_ID: User200;"),
                    "error must quote the offending source line: {err}"
                );
            }
            other => panic!("expected CouldNotParseRust, got: {other}"),
        }
    }

    /// Regression test for reasoning models that quote scratch snippets on
    /// their way to an answer. Taking the *first* fence made the harness
    /// reject `let data = &mut data;` — a fragment lifted out of the model's
    /// own musings — and burn a self-healing attempt while the real
    /// implementation sat in the final block.
    #[test]
    fn test_parse_rust_code_picks_answer_after_scratch_snippets() {
        let input = "Let me think. I could rebind the slice:
```rust
let data = &mut data;
```
No, that does not work. Nor does this:
```rust
data = &mut temp[1];
```
Here is the final implementation:
```rust
pub fn sort(data: &mut [f64], len: usize) {
    for i in 1..len {
        let mut j = i;
        while j > 0 && data[j - 1] > data[j] {
            data.swap(j - 1, j);
            j -= 1;
        }
    }
}
```";
        let (source, ast) = parse_rust_code(input)
            .expect("must recover the final block")
            .into_parts();
        assert_eq!(ast.items.len(), 1);
        assert!(
            matches!(&ast.items[0], syn::Item::Fn(f) if f.sig.ident == "sort"),
            "must pick the implementation, not a scratch fragment"
        );
        assert!(
            source.starts_with("pub fn sort"),
            "the source is the chosen block: {source}"
        );
    }

    /// The mirror case: some models append example usage or expected output
    /// after the answer. Such a trailing block holds statements rather than
    /// items, so the last block carrying a function is the answer.
    #[test]
    fn test_parse_rust_code_skips_trailing_usage_block() {
        let input = "```rust
pub fn double(x: i32) -> i32 {
    x * 2
}
```
Example usage:
```rust
let y = double(21);
assert_eq!(y, 42);
```";
        let candidate = parse_rust_code(input).expect("must skip the usage block");
        assert!(
            matches!(&candidate.ast().items[0], syn::Item::Fn(f) if f.sig.ident == "double"),
            "must pick the function, not the usage snippet"
        );
    }

    /// When no candidate parses, the diagnostic must describe the *last*
    /// block: that is the agent's answer, and quoting an earlier scratch
    /// snippet back at it would derail the next attempt.
    #[test]
    fn test_parse_error_describes_the_last_block() {
        let input = "First idea:
```rust
let x = ;
```
Final answer:
```rust
pub fn shade(x: f64) -> u32 {
    (x as u8 << 16) as u32
}
```";
        let err = match parse_rust_code(input) {
            Err(e) => e,
            Ok(_) => panic!("neither block is valid Rust"),
        };
        match err {
            Error::CouldNotParseRust { code, .. } => {
                assert!(
                    code.contains("x as u8 << 16"),
                    "must quote the final block, got: {code}"
                );
                assert!(
                    !code.contains("let x = ;"),
                    "must not quote the scratch snippet, got: {code}"
                );
            }
            other => panic!("expected CouldNotParseRust, got: {other}"),
        }
    }

    /// A tagged fence anywhere in the response suppresses untagged blocks
    /// entirely, so a trailing block of program output cannot win.
    #[test]
    fn test_extract_prefers_tagged_fences_over_later_untagged_ones() {
        let input = "```rust
pub fn f() -> i32 { 1 }
```
Output:
```
[1.0, 2.0, 3.0]
```";
        let blocks = extract_rust_code_blocks(input);
        assert_eq!(blocks, vec!["pub fn f() -> i32 { 1 }"]);
    }

    /// Every untagged block counts when the response tags nothing.
    #[test]
    fn test_extract_collects_all_untagged_blocks_in_order() {
        let input = "```\nfn first() {}\n```\nand\n```\nfn second() {}\n```";
        assert_eq!(
            extract_rust_code_blocks(input),
            vec!["fn first() {}", "fn second() {}"]
        );
    }
}

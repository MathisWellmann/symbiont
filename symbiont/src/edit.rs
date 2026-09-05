// SPDX-License-Identifier: MPL-2.0
//! Edits: how a repair round changes the previous candidate without
//! retyping it.
//!
//! After a rejected attempt the agent holds a candidate that is nearly
//! right. Asking for the whole function again costs output tokens and,
//! with a small model, invites new mistakes in the parts that were fine.
//! Instead the response may describe a change to the previous candidate,
//! the *base*, in three forms, from coarse to fine:
//!
//! 1. **Item replacement.** A ```` ```rust ```` block whose top-level items
//!    replace the base's items of the same name (functions by identifier,
//!    `use` declarations by their text). Items the block does not name stay
//!    as they are. A block that names every declared function is a full
//!    rewrite, which is the pre-existing behaviour.
//!
//! 2. **Search and replace.** A ```` ```rust-edit ```` block holding one or
//!    more `<<<<<<< SEARCH` / `=======` / `>>>>>>> REPLACE` hunks. The
//!    search text must match exactly one place in the base. The match is
//!    made on *tokens*, not on characters, so the agent's indentation and
//!    line breaks do not have to reproduce the base's.
//!
//! 3. **Diagnostic anchors.** A line `E<n> => <replacement>` (or a block
//!    `E<n> =>` followed by the replacement on the next lines, terminated by
//!    a blank line or the block's end) replaces the primary span of the
//!    `n`-th error of the previous compiler report. Nothing has to be
//!    located; the compiler already did that.
//!
//! Forms 2 and 3 can share a block. Every edit is resolved against the
//! unmodified base first, then all are applied from the end of the text
//! towards the start, so no edit's range moves another's. Two edits that
//! overlap are an error: the agent asked for two different texts in one
//! place.
//!
//! The result is a candidate like any other. It goes through the same
//! parse, validation and build as a complete response, so an edit that
//! breaks the syntax is caught in microseconds, without a build.

use std::ops::Range;

use proc_macro2::{
    TokenStream,
    TokenTree,
};

use crate::{
    EditRecord,
    diagnostics::Diagnostic,
    parser::Fence,
};

/// The text a repair round edits, together with the errors the agent was
/// shown about it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct EditBase {
    /// The previous candidate, as the compiler saw it.
    source: String,
    /// The compiler errors of that candidate, in the order the agent saw
    /// them: `E1` is `diagnostics[0]`. Empty when the previous attempt
    /// failed before compilation.
    diagnostics: Vec<Diagnostic>,
}

impl EditBase {
    /// A base of `source` whose reported errors are `diagnostics`.
    pub(crate) fn new(source: String, diagnostics: Vec<Diagnostic>) -> Self {
        Self {
            source,
            diagnostics,
        }
    }

    /// The previous candidate, as the compiler saw it.
    pub(crate) fn source(&self) -> &str {
        &self.source
    }
}

/// Why a response's edits could not be applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EditError {
    /// A `SEARCH` text was not found in the base.
    SearchNotFound { search: String },
    /// A `SEARCH` text was found more than once.
    SearchAmbiguous { search: String, count: usize },
    /// A `SEARCH` text does not tokenize as Rust, so it cannot be matched.
    SearchNotRust { search: String, reason: String },
    /// `E<n>` names an error the previous report did not have.
    UnknownAnchor { anchor: usize, available: usize },
    /// The error `E<n>` refers to has no span in the candidate to replace.
    AnchorWithoutSpan { anchor: usize },
    /// Two edits want to change the same text.
    Overlap,
    /// A `rust-edit` block did not follow the `<<<<<<< SEARCH` /
    /// `=======` / `>>>>>>> REPLACE` form.
    Malformed { reason: String },
    /// A replacement item names nothing in the base and is not a function
    /// or `use`, so there is nothing to replace and nowhere to put it.
    UnplaceableItem { item: String },
}

impl std::fmt::Display for EditError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SearchNotFound { search } => write!(
                f,
                "the SEARCH text was not found in your previous code:\n{search}"
            ),
            Self::SearchAmbiguous { search, count } => write!(
                f,
                "the SEARCH text matches {count} places in your previous code; include more \
                 context so it matches exactly one:\n{search}"
            ),
            Self::SearchNotRust { search, reason } => write!(
                f,
                "the SEARCH text is not valid Rust tokens ({reason}):\n{search}"
            ),
            Self::UnknownAnchor { anchor, available } => write!(
                f,
                "E{anchor} does not exist; the previous report had {available} error(s)"
            ),
            Self::AnchorWithoutSpan { anchor } => write!(
                f,
                "E{anchor} has no location in your code to replace; use a SEARCH/REPLACE edit"
            ),
            Self::Overlap => f.write_str("two edits change the same text; merge them into one"),
            Self::Malformed { reason } => write!(f, "malformed edit block: {reason}"),
            Self::UnplaceableItem { item } => write!(
                f,
                "the item `{item}` replaces nothing in your previous code; only functions and \
                 `use` declarations are merged by name"
            ),
        }
    }
}

/// How a response related to the base.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Resolved {
    /// The response carried edits; this is the base with them applied.
    Edited {
        /// The new candidate.
        source: String,
        /// The edits that were applied, by form, for the trace.
        edits: EditRecord,
    },
    /// The response carried no edits. The code block, if any, is the whole
    /// candidate, as before.
    Whole,
}

/// Apply the edits in `fences` to `base`.
///
/// Returns [`Resolved::Whole`] when the response carries no edits: no edit
/// block, and a code block (if any) that names every declared function. In
/// that case the caller treats the code block as a complete candidate,
/// exactly as it did before edits existed.
///
/// A code block that defines only some of the declared functions is an item
/// edit. It is read from the *last* code block, the same block a whole
/// response is read from. A code block beside edit blocks that is not an
/// item edit is most likely the model quoting context; the edit blocks are
/// the deliberate form and win.
pub(crate) fn resolve(
    base: &EditBase,
    fences: &[Fence],
    declared: &[&str],
) -> Result<Resolved, EditError> {
    let mut replacements: Vec<Replacement> = Vec::new();
    let mut edits = EditRecord::default();
    for block in fences.iter().filter(|fence| fence.is_edit()) {
        replacements.extend(parse_edit_block(block.body(), base, &mut edits)?);
    }

    if let Some(block) = crate::parser::code_blocks(fences).last()
        && let Ok(file) = syn::parse_file(block.body())
        && !replaces_everything(&file, declared)
        && let Ok(base_file) = syn::parse_file(&base.source)
    {
        let merged = merge_items(&base.source, &base_file, block.body(), &file)?;
        edits.items = merged.len();
        replacements.extend(merged);
    }

    if replacements.is_empty() {
        return Ok(Resolved::Whole);
    }
    debug_assert_eq!(edits.total(), replacements.len());
    let source = apply(&base.source, replacements)?;
    Ok(Resolved::Edited { source, edits })
}

/// One resolved change: replace `range` of the base with `text`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Replacement {
    range: Range<usize>,
    text: String,
}

/// Apply `replacements` to `base`, last range first. Overlaps are an error.
fn apply(base: &str, mut replacements: Vec<Replacement>) -> Result<String, EditError> {
    replacements.sort_by_key(|replacement| std::cmp::Reverse(replacement.range.start));
    let mut out = base.to_string();
    let mut applied_from = base.len();
    for replacement in replacements {
        if replacement.range.end > applied_from {
            return Err(EditError::Overlap);
        }
        out.replace_range(replacement.range.clone(), &replacement.text);
        applied_from = replacement.range.start;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Form 1: item replacement
// ---------------------------------------------------------------------------

/// Does `file` define every declared function? Then it is a whole
/// candidate, not an edit.
fn replaces_everything(file: &syn::File, declared: &[&str]) -> bool {
    declared.iter().all(|name| {
        file.items
            .iter()
            .any(|item| matches!(item, syn::Item::Fn(f) if f.sig.ident == name))
    })
}

/// The replacements that swap the items of `edit` into `base`, by name.
///
/// Every item of `edit` must match one of `base`'s: a function of the same
/// name, or a `use` with the same tokens. An item without a match is an
/// error rather than an append, because there is no right place to put it
/// and a silently appended duplicate would produce a confusing compiler
/// error on the next build.
///
/// Returns an empty list when `edit` holds no items, which the caller reads
/// as "not an item edit".
fn merge_items(
    base_src: &str,
    base: &syn::File,
    edit_src: &str,
    edit: &syn::File,
) -> Result<Vec<Replacement>, EditError> {
    let mut replacements = Vec::with_capacity(edit.items.len());
    for item in &edit.items {
        let Some(target) = base
            .items
            .iter()
            .find(|candidate| same_item(candidate, item))
        else {
            return Err(EditError::UnplaceableItem {
                item: item_name(item),
            });
        };
        replacements.push(Replacement {
            range: item_range(base_src, target),
            text: edit_src[item_range(edit_src, item)].to_string(),
        });
    }
    Ok(replacements)
}

/// Do two items denote the same thing for the purpose of replacement?
fn same_item(a: &syn::Item, b: &syn::Item) -> bool {
    use quote::ToTokens as _;
    match (a, b) {
        (syn::Item::Fn(a), syn::Item::Fn(b)) => a.sig.ident == b.sig.ident,
        (syn::Item::Use(a), syn::Item::Use(b)) => {
            a.tree.to_token_stream().to_string() == b.tree.to_token_stream().to_string()
        }
        (syn::Item::Const(a), syn::Item::Const(b)) => a.ident == b.ident,
        (syn::Item::Struct(a), syn::Item::Struct(b)) => a.ident == b.ident,
        (syn::Item::Enum(a), syn::Item::Enum(b)) => a.ident == b.ident,
        (syn::Item::Type(a), syn::Item::Type(b)) => a.ident == b.ident,
        (syn::Item::Trait(a), syn::Item::Trait(b)) => a.ident == b.ident,
        (syn::Item::Impl(a), syn::Item::Impl(b)) => {
            a.self_ty.to_token_stream().to_string() == b.self_ty.to_token_stream().to_string()
                && a.trait_
                    .as_ref()
                    .map(|(path, _)| path.to_token_stream().to_string())
                    == b.trait_
                        .as_ref()
                        .map(|(path, _)| path.to_token_stream().to_string())
        }
        _ => false,
    }
}

/// A short name of `item` for error messages.
fn item_name(item: &syn::Item) -> String {
    use quote::ToTokens as _;
    match item {
        syn::Item::Fn(f) => format!("fn {}", f.sig.ident),
        syn::Item::Use(u) => format!("use {}", u.tree.to_token_stream()),
        syn::Item::Const(c) => format!("const {}", c.ident),
        syn::Item::Struct(s) => format!("struct {}", s.ident),
        syn::Item::Enum(e) => format!("enum {}", e.ident),
        syn::Item::Type(t) => format!("type {}", t.ident),
        syn::Item::Trait(t) => format!("trait {}", t.ident),
        syn::Item::Impl(i) => format!("impl {}", i.self_ty.to_token_stream()),
        other => {
            let mut text = other.to_token_stream().to_string();
            text.truncate(text.floor_char_boundary(60));
            text
        }
    }
}

/// The byte range of `item` in `src`, including its attributes and doc
/// comments, up to and including its last token.
fn item_range(src: &str, item: &syn::Item) -> Range<usize> {
    use quote::ToTokens as _;
    let tokens = item.to_token_stream();
    let mut range: Option<Range<usize>> = None;
    for token in tokens {
        let span = token.span().byte_range();
        range = Some(match range {
            None => span,
            Some(current) => current.start.min(span.start)..current.end.max(span.end),
        });
    }
    let range = range.unwrap_or(0..0);
    // Doc comments are attributes and thus part of the token stream, but a
    // preceding plain comment is not; it belongs to the item visually, so
    // start at the beginning of the item's first line.
    let line_start = src[..range.start].rfind('\n').map_or(0, |idx| idx + 1);
    let prefix = &src[line_start..range.start];
    let start = if prefix.trim().is_empty() {
        line_start
    } else {
        range.start
    };
    start..range.end
}

// ---------------------------------------------------------------------------
// Forms 2 and 3: rust-edit blocks
// ---------------------------------------------------------------------------

const SEARCH_MARKER: &str = "<<<<<<<";
const DIVIDER: &str = "=======";
const REPLACE_MARKER: &str = ">>>>>>>";

/// Parse one `rust-edit` block into replacements against `base`, counting
/// each form into `edits`.
fn parse_edit_block(
    body: &str,
    base: &EditBase,
    edits: &mut EditRecord,
) -> Result<Vec<Replacement>, EditError> {
    let mut replacements = Vec::new();
    let lines: Vec<&str> = body.lines().collect();
    let mut idx = 0;
    while idx < lines.len() {
        let line = lines[idx].trim_end();
        if line.trim().is_empty() {
            idx += 1;
        } else if line.trim_start().starts_with(SEARCH_MARKER) {
            let (hunk, next) = parse_hunk(&lines, idx)?;
            replacements.push(resolve_search(base, &hunk.search, hunk.replace)?);
            edits.hunks += 1;
            idx = next;
        } else if let Some((anchor, rest)) = parse_anchor_head(line) {
            let (replacement, next) = anchor_replacement(&lines, idx, rest);
            replacements.push(resolve_anchor(base, anchor, replacement)?);
            edits.anchors += 1;
            idx = next;
        } else {
            return Err(EditError::Malformed {
                reason: format!(
                    "expected `{SEARCH_MARKER} SEARCH` or `E<n> =>`, found: {}",
                    line.trim()
                ),
            });
        }
    }
    Ok(replacements)
}

/// A parsed `SEARCH`/`REPLACE` hunk.
struct Hunk {
    search: String,
    replace: String,
}

/// Parse the hunk opening at `lines[start]`. Returns it and the index of
/// the line after its `>>>>>>> REPLACE` marker.
fn parse_hunk(lines: &[&str], start: usize) -> Result<(Hunk, usize), EditError> {
    let mut idx = start + 1;
    let mut search = Vec::new();
    while idx < lines.len() && !lines[idx].trim_start().starts_with(DIVIDER) {
        if lines[idx].trim_start().starts_with(SEARCH_MARKER)
            || lines[idx].trim_start().starts_with(REPLACE_MARKER)
        {
            return Err(EditError::Malformed {
                reason: format!("`{SEARCH_MARKER} SEARCH` without a `{DIVIDER}` divider"),
            });
        }
        search.push(lines[idx]);
        idx += 1;
    }
    if idx >= lines.len() {
        return Err(EditError::Malformed {
            reason: format!("`{SEARCH_MARKER} SEARCH` without a `{DIVIDER}` divider"),
        });
    }
    idx += 1;
    let mut replace = Vec::new();
    while idx < lines.len() && !lines[idx].trim_start().starts_with(REPLACE_MARKER) {
        if lines[idx].trim_start().starts_with(SEARCH_MARKER) {
            return Err(EditError::Malformed {
                reason: format!("`{DIVIDER}` without a `{REPLACE_MARKER} REPLACE` marker"),
            });
        }
        replace.push(lines[idx]);
        idx += 1;
    }
    if idx >= lines.len() {
        return Err(EditError::Malformed {
            reason: format!("`{DIVIDER}` without a `{REPLACE_MARKER} REPLACE` marker"),
        });
    }
    Ok((
        Hunk {
            search: search.join("\n"),
            replace: replace.join("\n"),
        },
        idx + 1,
    ))
}

/// `E<n> => rest`: the anchor number and whatever follows the arrow.
fn parse_anchor_head(line: &str) -> Option<(usize, &str)> {
    let rest = line.trim_start().strip_prefix('E')?;
    let digits_end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    if digits_end == 0 {
        return None;
    }
    let anchor: usize = rest[..digits_end].parse().ok()?;
    let after = rest[digits_end..].trim_start().strip_prefix("=>")?;
    Some((anchor, after.trim_start()))
}

/// The replacement text of an anchor edit at `lines[start]` with `inline`
/// after the arrow. An empty `inline` means the replacement is the block of
/// lines that follows, up to a blank line or the next edit. Returns the text
/// and the index of the line after it.
fn anchor_replacement(lines: &[&str], start: usize, inline: &str) -> (String, usize) {
    if !inline.is_empty() {
        return (inline.to_string(), start + 1);
    }
    let mut idx = start + 1;
    let mut text = Vec::new();
    while idx < lines.len() {
        let line = lines[idx];
        if line.trim().is_empty()
            || line.trim_start().starts_with(SEARCH_MARKER)
            || parse_anchor_head(line).is_some()
        {
            break;
        }
        text.push(line);
        idx += 1;
    }
    (text.join("\n"), idx)
}

/// The replacement of the primary span of error `anchor` (1-based).
fn resolve_anchor(
    base: &EditBase,
    anchor: usize,
    replacement: String,
) -> Result<Replacement, EditError> {
    let diagnostic = anchor
        .checked_sub(1)
        .and_then(|idx| base.diagnostics.get(idx))
        .ok_or(EditError::UnknownAnchor {
            anchor,
            available: base.diagnostics.len(),
        })?;
    let span = diagnostic
        .spans
        .iter()
        .find(|span| span.is_primary)
        .or(diagnostic.spans.first())
        .ok_or(EditError::AnchorWithoutSpan { anchor })?;
    Ok(Replacement {
        range: span.bytes.clone(),
        text: replacement,
    })
}

/// The replacement of the unique place in the base whose tokens are
/// `search`.
///
/// Both sides are tokenized with `proc_macro2`; the match is a run of the
/// base's tokens whose kinds and texts equal the search's, so whitespace,
/// line breaks and comments do not have to agree. The replaced range runs
/// from the first matched token's start to the last one's end, so the text
/// around the match (indentation, the trailing newline) is preserved.
fn resolve_search(
    base: &EditBase,
    search: &str,
    replace: String,
) -> Result<Replacement, EditError> {
    let needle: Vec<Token> = tokens(search).map_err(|reason| EditError::SearchNotRust {
        search: search.to_string(),
        reason,
    })?;
    if needle.is_empty() {
        return Err(EditError::SearchNotRust {
            search: search.to_string(),
            reason: "it is empty".to_string(),
        });
    }
    let haystack = tokens(&base.source).map_err(|reason| EditError::SearchNotRust {
        search: search.to_string(),
        reason: format!("the base does not tokenize: {reason}"),
    })?;
    let matches: Vec<usize> = (0..haystack.len().saturating_sub(needle.len() - 1))
        .filter(|&start| {
            haystack[start..start + needle.len()]
                .iter()
                .zip(&needle)
                .all(|(a, b)| a.text == b.text)
        })
        .collect();
    match matches.as_slice() {
        [] => Err(EditError::SearchNotFound {
            search: search.to_string(),
        }),
        [start] => {
            let first = &haystack[*start];
            let last = &haystack[start + needle.len() - 1];
            Ok(Replacement {
                range: first.range.start..last.range.end,
                text: replace,
            })
        }
        many => Err(EditError::SearchAmbiguous {
            search: search.to_string(),
            count: many.len(),
        }),
    }
}

/// One leaf token with its byte range in the text it was lexed from.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Token {
    text: String,
    range: Range<usize>,
}

/// The leaf tokens of `text`, in order, with group delimiters as their own
/// tokens. Fails when `text` is not lexically valid Rust (an unbalanced
/// bracket, an unterminated string).
fn tokens(text: &str) -> Result<Vec<Token>, String> {
    let stream: TokenStream = text
        .parse()
        .map_err(|e: proc_macro2::LexError| e.to_string())?;
    let mut out = Vec::new();
    flatten(stream, &mut out);
    Ok(out)
}

fn flatten(stream: TokenStream, out: &mut Vec<Token>) {
    for tree in stream {
        match tree {
            TokenTree::Group(group) => {
                let (open, close) = match group.delimiter() {
                    proc_macro2::Delimiter::Parenthesis => ("(", ")"),
                    proc_macro2::Delimiter::Brace => ("{", "}"),
                    proc_macro2::Delimiter::Bracket => ("[", "]"),
                    proc_macro2::Delimiter::None => ("", ""),
                };
                let open_range = group.span_open().byte_range();
                let close_range = group.span_close().byte_range();
                if !open.is_empty() {
                    out.push(Token {
                        text: open.to_string(),
                        range: open_range,
                    });
                }
                flatten(group.stream(), out);
                if !close.is_empty() {
                    out.push(Token {
                        text: close.to_string(),
                        range: close_range,
                    });
                }
            }
            leaf => out.push(Token {
                text: leaf.to_string(),
                range: leaf.span().byte_range(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        diagnostics::DiagnosticSpan,
        parser::fences,
    };

    /// A base as the compiler saw it: it parses (a candidate that did not
    /// parse never reaches the compiler and never becomes a base) but has a
    /// type error, `len / 2` where a `f64` is due.
    const BASE: &str = "fn sort(data: &mut [f64], len: usize) {\n    let mid: f64 = len / 2;\n    for i in 0..len {\n        data.swap(i, i);\n    }\n}\n\nfn helper(x: f64) -> f64 {\n    x * 2.0\n}";

    fn base() -> EditBase {
        EditBase::new(BASE.to_string(), Vec::new())
    }

    fn base_with_error_at(needle: &str) -> EditBase {
        let start = BASE.find(needle).expect("needle in base");
        let line = BASE[..start].matches('\n').count() + 1;
        EditBase::new(
            BASE.to_string(),
            vec![Diagnostic {
                code: Some("E0308".to_string()),
                message: "mismatched types".to_string(),
                spans: vec![DiagnosticSpan {
                    bytes: start..start + needle.len(),
                    line_start: line,
                    column_start: 1,
                    line_end: line,
                    column_end: 1,
                    is_primary: true,
                    label: None,
                    text: needle.to_string(),
                }],
                suggestions: Vec::new(),
                rendered: String::new(),
            }],
        )
    }

    fn resolve_response(base: &EditBase, response: &str) -> Result<Resolved, EditError> {
        resolve(base, &fences(response), &["sort"])
    }

    fn edited(result: Result<Resolved, EditError>) -> String {
        match result {
            Ok(Resolved::Edited { source, .. }) => source,
            other => panic!("expected an edit, got {other:?}"),
        }
    }

    #[test]
    fn search_replace_matches_on_tokens_not_whitespace() {
        // The base has `let mid: f64 = len / 2;` on one line; the search
        // spreads it over two and spaces it differently.
        let response = "```rust-edit\n<<<<<<< SEARCH\nlet mid:f64 =\n    len/2;\n=======\nlet mid = len as f64 / 2.0;\n>>>>>>> REPLACE\n```";
        let source = edited(resolve_response(&base(), response));
        assert!(
            source.contains("    let mid = len as f64 / 2.0;\n    for i"),
            "{source}"
        );
        assert!(source.contains("fn helper"), "the rest of the base is kept");
    }

    #[test]
    fn several_hunks_apply_together() {
        let response = "```rust-edit\n<<<<<<< SEARCH\nlen / 2\n=======\nlen as f64 / 2.0\n>>>>>>> REPLACE\n<<<<<<< SEARCH\nx * 2.0\n=======\nx * 3.0\n>>>>>>> REPLACE\n```";
        let result = resolve_response(&base(), response);
        assert!(
            matches!(
                result,
                Ok(Resolved::Edited {
                    edits: EditRecord {
                        anchors: 0,
                        hunks: 2,
                        items: 0
                    },
                    ..
                })
            ),
            "{result:?}"
        );
        let source = edited(result);
        assert!(
            source.contains("let mid: f64 = len as f64 / 2.0;"),
            "{source}"
        );
        assert!(source.contains("x * 3.0"));
    }

    #[test]
    fn a_search_that_is_not_found_is_reported() {
        let response = "```rust-edit\n<<<<<<< SEARCH\nlet nope = 1;\n=======\nlet yes = 1;\n>>>>>>> REPLACE\n```";
        assert!(matches!(
            resolve_response(&base(), response),
            Err(EditError::SearchNotFound { .. })
        ));
    }

    #[test]
    fn an_ambiguous_search_is_reported_with_its_count() {
        // `len` appears three times in the base.
        let response = "```rust-edit\n<<<<<<< SEARCH\nlen\n=======\nn\n>>>>>>> REPLACE\n```";
        assert!(matches!(
            resolve_response(&base(), response),
            Err(EditError::SearchAmbiguous { count: 3, .. })
        ));
    }

    #[test]
    fn an_unbalanced_search_is_reported_as_not_rust() {
        let response = "```rust-edit\n<<<<<<< SEARCH\nfor i in 0..len {\n=======\nfor i in 0..len.min(3) {\n>>>>>>> REPLACE\n```";
        assert!(matches!(
            resolve_response(&base(), response),
            Err(EditError::SearchNotRust { .. })
        ));
    }

    #[test]
    fn a_malformed_hunk_is_reported() {
        let response = "```rust-edit\n<<<<<<< SEARCH\nlen / 2\nlen as f64 / 2.0\n```";
        assert!(matches!(
            resolve_response(&base(), response),
            Err(EditError::Malformed { .. })
        ));
    }

    #[test]
    fn an_edit_block_tagged_rust_is_still_an_edit() {
        let response = "```rust\n<<<<<<< SEARCH\nx * 2.0\n=======\nx * 4.0\n>>>>>>> REPLACE\n```";
        let source = edited(resolve_response(&base(), response));
        assert!(source.contains("x * 4.0"));
    }

    #[test]
    fn anchor_replaces_the_primary_span_inline() {
        let base = base_with_error_at("len / 2");
        let source = edited(resolve_response(
            &base,
            "```rust-edit\nE1 => len as f64 / 2.0\n```",
        ));
        assert!(
            source.contains("let mid: f64 = len as f64 / 2.0;\n"),
            "{source}"
        );
    }

    #[test]
    fn anchor_replacement_may_span_lines() {
        let base = base_with_error_at("data.swap(i, i);");
        let response = "```rust-edit\nE1 =>\nif i + 1 < len {\n    data.swap(i, i + 1);\n}\n```";
        let source = edited(resolve_response(&base, response));
        assert!(
            source.contains("        if i + 1 < len {\n    data.swap(i, i + 1);\n}\n"),
            "{source}"
        );
    }

    #[test]
    fn unknown_anchor_is_reported_with_the_available_count() {
        let with_one = base_with_error_at("len / 2");
        assert_eq!(
            resolve_response(&with_one, "```rust-edit\nE2 => 1\n```"),
            Err(EditError::UnknownAnchor {
                anchor: 2,
                available: 1
            })
        );
        assert!(matches!(
            resolve_response(&base(), "```rust-edit\nE1 => 1\n```"),
            Err(EditError::UnknownAnchor { available: 0, .. })
        ));
    }

    #[test]
    fn overlapping_edits_are_rejected() {
        let base = base_with_error_at("len / 2");
        let response = "```rust-edit\nE1 => len as f64 / 3.0\n<<<<<<< SEARCH\nlet mid: f64 = len / 2;\n=======\nlet mid = 0.0;\n>>>>>>> REPLACE\n```";
        assert_eq!(resolve_response(&base, response), Err(EditError::Overlap));
    }

    #[test]
    fn a_block_with_one_of_two_functions_replaces_that_function() {
        let response = "Fixing the helper only:\n```rust\nfn helper(x: f64) -> f64 {\n    x * 2.0 + 1.0\n}\n```";
        let source = edited(resolve_response(&base(), response));
        assert!(
            source.starts_with(
                "fn sort(data: &mut [f64], len: usize) {\n    let mid: f64 = len / 2;\n"
            ),
            "{source}"
        );
        assert!(
            source.ends_with("fn helper(x: f64) -> f64 {\n    x * 2.0 + 1.0\n}"),
            "{source}"
        );
    }

    #[test]
    fn a_block_with_every_declared_function_is_a_whole_response() {
        let response = "```rust\nfn sort(data: &mut [f64], len: usize) {}\n```";
        assert_eq!(resolve_response(&base(), response), Ok(Resolved::Whole));
    }

    #[test]
    fn a_block_naming_nothing_in_the_base_is_reported() {
        let response = "```rust\nfn brand_new() {}\n```";
        assert!(matches!(
            resolve_response(&base(), response),
            Err(EditError::UnplaceableItem { ref item }) if item == "fn brand_new"
        ));
    }

    #[test]
    fn a_response_without_edits_is_whole() {
        assert_eq!(
            resolve_response(&base(), "no code here"),
            Ok(Resolved::Whole)
        );
    }

    #[test]
    fn item_range_covers_doc_comments_and_attributes() {
        let src = "use std::cmp::Ordering;\n\n/// Docs.\n#[inline]\nfn f() -> Ordering {\n    Ordering::Less\n}\n";
        let file = syn::parse_file(src).expect("parses");
        let range = item_range(src, &file.items[1]);
        assert_eq!(
            &src[range],
            "/// Docs.\n#[inline]\nfn f() -> Ordering {\n    Ordering::Less\n}"
        );
    }

    #[test]
    fn edit_errors_explain_themselves() {
        let text = EditError::SearchAmbiguous {
            search: "len".to_string(),
            count: 3,
        }
        .to_string();
        assert!(text.contains("matches 3 places"), "{text}");
        assert!(text.contains("include more context"), "{text}");
    }
}

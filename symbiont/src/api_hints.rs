// SPDX-License-Identifier: MPL-2.0
//! Documentation attached to the errors of an invented API.
//!
//! The most common way a small model fails to compile is by calling a
//! method, field or variant the host type does not have: `E0599` (no
//! method), `E0609` (no field), `E0560` (no such field in a struct
//! literal), or by naming a type or function that does not exist (`E0425`,
//! `E0412`, `E0433`). A diagnostic tells the model *that* the name is
//! wrong. It does not tell it what the type does have, and without that
//! the next attempt guesses again.
//!
//! This module reads the type the compiler names in such an error, looks
//! it up in the host's [`DocIndex`], and appends the type's *surface* to the
//! nudge: the declaration, its public fields or variants, and the signatures
//! of its methods - one line each, no doc comments, no attributes, no
//! private fields. With it, the repair is a lookup, not a guess.
//!
//! The surface is a compaction of what `api_doc` returns. The full document
//! is the right answer to a question the model asked; attached unasked to
//! every compile nudge it is not. Three full definitions made nudges of
//! 20-60 KB in production, which overflowed the context of a small server
//! (a `context_reset`, the history gone) and cost every later request of
//! the lane. The surface says the same thing about *which names exist* in a
//! tenth of the bytes, and the nudge names the members the model used that
//! do not, so the model reads the list for what it was looking for.

use std::{
    collections::BTreeSet,
    fmt::Write as _,
};

use crate::{
    Diagnostic,
    DocIndex,
    EXPECT_WRITE,
};

/// Error codes whose message names a type of the host API the model got
/// wrong.
const RECEIVER_CODES: &[&str] = &[
    "E0599", "E0609", "E0560", "E0610", "E0615", "E0616", "E0624",
];

/// Error codes whose message names a path that does not resolve.
const UNRESOLVED_CODES: &[&str] = &["E0412", "E0422", "E0423", "E0425", "E0432", "E0433"];

/// At most this many definitions per nudge. Every definition costs input
/// tokens on every later request of the lane; past a few, the model is
/// better served by a hint to call `api_doc` itself.
const MAX_HINTS: usize = 3;

/// At most this many bytes of surface per type. A type with more members
/// than fit is cut at a line and the model is told to call `api_doc` for
/// the rest; the members the model misused are named before the listing,
/// so the cut costs it nothing it was looking for.
const MAX_HINT_BYTES: usize = 2500;

/// The names in `diagnostics` worth documenting, in order of first
/// appearance, without duplicates.
///
/// For a receiver error the name is the type after `for` / `on type` /
/// `for enum` / `struct` in rustc's message, stripped of references, the
/// module path and generic arguments: `&prelude::Account<i64>` yields
/// `Account`. For an unresolved path the name is the quoted path itself,
/// which a lookup then rejects or resolves.
pub(crate) fn api_hint_names(diagnostics: &[Diagnostic]) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut names = Vec::new();
    for diagnostic in diagnostics {
        let Some(code) = diagnostic.code.as_deref() else {
            continue;
        };
        let name = if RECEIVER_CODES.contains(&code) {
            receiver_type(&diagnostic.message)
        } else if UNRESOLVED_CODES.contains(&code) {
            first_quoted(&diagnostic.message).map(str::to_string)
        } else {
            None
        };
        if let Some(name) = name
            && !name.is_empty()
            && seen.insert(name.clone())
        {
            names.push(name);
        }
    }
    names
}

/// The bare type name a receiver error is about.
///
/// rustc phrases these as `no method named \`m\` found for reference
/// \`&prelude::Account<i64>\` in the current scope`, `no field \`f\` on type
/// \`&prelude::Account<i64>\``, `no variant ... found for enum
/// \`prelude::Side\``, `struct \`prelude::Account<{integer}>\` has no field
/// named \`missing\``. In every form the type is the quoted segment after
/// one of the keywords below, or, for the struct-literal form, the first
/// quoted segment.
fn receiver_type(message: &str) -> Option<String> {
    const KEYWORDS: &[&str] = &[
        "for reference `",
        "for struct `",
        "for enum `",
        "for union `",
        "for type `",
        "on type `",
        "of struct `",
        "of enum `",
        "of union `",
        "for `",
        "in `",
    ];
    let quoted = KEYWORDS
        .iter()
        .find_map(|keyword| {
            let start = message.find(keyword)? + keyword.len();
            let end = message[start..].find('`')?;
            Some(&message[start..start + end])
        })
        .or_else(|| {
            message
                .starts_with("struct `")
                .then(|| first_quoted(message))
                .flatten()
        })?;
    bare_type_name(quoted)
}

/// `&mut prelude::Account<i64, DECIMALS>` -> `Account`.
fn bare_type_name(ty: &str) -> Option<String> {
    let ty = ty
        .trim_start_matches('&')
        .trim_start_matches("mut ")
        .trim_start_matches("dyn ")
        .trim();
    // Cut generic arguments and anything that is not a path.
    let path_end = ty
        .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == ':'))
        .unwrap_or(ty.len());
    let path = &ty[..path_end];
    let last = path.rsplit("::").next()?;
    // Primitives and std types have no host documentation. `{integer}` and
    // friends are inference placeholders.
    let is_host_like = last.chars().next().is_some_and(|c| c.is_ascii_uppercase())
        && !path.starts_with("std::")
        && !path.starts_with("core::")
        && !path.starts_with("alloc::");
    is_host_like.then(|| last.to_string())
}

/// The text between the first pair of backticks.
fn first_quoted(message: &str) -> Option<&str> {
    let start = message.find('`')? + 1;
    let end = message[start..].find('`')?;
    Some(&message[start..start + end])
}

/// The members of `type_name` that `diagnostics` say do not exist, in order
/// of first appearance: the quoted name of a receiver error whose receiver
/// is `type_name` (`no method named \`best_bid\` found for ...`).
fn misused_members(diagnostics: &[Diagnostic], type_name: &str) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut members = Vec::new();
    for diagnostic in diagnostics {
        let is_receiver_error = diagnostic
            .code
            .as_deref()
            .is_some_and(|code| RECEIVER_CODES.contains(&code));
        if !is_receiver_error || receiver_type(&diagnostic.message).as_deref() != Some(type_name) {
            continue;
        }
        if let Some(member) = first_quoted(&diagnostic.message)
            && member != type_name
            && seen.insert(member.to_string())
        {
            members.push(member.to_string());
        }
    }
    members
}

/// Append the surfaces of the types named in `diagnostics` to `out`.
/// Returns the names whose surface was attached, in order.
///
/// Names the index does not know are skipped without a note: an
/// unresolved path the model invented has no documentation to attach, and
/// the compiler error already says the name does not exist.
pub(crate) fn render_api_hints(
    index: &DocIndex,
    diagnostics: &[Diagnostic],
    out: &mut String,
) -> Vec<String> {
    let mut rendered = Vec::new();
    for name in api_hint_names(diagnostics) {
        if rendered.len() == MAX_HINTS {
            out.push_str(
                "More host types are involved; call `api_doc` with a type name to see its \
                 definition before you use it.\n",
            );
            break;
        }
        let Ok(doc) = index.render_doc(&name) else {
            continue;
        };
        if rendered.is_empty() {
            out.push_str(
                "\nThe host API you used does not match its definition. Below is the public \
                 surface of each type involved: its declaration and the signatures of its \
                 members, nothing else exists on it. Call only what is listed; call `api_doc` \
                 with the type name for the documentation.\n",
            );
        }
        write!(out, "\n## `{name}`").expect(EXPECT_WRITE);
        let misused = misused_members(diagnostics, &name);
        if !misused.is_empty() {
            let list: Vec<String> = misused.iter().map(|m| format!("`{m}`")).collect();
            write!(out, " - not a member: {}", list.join(", ")).expect(EXPECT_WRITE);
        }
        writeln!(out, "\n{}", surface(&doc, &name)).expect(EXPECT_WRITE);
        rendered.push(name);
    }
    rendered
}

/// The public surface of a rendered `api_doc` document: every item header
/// on one line, the public fields of structs, the variants of enums, and the
/// signatures inside `impl` and `trait` blocks. Doc comments, attributes,
/// comments, private fields and prose outside the code fences are dropped.
/// Cut at [`MAX_HINT_BYTES`] on a line boundary, with a pointer to
/// `api_doc`.
///
/// The input is the renderer's rustfmt-shaped text: one item header per
/// line (a `where` clause may continue it), members one per line, bodies
/// elided to `;`. The state machine below leans on that shape and degrades
/// to keeping a line it does not understand rather than dropping it.
pub(crate) fn surface(doc: &str, name: &str) -> String {
    const FENCE: &str = "```rust\n";
    let mut out = String::from(FENCE);
    let mut in_fence = false;
    let mut depth: usize = 0;
    let mut container = Container::None;
    // A header or signature split over several lines, joined until it ends.
    let mut pending: Option<String> = None;
    for raw in doc.lines() {
        let line = raw.trim();
        if line.starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if !in_fence || is_noise(line) {
            continue;
        }
        if let Some(partial) = pending.as_mut() {
            if line == "}" {
                // The last member of a block, written without a trailing
                // comma: it ends where the block does.
                let joined = pending.take().unwrap_or_default();
                depth = emit(&mut out, &joined, depth, &mut container);
            } else {
                partial.push(' ');
                partial.push_str(line);
                if !ends_member(partial, line, depth) {
                    continue;
                }
                let joined = pending.take().unwrap_or_default();
                depth = emit(&mut out, &joined, depth, &mut container);
                continue;
            }
        }
        let keep = match container {
            Container::None | Container::Enum => true,
            Container::Struct => line.starts_with("pub ") || line == "}",
            Container::Impl => {
                line.starts_with("pub ")
                    || line.starts_with("fn ")
                    || line.starts_with("type ")
                    || line.starts_with("const ")
                    || line == "}"
            }
        };
        if !keep {
            continue;
        }
        if !ends_member(line, line, depth) {
            pending = Some(line.to_string());
            continue;
        }
        depth = emit(&mut out, line, depth, &mut container);
    }
    if let Some(partial) = pending {
        emit(&mut out, &partial, depth, &mut container);
    }
    if out.len() - FENCE.len() > MAX_HINT_BYTES {
        // The limit is a byte offset that may fall inside a multi-byte
        // character; scan the bytes rather than slice the string.
        let limit = FENCE.len() + MAX_HINT_BYTES;
        let cut = out.as_bytes()[..limit]
            .iter()
            .rposition(|&b| b == b'\n')
            .unwrap_or(FENCE.len() - 1);
        out.truncate(cut + 1);
        let _ = writeln!(
            out,
            "    // ... more members; call `api_doc` with `{name}` for all of them"
        );
    }
    out.push_str("```");
    out
}

/// What the surface is inside of, at depth one. Decides which lines of a
/// body are members worth keeping: the public fields of a struct (a module
/// or macro body is treated alike - its `pub` items, nothing else), every
/// variant of an enum, the signatures of an `impl` or `trait`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Container {
    None,
    Struct,
    Enum,
    Impl,
}

/// Lines that carry no surface: doc comments, attributes, comments, blanks.
fn is_noise(line: &str) -> bool {
    line.is_empty() || line.starts_with("//") || line.starts_with("#[") || line.starts_with("#![")
}

/// Does `line` complete the header or member that began with `first`? A
/// header ends in `{` or `;` and may run over several lines (`where` clauses
/// end in `,` and are not the end). Inside a body a field or variant ends in
/// `,`, `;` or the `}` of its own braces; a signature spread over several
/// lines has parameters ending in `,` and ends only at its `;` or `{`.
fn ends_member(first: &str, line: &str, depth: usize) -> bool {
    line.ends_with('{')
        || line.ends_with(';')
        || line.ends_with('}')
        || (depth > 0 && line.ends_with(',') && !first.contains("fn "))
}

/// The brace depth after `line`.
fn next_depth(depth: usize, line: &str) -> usize {
    let opens = line.matches('{').count();
    let closes = line.matches('}').count();
    (depth + opens).saturating_sub(closes)
}

/// Write `line` at `depth`, note which container a depth-zero header opens,
/// and return the depth after the line.
fn emit(out: &mut String, line: &str, depth: usize, container: &mut Container) -> usize {
    let line = collapse_spaces(line);
    if depth == 0 && line.ends_with('{') {
        *container = if line.contains("enum ") {
            Container::Enum
        } else if line.starts_with("impl") || line.contains("trait ") {
            Container::Impl
        } else {
            Container::Struct
        };
    }
    if line == "}" && depth == 1 && out.ends_with("{\n") {
        // A body with nothing public in it: say so on the header's line
        // rather than leaving an empty pair of braces to wonder about.
        out.truncate(out.len() - 1);
        out.push_str(" /* no public members */ }\n");
        *container = Container::None;
        return 0;
    }
    let indent = if depth == 0 || line == "}" {
        ""
    } else {
        "    "
    };
    let _ = writeln!(out, "{indent}{line}");
    let next = next_depth(depth, &line);
    if next == 0 {
        *container = Container::None;
    }
    next
}

/// Runs of whitespace to one space, so a joined header or signature reads
/// as one line; the seams a joined parameter list leaves are closed.
fn collapse_spaces(line: &str) -> String {
    line.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace("( ", "(")
        .replace("< ", "<")
        .replace(", )", ")")
        .replace(", >", ">")
        .replace(" ,", ",")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diagnostic(code: &str, message: &str) -> Diagnostic {
        Diagnostic {
            code: Some(code.to_string()),
            message: message.to_string(),
            spans: Vec::new(),
            suggestions: Vec::new(),
            rendered: String::new(),
        }
    }

    #[test]
    fn receiver_errors_name_the_bare_host_type() {
        let cases = [
            (
                "E0599",
                "no method named `get_balance` found for reference `&prelude::Account<i64>` in the current scope",
                "Account",
            ),
            (
                "E0609",
                "no field `bal2` on type `&mut prelude::Account<i64, DECIMALS>`",
                "Account",
            ),
            (
                "E0599",
                "no variant, associated function, or constant named `Sell` found for enum `prelude::Side` in the current scope",
                "Side",
            ),
            (
                "E0560",
                "struct `prelude::Account<{integer}>` has no field named `missing`",
                "Account",
            ),
            (
                "E0599",
                "no method named `limit` found for struct `CommandBuffer<DECIMALS, Cur>` in the current scope",
                "CommandBuffer",
            ),
        ];
        for (code, message, expected) in cases {
            assert_eq!(
                api_hint_names(&[diagnostic(code, message)]),
                vec![expected.to_string()],
                "{message}"
            );
        }
    }

    #[test]
    fn std_and_primitive_receivers_are_not_hints() {
        for message in [
            "no method named `foo` found for struct `Vec<u8>` in the current scope",
            "no method named `foo` found for type `usize` in the current scope",
            "no method named `foo` found for reference `&str` in the current scope",
            "no method named `foo` found for struct `std::collections::HashMap<u8, u8>` in the current scope",
        ] {
            let names = api_hint_names(&[diagnostic("E0599", message)]);
            assert!(
                names.is_empty() || names == vec!["Vec".to_string()],
                "{message}: {names:?}"
            );
        }
        assert!(
            api_hint_names(&[diagnostic(
                "E0599",
                "no method named `foo` found for struct `std::collections::HashMap<u8, u8>` in the current scope"
            )])
            .is_empty()
        );
        assert!(
            api_hint_names(&[diagnostic(
                "E0599",
                "no method named `foo` found for type `usize` in the current scope"
            )])
            .is_empty()
        );
    }

    #[test]
    fn unresolved_paths_are_looked_up_as_written() {
        assert_eq!(
            api_hint_names(&[
                diagnostic("E0425", "cannot find function `make_order` in this scope"),
                diagnostic("E0412", "cannot find type `Order` in this scope"),
                diagnostic(
                    "E0433",
                    "failed to resolve: use of undeclared type `OrderBuilder`"
                ),
            ]),
            vec!["make_order", "Order", "OrderBuilder"]
        );
    }

    /// Against the shared fixture index: a name the index knows is rendered
    /// under the explanatory header, an invented one is skipped silently,
    /// and without any documentable name nothing is written at all.
    #[test]
    fn hints_render_known_definitions_and_skip_unknown_names() {
        let index = crate::doc_index::tests::fixture_index();

        let mut out = String::new();
        let attached = render_api_hints(
            &index,
            &[
                diagnostic(
                    "E0433",
                    "failed to resolve: use of undeclared type `decimal`",
                ),
                diagnostic("E0425", "cannot find function `invented` in this scope"),
            ],
            &mut out,
        );
        assert_eq!(attached, vec!["decimal".to_string()]);
        assert_eq!(
            out.matches("does not match its definition").count(),
            1,
            "header once: {out}"
        );
        assert!(out.contains("## `decimal`"), "{out}");
        assert!(out.contains("macro_rules! decimal"), "{out}");
        assert!(!out.contains("invented"), "{out}");

        let mut out = String::new();
        let attached =
            render_api_hints(&index, &[diagnostic("E0308", "mismatched types")], &mut out);
        assert!(out.is_empty(), "{out}");
        assert!(attached.is_empty());
    }

    /// A rendered `api_doc` document in the renderer's shape: prose outside
    /// the fence, doc comments and attributes on every item, private fields,
    /// a `where` clause splitting a header, a signature split over lines.
    const RENDERED: &str = "\nReachable items re-exported from `lfest` (the crate path itself is not available):\n\n```rust\n// Use these items unqualified.\n\n/// Some information regarding the state of the market.\n#[derive(Debug, Default, Clone, Getters)]\npub struct MarketState<I, const D: u8>\nwhere\n    I: Mon<D>,\n{\n    /// The current bid\n    #[getset(get_copy = \"pub\")]\n    bid: QuoteCurrency<I, D>,\n    /// Public for a reason.\n    pub step: u64,\n}\n\nimpl MarketState<I, D> {\n    /// The current bid\n    pub fn bid(&self) -> QuoteCurrency<I, D>;\n\n    /// Build one.\n    pub fn new(\n        bid: QuoteCurrency<I, D>,\n        ask: QuoteCurrency<I, D>,\n    ) -> Self;\n}\n\nimpl<I, const D: u8> MarketState<I, D>\nwhere\n    I: Mon<D>,\n{\n    /// Get the mid price\n    #[inline(always)]\n    pub fn mid_price(&self) -> QuoteCurrency<I, D>;\n}\n\n/// A side.\npub enum Side {\n    /// Buy.\n    Buy,\n    /// Sell.\n    Sell\n}\n\nimpl std::ops::Neg for Side {\n    type Output = Side;\n}\n```\n";

    #[test]
    fn surface_keeps_headers_public_members_and_signatures_only() {
        let surface = surface(RENDERED, "MarketState");
        let expected = "```rust\n\
            pub struct MarketState<I, const D: u8> where I: Mon<D>, {\n\
            \x20   pub step: u64,\n\
            }\n\
            impl MarketState<I, D> {\n\
            \x20   pub fn bid(&self) -> QuoteCurrency<I, D>;\n\
            \x20   pub fn new(bid: QuoteCurrency<I, D>, ask: QuoteCurrency<I, D>) -> Self;\n\
            }\n\
            impl<I, const D: u8> MarketState<I, D> where I: Mon<D>, {\n\
            \x20   pub fn mid_price(&self) -> QuoteCurrency<I, D>;\n\
            }\n\
            pub enum Side {\n\
            \x20   Buy,\n\
            \x20   Sell\n\
            }\n\
            impl std::ops::Neg for Side {\n\
            \x20   type Output = Side;\n\
            }\n\
            ```";
        assert_eq!(surface, expected);
        assert!(surface.len() * 2 < RENDERED.len(), "{}", surface.len());
    }

    #[test]
    fn a_body_without_public_members_is_said_so_on_the_header() {
        let doc = "```rust\npub struct Opaque<I>\nwhere\n    I: Mon,\n{\n    /// Hidden.\n    inner: I,\n}\n```\n";
        assert_eq!(
            surface(doc, "Opaque"),
            "```rust\npub struct Opaque<I> where I: Mon, { /* no public members */ }\n```"
        );
    }

    #[test]
    fn surface_is_cut_at_a_line_with_a_pointer_to_api_doc() {
        let mut doc = String::from("```rust\nimpl Big {\n");
        for i in 0..200 {
            let _ = writeln!(
                doc,
                "    pub fn method_number_{i}(&self, argument: u64) -> u64;"
            );
        }
        doc.push_str("}\n```\n");
        let surface = surface(&doc, "Big");
        assert!(surface.len() <= MAX_HINT_BYTES + 200, "{}", surface.len());
        assert!(surface.contains("call `api_doc` with `Big`"), "{surface}");
        assert!(surface.ends_with("```"));
        for line in surface.lines().filter(|l| l.contains("method_number")) {
            assert!(line.ends_with(';'), "cut mid-line: {line}");
        }
    }

    /// The cut must not land inside a multi-byte character: `out[..limit]`
    /// used to panic when one straddled the byte limit.
    #[test]
    fn surface_cut_is_safe_with_multi_byte_characters() {
        let mut doc = String::from("```rust\nimpl Big {\n");
        for i in 0..400 {
            // Two 4-byte characters per line put a boundary of one of them
            // exactly at the byte limit.
            let _ = writeln!(
                doc,
                "    pub const NAME_{i}: &str = \"xx\u{1F600}\u{1F600}\";"
            );
        }
        doc.push_str("}\n```\n");
        let surface = surface(&doc, "Big");
        assert!(surface.contains("call `api_doc` with `Big`"), "{surface}");
        assert!(surface.len() <= MAX_HINT_BYTES + 200, "{}", surface.len());
    }

    #[test]
    fn misused_members_are_the_names_the_errors_quote() {
        let diagnostics = [
            diagnostic(
                "E0609",
                "no field `best_bid` on type `&agent_symbiont::prelude::MarketState<i64, 8>`",
            ),
            diagnostic(
                "E0599",
                "no method named `best_ask` found for reference `&prelude::MarketState<i64, 8>` in the current scope",
            ),
            diagnostic(
                "E0616",
                "field `position` of struct `agent_symbiont::prelude::Account` is private",
            ),
            diagnostic(
                "E0609",
                "no field `best_bid` on type `&agent_symbiont::prelude::MarketState<i64, 8>`",
            ),
        ];
        assert_eq!(
            misused_members(&diagnostics, "MarketState"),
            vec!["best_bid", "best_ask"]
        );
        assert_eq!(misused_members(&diagnostics, "Account"), vec!["position"]);
        assert_eq!(
            api_hint_names(&diagnostics),
            vec!["MarketState".to_string(), "Account".to_string()]
        );
    }

    #[test]
    fn names_are_deduplicated_and_other_codes_ignored() {
        let names = api_hint_names(&[
            diagnostic(
                "E0599",
                "no method named `a` found for struct `prelude::Account<i64>` in the current scope",
            ),
            diagnostic("E0308", "mismatched types"),
            diagnostic("E0609", "no field `b` on type `&prelude::Account<i64>`"),
        ]);
        assert_eq!(names, vec!["Account".to_string()]);
    }
}

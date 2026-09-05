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
//! it up in the host's [`DocIndex`], and appends the type's definition to
//! the nudge: the declaration, the inherent methods, and the operator
//! impls. The same text `api_doc` would return had the model asked. With
//! it, the repair is a lookup, not a guess.

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

/// Append the definitions of the types named in `diagnostics` to `out`.
///
/// Names the index does not know are skipped without a note: an
/// unresolved path the model invented has no documentation to attach, and
/// the compiler error already says the name does not exist.
pub(crate) fn render_api_hints(index: &DocIndex, diagnostics: &[Diagnostic], out: &mut String) {
    let mut rendered = 0;
    for name in api_hint_names(diagnostics) {
        if rendered == MAX_HINTS {
            out.push_str(
                "More host types are involved; call `api_doc` with a type name to see its \
                 definition before you use it.\n",
            );
            break;
        }
        let Ok(doc) = index.render_doc(&name) else {
            continue;
        };
        if rendered == 0 {
            out.push_str(
                "\nThe host API you used does not match its definition. These are the exact \
                 definitions of the types involved; call only what is listed.\n",
            );
        }
        // `render_doc` returns a complete document with its own code fences,
        // so it is not wrapped in another one.
        writeln!(out, "\n## `{name}`\n{doc}").expect(EXPECT_WRITE);
        rendered += 1;
    }
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
        render_api_hints(
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
        assert_eq!(
            out.matches("does not match its definition").count(),
            1,
            "header once: {out}"
        );
        assert!(out.contains("## `decimal`"), "{out}");
        assert!(out.contains("macro_rules! decimal"), "{out}");
        assert!(!out.contains("invented"), "{out}");

        let mut out = String::new();
        render_api_hints(&index, &[diagnostic("E0308", "mismatched types")], &mut out);
        assert!(out.is_empty(), "{out}");
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

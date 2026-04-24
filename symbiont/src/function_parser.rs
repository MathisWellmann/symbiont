//! Parse a Rust source file and extract function signatures as strings.
//!
//! Uses `syn` to parse the AST and `quote` to convert types back to source text.

use crate::Result;
use quote::ToTokens;
use std::fs;
use syn::{ReturnType, parse_file};
use tracing::debug;

/// Represents a parsed function's metadata and stringified signature.
#[derive(Debug)]
pub(crate) struct FuncSig {
    /// The function name.
    name: String,
    /// Full signature, e.g. `fn step(state: &mut State)`.
    signature: String,
    /// Whether the function is `pub`.
    is_public: bool,
}

/// Parse a Rust source file and return all function signatures.
pub(crate) fn parse_functions() -> Result<Vec<FuncSig>> {
    let file_path = default_lib_path();

    let source = fs::read_to_string(file_path)?;
    let file = parse_file(&source)?;

    Ok(Vec::from_iter(file.items.iter().filter_map(|item| {
        if let syn::Item::Fn(item_fn) = item {
            let is_public = matches!(item_fn.vis, syn::Visibility::Public(_));
            let is_no_mangle = item_fn
                .attrs
                .iter()
                .any(|attr| attr.path().is_ident("no_mangle"));

            if !is_no_mangle {
                debug!("Function is not `#[no_mangle]`, ignoring");
                return None;
            }

            let name = item_fn.sig.ident.to_string();
            format_signature(&item_fn.sig).map(|signature| FuncSig {
                name,
                signature,
                is_public,
            })
        } else {
            None
        }
    })))
}

/// Format a `syn::Signature` into a human-readable string.
fn format_signature(sig: &syn::Signature) -> Option<String> {
    let mut out = String::from("fn ");
    out.push_str(&sig.ident.to_string());

    // Generics are not suppored.
    if !sig.generics.params.is_empty() {
        return None;
    }

    // Inputs: `(a: i32, b: &str)`
    out.push('(');
    let inputs: Vec<_> = sig.inputs.iter().map(|arg| format_fn_arg(arg)).collect();
    out.push_str(&inputs.join(", "));
    out.push(')');

    // Return type: `-> Result<(), Error>`
    if let ReturnType::Type(_, ty) = &sig.output {
        out.push_str(" -> ");
        out.push_str(&format_return_type(ty));
    }

    // Where clause
    if !sig.generics.where_clause.is_none() {
        out.push_str(" where ");
        let preds: Vec<_> = sig
            .generics
            .where_clause
            .as_ref()
            .map(|wc| {
                wc.predicates
                    .iter()
                    .map(|p| normalize_tokens(p.to_token_stream().to_string()))
                    .collect()
            })
            .unwrap_or_default();
        out.push_str(&preds.join(", "));
    }

    Some(out)
}

/// Format a single function argument.
fn format_fn_arg(arg: &syn::FnArg) -> String {
    match arg {
        syn::FnArg::Receiver(recv) => {
            if recv.mutability.is_some() {
                "&mut self".into()
            } else {
                "&self".into()
            }
        }
        syn::FnArg::Typed(pat) => {
            let name = pat.pat.to_token_stream().to_string();
            let ty = normalize_tokens(pat.ty.to_token_stream().to_string());
            format!("{name}: {ty}")
        }
    }
}

/// Clean up spacing artifacts from `quote` output (e.g. `& mut T` → `&mut T`).
fn normalize_tokens(mut s: String) -> String {
    // `quote` produces `& mut T` instead of `&mut T`, `* const T` instead of `*const T`
    s = s.replace("& mut ", "&mut ");
    s = s.replace("& ref ", "&ref ");
    s = s.replace("* mut ", "*mut ");
    s = s.replace("* const ", "*const ");
    s = s.replace("mut & ", "mut &");
    s
}

/// Format the return type, normalizing quote artifacts.
fn format_return_type(ty: &syn::Type) -> String {
    normalize_tokens(ty.to_token_stream().to_string())
}

/// Resolve the path to symbiont-lib's `lib.rs` relative to this crate's manifest dir.
fn default_lib_path() -> String {
    let manifest = env!("CARGO_MANIFEST_DIR");
    format!("{}/../symbiont-lib/src/lib.rs", manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_function() {
        let code = r#"
            pub fn add(a: i32, b: i32) -> i32 {
                a + b
            }
        "#;
        let file = parse_file(code).unwrap();
        let funcs = file
            .items
            .into_iter()
            .filter_map(|item| match item {
                syn::Item::Fn(f) => Some(f),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(funcs.len(), 1);
        let sig = format_signature(&funcs[0].sig).unwrap();
        assert_eq!(sig, "fn add(a: i32, b: i32) -> i32");
    }

    #[test]
    fn test_parse_mut_reference() {
        let code = r#"
            pub fn step(state: &mut State) {
                state.counter += 1;
            }
        "#;
        let file = parse_file(code).unwrap();
        let funcs = file
            .items
            .into_iter()
            .filter_map(|item| match item {
                syn::Item::Fn(f) => Some(f),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(funcs.len(), 1);
        let sig = format_signature(&funcs[0].sig).unwrap();
        assert_eq!(sig, "fn step(state: &mut State)");
    }

    #[test]
    fn test_parse_generics() {
        let code = r#"
            pub fn identity<T>(value: T) -> T {
                value
            }
        "#;
        let file = parse_file(code).unwrap();
        let funcs = file
            .items
            .into_iter()
            .filter_map(|item| match item {
                syn::Item::Fn(f) => Some(f),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(funcs.len(), 1);
        let sig = format_signature(&funcs[0].sig).unwrap();
        assert_eq!(sig, "fn identity<T>(value: T) -> T");
    }

    #[test]
    fn test_default_lib_path() {
        let path = default_lib_path();
        assert!(path.ends_with("../symbiont-lib/src/lib.rs"));
        assert!(std::path::Path::new(&path).exists());
    }
}

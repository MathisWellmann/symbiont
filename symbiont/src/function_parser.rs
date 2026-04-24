//! Parse a Rust source file and extract function signatures as strings.
//!
//! Uses `syn` to parse the AST and `quote` to convert types back to source text.

use crate::Result;
use quote::ToTokens;
use std::fs;
use syn::{ItemFn, ReturnType, parse_file};
use tracing::debug;

pub(crate) type FuncSig = String;

/// Parse a Rust source file and return all function signatures.
pub(crate) fn parse_functions() -> Result<Vec<FuncSig>> {
    let file_path = default_lib_path();

    let source = fs::read_to_string(file_path)?;
    let file = parse_file(&source)?;

    Ok(Vec::from_iter(file.items.iter().filter_map(|item| {
        if let syn::Item::Fn(item_fn) = item {
            let name = item_fn.sig.ident.to_string();
            let is_public = matches!(item_fn.vis, syn::Visibility::Public(_));

            if !is_public {
                debug!("Function is not `pub`, skipping.");
                return None;
            }

            if !is_no_mangle(item_fn) {
                debug!("Function is not `#[no_mangle]`, ignoring: {name}");
                return None;
            }

            format_signature(&item_fn.sig)
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

    assert!(
        sig.generics.where_clause.is_none(),
        "Generics are not supported and this is checked above"
    );

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

/// If `true`, then the function is marked as `#[no_mangle]` or `#[unsafe(no_mangle)]`
fn is_no_mangle(code: &ItemFn) -> bool {
    code.attrs.iter().any(|attr| {
        // `#[no_mangle]` → path is ["no_mangle"]
        // `#[unsafe(no_mangle)]` → path is ["unsafe"], meta is "unsafe (no_mangle)"
        attr.path().is_ident("no_mangle")
            || format!("{}", attr.meta.to_token_stream()).contains("no_mangle")
    })
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
    fn test_generics_returns_none() {
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
        // Generics are not supported by format_signature (returns None)
        assert!(format_signature(&funcs[0].sig).is_none());
    }

    #[test]
    fn test_default_lib_path() {
        let path = default_lib_path();
        assert!(path.ends_with("../symbiont-lib/src/lib.rs"));
        assert!(std::path::Path::new(&path).exists());
    }

    #[test]
    fn test_unsafe_no_mangle_attr_detection() {
        let code: syn::ItemFn = syn::parse_quote! {
            #[unsafe(no_mangle)]
            pub fn step(state: &mut State) {}
        };

        assert!(is_no_mangle(&code));
    }
}

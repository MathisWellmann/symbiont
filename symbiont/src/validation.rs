use quote::ToTokens;
use syn::{
    FnArg,
    Signature,
    Type,
    Visibility,
};
use tracing::info;

use crate::{
    Error,
    Result,
    utils::{
        is_no_mangle,
        is_pub,
    },
};

/// Validate that a parsed AST enforces typed generation:
/// - All functions are `pub`
/// - All functions have `#[unsafe(no_mangle)]`)
/// - All function signatures match the expected signatures from lib.rs.
///
/// Returns `Err` with a descriptive message if any check fails.
pub(crate) fn validate_generated_ast(file: &mut syn::File, expected_sigs: &[String]) -> Result<()> {
    if expected_sigs.is_empty() {
        return Ok(());
    }

    let mut found_sigs: Vec<String> = Vec::new();

    for item in &mut file.items {
        if let syn::Item::Fn(item_fn) = item {
            let name = item_fn.sig.ident.to_string();

            // Add `pub` visibility if missing
            if !is_pub(item_fn) {
                info!("Function `{name}` missing `pub` visibility, adding it");
                item_fn.vis = Visibility::Public(syn::token::Pub::default());
            }
            // Add #[unsafe(no_mangle)] if missing
            if !is_no_mangle(item_fn) {
                info!("Function `{name}` missing #[unsafe(no_mangle)], adding it");
                let attr: syn::Attribute = syn::parse_quote!(#[unsafe(no_mangle)]);
                item_fn.attrs.insert(0, attr);
            }

            // Format the signature and check it matches one of the expected ones
            let sig = format_signature(&item_fn.sig)
                .unwrap_or_else(|| format!("fn {}(...)", item_fn.sig.ident));
            found_sigs.push(sig.clone());

            if !expected_sigs.contains(&sig) {
                let expected = expected_sigs.join(", ");
                return Err(Error::SignatureMismatch {
                    name,
                    expected,
                    got: sig,
                });
            }
        }
    }

    // Ensure all expected signatures were found
    for expected in expected_sigs {
        if !found_sigs.contains(expected) {
            // Try to extract function name for a better error message
            let name = expected
                .trim_start_matches("fn ")
                .split('(')
                .next()
                .unwrap_or("unknown");
            return Err(Error::SignatureMismatch {
                name: name.to_string(),
                expected: expected.clone(),
                got: "not found".to_string(),
            });
        }
    }

    Ok(())
}

/// Format a `syn::Signature` into a human-readable string (same format as function_parser).
fn format_signature(sig: &Signature) -> Option<String> {
    let mut out = String::from("fn ");
    out.push_str(&sig.ident.to_string());

    if !sig.generics.params.is_empty() {
        return None;
    }

    out.push('(');
    let inputs: Vec<_> = sig.inputs.iter().map(format_fn_arg).collect();
    out.push_str(&inputs.join(", "));
    out.push(')');

    if let syn::ReturnType::Type(_, ty) = &sig.output {
        out.push_str(" -> ");
        out.push_str(&format_return_type(ty));
    }

    Some(out)
}

fn format_fn_arg(arg: &FnArg) -> String {
    use FnArg::*;
    match arg {
        Receiver(recv) => {
            if recv.mutability.is_some() {
                "&mut self".into()
            } else {
                "&self".into()
            }
        }
        Typed(pat) => {
            let name = pat.pat.to_token_stream().to_string();
            let ty = normalize_tokens(pat.ty.to_token_stream().to_string());
            format!("{name}: {ty}")
        }
    }
}

fn normalize_tokens(mut s: String) -> String {
    s = s.replace("& mut ", "&mut ");
    s = s.replace("& ref ", "&ref ");
    s = s.replace("* mut ", "*mut ");
    s = s.replace("* const ", "*const ");
    s = s.replace("mut & ", "mut &");
    s
}

fn format_return_type(ty: &Type) -> String {
    normalize_tokens(ty.to_token_stream().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_rust_code;

    #[test]
    fn test_validate_valid_code() {
        let input = r#"```rust
#[unsafe(no_mangle)]
pub fn step(counter: &mut usize) {
    *counter += 1;
}
```"#;
        let mut file = parse_rust_code(input).unwrap();
        let expected = vec!["fn step(counter: &mut usize)".to_string()];
        validate_generated_ast(&mut file, &expected).expect("validation passed");
    }

    #[test]
    fn test_validate_missing_no_mangle_gets_added() {
        let input = r#"```rust
pub fn step(counter: &mut usize) {
    *counter += 1;
}
```"#;
        let mut file = parse_rust_code(input).unwrap();
        let expected = vec!["fn step(counter: &mut usize)".to_string()];
        validate_generated_ast(&mut file, &expected)
            .expect("should succeed by adding #[unsafe(no_mangle)]");

        // Verify the attribute was actually added
        let item_fn = match &file.items[0] {
            syn::Item::Fn(f) => f,
            _ => panic!("expected fn item"),
        };
        assert!(
            is_no_mangle(item_fn),
            "expected #[unsafe(no_mangle)] to be present after validation"
        );
    }

    #[test]
    fn test_validate_non_public_gets_pub_added() {
        let input = r#"```rust
fn step(counter: &mut usize) {
    *counter += 1;
}
```"#;
        let mut file = parse_rust_code(input).unwrap();
        let expected = vec!["fn step(counter: &mut usize)".to_string()];
        validate_generated_ast(&mut file, &expected)
            .expect("should succeed by adding `pub` and #[unsafe(no_mangle)]");

        let item_fn = match &file.items[0] {
            syn::Item::Fn(f) => f,
            _ => panic!("expected fn item"),
        };
        assert!(
            is_pub(item_fn),
            "expected `pub` visibility after validation"
        );
        assert!(
            is_no_mangle(item_fn),
            "expected #[unsafe(no_mangle)] after validation"
        );
    }

    #[test]
    fn test_validate_signature_mismatch() {
        let input = r#"```rust
#[unsafe(no_mangle)]
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
```"#;
        let mut file = parse_rust_code(input).unwrap();
        let expected = vec!["fn step(counter: &mut usize)".to_string()];
        let err = validate_generated_ast(&mut file, &expected).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("step") && msg.contains("add"),
            "expected signature mismatch error, got: {msg}"
        );
    }

    #[test]
    fn test_validate_unsafe_no_mangle() {
        let input = r#"```rust
#[unsafe(no_mangle)]
pub fn step(counter: &mut usize) {
    *counter += 1;
}
```"#;
        let mut file = parse_rust_code(input).unwrap();
        let expected = vec!["fn step(counter: &mut usize)".to_string()];
        validate_generated_ast(&mut file, &expected).expect("#[unsafe(no_mangle)] should be valid");
    }

    #[test]
    fn test_validate_with_return_type() {
        let input = r#"```rust
#[unsafe(no_mangle)]
pub fn step(counter: &mut usize) -> usize {
    *counter
}
```"#;
        let mut file = parse_rust_code(input).unwrap();
        let expected = vec!["fn step(counter: &mut usize) -> usize".to_string()];
        validate_generated_ast(&mut file, &expected).expect("validation with return type passed");
    }
}

use quote::ToTokens;

use crate::{
    Error, Result,
    utils::{is_no_mangle, is_pub},
};

/// Validate that a parsed AST enforces typed generation:
/// - All functions are `pub`
/// - All functions have `#[unsafe(no_mangle)]`)
/// - All function signatures match the expected signatures from lib.rs.
///
/// Returns `Err` with a descriptive message if any check fails.
pub(crate) fn validate_generated_ast(file: &syn::File, expected_sigs: &[String]) -> Result<()> {
    if expected_sigs.is_empty() {
        return Ok(());
    }

    let mut found_sigs: Vec<String> = Vec::new();

    for item in &file.items {
        match item {
            syn::Item::Fn(item_fn) => {
                let name = item_fn.sig.ident.to_string();

                // Check pub visibility
                if !is_pub(item_fn) {
                    return Err(Error::NonPublicFunction(name));
                }
                // Check #[no_mangle] attribute
                if !is_no_mangle(item_fn) {
                    return Err(Error::MissingNoMangle(name));
                }

                // Format the signature and check it matches one of the expected ones
                let sig = format_signature(&item_fn.sig)
                    .unwrap_or_else(|| format!("fn {}(...)", item_fn.sig.ident));
                found_sigs.push(sig.clone());

                if !expected_sigs.contains(&sig) {
                    let expected_str = expected_sigs.join(", ");
                    return Err(Error::SignatureMismatch(name, expected_str, sig));
                }
            }
            _ => {}
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
            return Err(Error::SignatureMismatch(
                name.to_string(),
                expected.clone(),
                "not found".to_string(),
            ));
        }
    }

    Ok(())
}

/// Format a `syn::Signature` into a human-readable string (same format as function_parser).
fn format_signature(sig: &syn::Signature) -> Option<String> {
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

fn normalize_tokens(mut s: String) -> String {
    s = s.replace("& mut ", "&mut ");
    s = s.replace("& ref ", "&ref ");
    s = s.replace("* mut ", "*mut ");
    s = s.replace("* const ", "*const ");
    s = s.replace("mut & ", "mut &");
    s
}

fn format_return_type(ty: &syn::Type) -> String {
    normalize_tokens(ty.to_token_stream().to_string())
}

#[cfg(test)]
mod tests {
    use crate::parser::parse_rust_code;

    use super::*;

    #[test]
    fn test_validate_valid_code() {
        let input = r#"```rust
#[unsafe(no_mangle)]
pub fn step(counter: &mut usize) {
    *counter += 1;
}
```"#;
        let file = parse_rust_code(input).unwrap();
        let expected = vec!["fn step(counter: &mut usize)".to_string()];
        validate_generated_ast(&file, &expected).expect("validation passed");
    }

    #[test]
    fn test_validate_missing_no_mangle() {
        let input = r#"```rust
pub fn step(counter: &mut usize) {
    *counter += 1;
}
```"#;
        let file = parse_rust_code(input).unwrap();
        let expected = vec!["fn step(counter: &mut usize)".to_string()];
        let err = validate_generated_ast(&file, &expected).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("no_mangle"),
            "expected no_mangle error, got: {msg}"
        );
    }

    #[test]
    fn test_validate_non_public_function() {
        let input = r#"```rust
#[no_mangle]
fn step(counter: &mut usize) {
    *counter += 1;
}
```"#;
        let file = parse_rust_code(input).unwrap();
        let expected = vec!["fn step(counter: &mut usize)".to_string()];
        let err = validate_generated_ast(&file, &expected).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("not `pub`") || msg.contains("NonPublic"),
            "expected pub error, got: {msg}"
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
        let file = parse_rust_code(input).unwrap();
        let expected = vec!["fn step(counter: &mut usize)".to_string()];
        let err = validate_generated_ast(&file, &expected).unwrap_err();
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
        let file = parse_rust_code(input).unwrap();
        let expected = vec!["fn step(counter: &mut usize)".to_string()];
        validate_generated_ast(&file, &expected).expect("#[unsafe(no_mangle)] should be valid");
    }

    #[test]
    fn test_validate_with_return_type() {
        let input = r#"```rust
#[unsafe(no_mangle)]
pub fn step(counter: &mut usize) -> usize {
    *counter
}
```"#;
        let file = parse_rust_code(input).unwrap();
        let expected = vec!["fn step(counter: &mut usize) -> usize".to_string()];
        validate_generated_ast(&file, &expected).expect("validation with return type passed");
    }
}

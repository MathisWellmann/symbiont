use prettyplease::unparse;
// SPDX-License-Identifier: MPL-2.0
use quote::ToTokens;
use syn::{
    FnArg,
    Signature,
    Type,
    Visibility,
    visit::{
        self,
        Visit,
    },
};
use tracing::debug;

use crate::{
    Error,
    Result,
    utils::{
        is_no_mangle,
        is_pub,
    },
};

/// Validate that a parsed AST enforces typed generation:
/// - No `unsafe` code anywhere (see [`reject_unsafe_code`])
/// - All functions are `pub`
/// - All functions have `#[unsafe(no_mangle)]`)
/// - All function signatures match the expected signatures from lib.rs.
///
/// Signatures are compared by function name, argument **types**, and return
/// type. Argument names are ignored: they carry no meaning at the dylib ABI
/// boundary (symbol lookup is by function name only), so renaming an unused
/// parameter (e.g. to `_market_state`) is not a mismatch.
///
/// Returns `Err` with a descriptive message if any check fails.
pub(crate) fn validate_generated_ast(file: &mut syn::File, expected_sigs: &[String]) -> Result<()> {
    reject_unsafe_code(file)?;

    if expected_sigs.is_empty() {
        return Ok(());
    }

    let mut found_sigs: Vec<(String, String)> = Vec::with_capacity(4);

    for item in &mut file.items {
        if let syn::Item::Fn(item_fn) = item {
            let name = item_fn.sig.ident.to_string();

            // Add `pub` visibility if missing
            if !is_pub(item_fn) {
                debug!("Function `{name}` missing `pub` visibility, adding it");
                item_fn.vis = Visibility::Public(syn::token::Pub::default());
            }
            // Add #[unsafe(no_mangle)] if missing
            if !is_no_mangle(item_fn) {
                debug!("Function `{name}` missing #[unsafe(no_mangle)], adding it");
                let attr: syn::Attribute = syn::parse_quote!(#[unsafe(no_mangle)]);
                item_fn.attrs.insert(0, attr);
            }

            let sig = format_signature(&item_fn.sig)
                .unwrap_or_else(|| normalize_tokens(item_fn.sig.to_token_stream().to_string()));
            found_sigs.push((name, sig));
        }
    }

    // Compare each expected signature against the generated function of the
    // same name, so the error pinpoints the offending function instead of the
    // first expected signature that had no exact match. Expected signatures
    // retain argument names for prompts; only this compatibility comparison
    // canonicalizes them to argument types.
    for expected in expected_sigs {
        let Some(fn_name) = expected_fn_name(expected) else {
            continue;
        };
        let expected_canonical = syn::parse_str::<Signature>(expected)
            .ok()
            .and_then(|sig| format_signature(&sig))
            .unwrap_or_else(|| expected.clone());
        let found = found_sigs.iter().find(|(name, _)| name == fn_name);
        match found {
            Some((_, sig)) if sig == &expected_canonical => {}
            Some((_, sig)) => {
                return Err(Error::SignatureMismatch {
                    code: unparse(file),
                    expected: expected.clone(),
                    got: sig.clone(),
                });
            }
            None => {
                return Err(Error::SignatureMismatch {
                    code: unparse(file),
                    expected: expected.clone(),
                    got: format!("function `{fn_name}` not found"),
                });
            }
        }
    }

    Ok(())
}

/// Reject any `unsafe` construct in LLM-generated code.
///
/// Enforced on the parsed AST *before* compilation: the rejection is
/// cheap (no cargo round-trip), pinpoints the offending construct for the
/// backpressure prompt, and cannot be evaded with `#[allow(unsafe_code)]`
/// the way a compiler lint could. A crate-level `#![forbid(unsafe_code)]`
/// is not an option anyway: the injected panic preamble is legitimately
/// unsafe, and the `#[unsafe(no_mangle)]` export attribute on every
/// evolvable function trips the `unsafe_code` lint in edition 2024.
///
/// Rejected constructs:
/// - `unsafe { .. }` blocks
/// - `unsafe fn` (free, impl, trait, and foreign)
/// - `unsafe impl` and `unsafe trait`
/// - `extern` blocks
/// - unsafe attributes such as `#[unsafe(export_name = ..)]` — except the
///   exact `#[unsafe(no_mangle)]` export attribute the harness itself
///   manages
/// - an `unsafe` token anywhere inside a macro definition or invocation,
///   which would otherwise smuggle unsafe code past the AST scan
pub(crate) fn reject_unsafe_code(file: &syn::File) -> Result<()> {
    let mut scan = UnsafeScan { finding: None };
    scan.visit_file(file);
    match scan.finding {
        Some(construct) => Err(Error::UnsafeCode {
            code: unparse(file),
            construct,
        }),
        None => Ok(()),
    }
}

/// AST visitor recording the first forbidden `unsafe` construct.
struct UnsafeScan {
    finding: Option<String>,
}

impl UnsafeScan {
    fn record(&mut self, what: &str, tokens: &dyn ToTokens) {
        if self.finding.is_none() {
            let mut snippet = tokens.to_token_stream().to_string();
            if snippet.len() > 120 {
                let cut = (0..=120)
                    .rev()
                    .find(|i| snippet.is_char_boundary(*i))
                    .unwrap_or(0);
                snippet.truncate(cut);
                snippet.push('…');
            }
            self.finding = Some(format!("{what}: `{snippet}`"));
        }
    }
}

/// `true` if any token (recursively) is the `unsafe` keyword.
fn tokens_contain_unsafe(tokens: proc_macro2::TokenStream) -> bool {
    tokens.into_iter().any(|tt| match tt {
        proc_macro2::TokenTree::Ident(ident) => ident == "unsafe",
        proc_macro2::TokenTree::Group(group) => tokens_contain_unsafe(group.stream()),
        _ => false,
    })
}

impl<'ast> Visit<'ast> for UnsafeScan {
    fn visit_expr_unsafe(&mut self, node: &'ast syn::ExprUnsafe) {
        self.record("an `unsafe` block", node);
    }

    // Covers free functions, impl methods, trait methods, and foreign fns.
    fn visit_signature(&mut self, node: &'ast syn::Signature) {
        if node.unsafety.is_some() {
            self.record("an `unsafe fn`", node);
        }
        visit::visit_signature(self, node);
    }

    fn visit_item_impl(&mut self, node: &'ast syn::ItemImpl) {
        if node.unsafety.is_some() {
            self.record("an `unsafe impl`", node);
        }
        visit::visit_item_impl(self, node);
    }

    fn visit_item_trait(&mut self, node: &'ast syn::ItemTrait) {
        if node.unsafety.is_some() {
            self.record("an `unsafe trait`", node);
        }
        visit::visit_item_trait(self, node);
    }

    fn visit_item_foreign_mod(&mut self, node: &'ast syn::ItemForeignMod) {
        self.record("an `extern` block", node);
    }

    fn visit_attribute(&mut self, node: &'ast syn::Attribute) {
        // The exact export attribute the harness manages is the only
        // permitted unsafe attribute; see `crate::utils::is_no_mangle`.
        let is_no_mangle_export = node.path().is_ident("unsafe")
            && matches!(&node.meta, syn::Meta::List(list) if list.tokens.to_string() == "no_mangle");
        if node.path().is_ident("unsafe") && !is_no_mangle_export {
            self.record("an unsafe attribute", node);
        }
        visit::visit_attribute(self, node);
    }

    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        if tokens_contain_unsafe(node.tokens.clone()) {
            self.record("an `unsafe` token inside a macro", node);
        }
        visit::visit_macro(self, node);
    }
}

/// Extract the function name from a signature rendered by [`format_signature`],
/// e.g. `fn step(&mut usize)` -> `step`.
fn expected_fn_name(sig: &str) -> Option<&str> {
    sig.strip_prefix("fn ")?.split('(').next()
}

/// Format a `syn::Signature` into a canonical string for comparison.
///
/// Renders `fn name(ty0, ty1, ...) -> ret` **without argument names**: two
/// signatures that differ only in argument names are the same function at the
/// dylib boundary and must compare equal.
fn format_signature(sig: &Signature) -> Option<String> {
    let mut out = String::from("fn ");
    out.push_str(&sig.ident.to_string());

    if sig.asyncness.is_some()
        || sig.unsafety.is_some()
        || sig.abi.is_some()
        || sig.variadic.is_some()
        || !sig.generics.params.is_empty()
    {
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
        Typed(pat) => normalize_tokens(pat.ty.to_token_stream().to_string()),
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
        let input = "```rust
#[unsafe(no_mangle)]
pub fn step(counter: &mut usize) {
    *counter += 1;
}
```";
        let mut file = parse_rust_code(input).expect("can parse");
        let expected = vec!["fn step(counter: &mut usize)".to_string()];
        validate_generated_ast(&mut file, &expected).expect("validation passed");
    }

    #[test]
    fn test_validate_missing_no_mangle_gets_added() {
        let input = "```rust
pub fn step(counter: &mut usize) {
    *counter += 1;
}
```";
        let mut file = parse_rust_code(input).expect("can parse");
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
        let input = "```rust
fn step(counter: &mut usize) {
    *counter += 1;
}
```";
        let mut file = parse_rust_code(input).expect("can parse");
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
        let input = "```rust
#[unsafe(no_mangle)]
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
```";
        let mut file = parse_rust_code(input).expect("can parse");
        let expected = vec!["fn step(counter: &mut usize)".to_string()];
        let err = validate_generated_ast(&mut file, &expected).expect_err("should error");
        dbg!(&err);
        match err {
            Error::SignatureMismatch {
                code,
                expected,
                got,
            } => {
                assert_eq!(
                    &code,
                    "#[unsafe(no_mangle)]\npub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n"
                );
                assert_eq!(expected, "fn step(counter: &mut usize)");
                assert_eq!(got, "function `step` not found");
            }
            _ => panic!("Invalid error"),
        }
    }

    #[test]
    fn argument_names_are_ignored() {
        // Renaming an argument (e.g. marking it unused) is ABI-compatible:
        // dylib dispatch is by function name only.
        let input = "```rust
#[unsafe(no_mangle)]
pub fn step(_counter: &mut usize) {
    *_counter += 1;
}
```";
        let mut file = parse_rust_code(input).expect("can parse");
        let expected = vec!["fn step(counter: &mut usize)".to_string()];
        validate_generated_ast(&mut file, &expected).expect("renamed argument must validate");
    }

    #[test]
    fn abi_incompatible_modifiers_are_reported() {
        // `unsafe` modifiers are intercepted by the unsafe scan first.
        for input in [
            "pub unsafe fn step(counter: &mut usize) {}",
            "pub unsafe extern \"C\" fn step(counter: &mut usize, ...) {}",
        ] {
            let mut file = syn::parse_str(input).expect("can parse");
            let expected = vec!["fn step(counter: &mut usize)".to_string()];
            let err = validate_generated_ast(&mut file, &expected)
                .expect_err("unsafe signature must be rejected");
            assert!(
                matches!(
                    err,
                    Error::UnsafeCode { ref construct, .. } if construct.contains("an `unsafe fn`")
                ),
                "feedback must identify the unsafe fn in {input}: {err}"
            );
        }

        // Safe but ABI-incompatible modifiers surface as signature mismatches.
        for (input, modifier) in [
            ("pub async fn step(counter: &mut usize) {}", "async"),
            (
                "pub extern \"C\" fn step(counter: &mut usize) {}",
                "extern \"C\"",
            ),
        ] {
            let mut file = syn::parse_str(input).expect("can parse");
            let expected = vec!["fn step(counter: &mut usize)".to_string()];
            let err = validate_generated_ast(&mut file, &expected)
                .expect_err("incompatible signature must be rejected");
            assert!(
                matches!(
                    err,
                    Error::SignatureMismatch { ref got, .. } if got.contains(modifier)
                ),
                "feedback must identify `{modifier}` in {input}: {err}"
            );
        }
    }

    #[test]
    fn mismatch_names_the_offending_function() {
        // `action` matches but `on_order_update` does not; the error must
        // point at `on_order_update`, not at the first expected signature.
        let input = "```rust
#[unsafe(no_mangle)]
pub fn action(tick: &Tick) {
    let _ = tick;
}
#[unsafe(no_mangle)]
pub fn on_order_update(update: &Update, extra: bool) {
    let _ = (update, extra);
}
```";
        let mut file = parse_rust_code(input).expect("can parse");
        let expected = vec![
            "fn action(tick: & Tick)".to_string(),
            "fn on_order_update(update: & Update)".to_string(),
        ];
        let err = validate_generated_ast(&mut file, &expected).expect_err("should error");
        match err {
            Error::SignatureMismatch { expected, got, .. } => {
                assert_eq!(expected, "fn on_order_update(update: & Update)");
                assert_eq!(got, "fn on_order_update(& Update, bool)");
            }
            _ => panic!("Invalid error"),
        }
    }

    #[test]
    fn test_validate_unsafe_no_mangle() {
        let input = "```rust
#[unsafe(no_mangle)]
pub fn step(counter: &mut usize) {
    *counter += 1;
}
```";
        let mut file = parse_rust_code(input).expect("can parse");
        let expected = vec!["fn step(counter: &mut usize)".to_string()];
        validate_generated_ast(&mut file, &expected).expect("#[unsafe(no_mangle)] should be valid");
    }

    #[test]
    fn multi_fn_with_renamed_unused_argument_validates() {
        // Regression test for a stuck self-healing loop: the generated
        // `on_order_update` renamed the unused `market_state` parameter to
        // `_market_state`, which must not be a signature mismatch.
        let input = "```rust
#[unsafe(no_mangle)]
pub fn action(
    step_data: &TickData,
    account: &Account<i64, DECIMALS, Cur, UserOrderId>,
    market_state: &MarketState<i64, DECIMALS>,
    account_tracker: &FullAccountTracker<DECIMALS, Cur>,
    commands: &mut CommandBuffer<DECIMALS, Cur>,
) {
}
#[unsafe(no_mangle)]
pub fn on_order_update(
    order_update: OrderUpdate<DECIMALS, Cur>,
    account: &Account<i64, DECIMALS, Cur, UserOrderId>,
    _market_state: &MarketState<i64, DECIMALS>,
    commands: &mut CommandBuffer<DECIMALS, Cur>,
) {
}
```";
        let mut file = parse_rust_code(input).expect("can parse");
        let expected = vec![
            "fn action(step_data: & TickData, account: & Account < i64, DECIMALS, Cur, UserOrderId >, market_state: & MarketState < i64, DECIMALS >, account_tracker: & FullAccountTracker < DECIMALS, Cur >, commands: &mut CommandBuffer < DECIMALS, Cur >)".to_string(),
            "fn on_order_update(order_update: OrderUpdate < DECIMALS, Cur >, account: & Account < i64, DECIMALS, Cur, UserOrderId >, market_state: & MarketState < i64, DECIMALS >, commands: &mut CommandBuffer < DECIMALS, Cur >)".to_string(),
        ];
        validate_generated_ast(&mut file, &expected)
            .expect("renamed `_market_state` argument must validate");
    }

    #[test]
    #[tracing_test::traced_test]
    fn test_validate_with_return_type() {
        let input = "```rust
#[unsafe(no_mangle)]
pub fn step(counter: &mut usize) -> usize {
    *counter
}
```";
        let mut file = parse_rust_code(input).expect("can parse");
        let expected = vec!["fn step(counter: &mut usize) -> usize".to_string()];
        validate_generated_ast(&mut file, &expected).expect("validation with return type passed");
    }

    /// Assert `reject_unsafe_code` rejects `code` and names `construct`.
    fn assert_rejects_unsafe(code: &str, construct: &str) {
        let file: syn::File = syn::parse_str(code).expect("can parse");
        let err = reject_unsafe_code(&file).expect_err("unsafe construct must be rejected");
        match err {
            Error::UnsafeCode {
                construct: found, ..
            } => {
                assert!(
                    found.contains(construct),
                    "expected construct `{construct}`, got `{found}`"
                );
            }
            other => panic!("expected Error::UnsafeCode, got {other}"),
        }
    }

    #[test]
    fn unsafe_block_is_rejected() {
        assert_rejects_unsafe(
            "pub fn step(counter: &mut usize) { unsafe { *counter += 1; } }",
            "an `unsafe` block",
        );
    }

    #[test]
    fn unsafe_fn_is_rejected() {
        assert_rejects_unsafe("pub unsafe fn helper(ptr: *mut usize) {}", "an `unsafe fn`");
        assert_rejects_unsafe(
            "struct S; impl S { unsafe fn helper(&self) {} }",
            "an `unsafe fn`",
        );
    }

    #[test]
    fn unsafe_impl_and_trait_are_rejected() {
        assert_rejects_unsafe("struct S; unsafe impl Send for S {}", "an `unsafe impl`");
        assert_rejects_unsafe("unsafe trait Scary {}", "an `unsafe trait`");
    }

    #[test]
    fn extern_block_is_rejected() {
        assert_rejects_unsafe(
            "unsafe extern \"C\" { fn malloc(size: usize) -> *mut u8; }",
            "an `extern` block",
        );
    }

    #[test]
    fn unsafe_attribute_is_rejected_but_no_mangle_is_allowed() {
        assert_rejects_unsafe(
            "#[unsafe(export_name = \"evil\")] pub fn step(counter: &mut usize) {}",
            "an unsafe attribute",
        );

        let file: syn::File = syn::parse_str(
            "#[unsafe(no_mangle)] pub fn step(counter: &mut usize) { *counter += 1; }",
        )
        .expect("can parse");
        reject_unsafe_code(&file).expect("#[unsafe(no_mangle)] is harness-managed and allowed");
    }

    #[test]
    fn unsafe_smuggled_through_macro_is_rejected() {
        assert_rejects_unsafe(
            "macro_rules! sneaky { () => { unsafe { core::hint::unreachable_unchecked() } }; }\n\
             pub fn step(counter: &mut usize) { sneaky!(); }",
            "an `unsafe` token inside a macro",
        );
        assert_rejects_unsafe(
            "pub fn step(counter: &mut usize) { let _ = stringify!(unsafe); }",
            "an `unsafe` token inside a macro",
        );
    }

    #[test]
    fn safe_code_passes_unsafe_scan() {
        let file: syn::File = syn::parse_str(
            "pub fn step(counter: &mut usize) {\n\
                 let values = vec![1usize, 2, 3];\n\
                 *counter += values.iter().sum::<usize>();\n\
                 println!(\"counter is {counter}\");\n\
             }",
        )
        .expect("can parse");
        reject_unsafe_code(&file).expect("safe code must pass");
    }
}

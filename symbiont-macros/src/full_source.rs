use quote::quote;

use crate::evolvable::EvolvableFn;

/// Build the `full_source` string for the dylib from a function declaration.
///
/// Forces `pub` visibility and prepends `#[unsafe(no_mangle)]`.
pub(crate) fn build_full_source(func: &EvolvableFn) -> String {
    let sig = func.sig();

    // Keep the body as a TokenStream so `quote!` splices it as code, not as a string literal.
    let body_tokens: proc_macro2::TokenStream = match func {
        EvolvableFn::WithBody(f) => {
            let block = &f.block;
            quote!(#block)
        }
        EvolvableFn::WithoutBody(_) => quote!({ todo!() }),
    };

    let inputs = &sig.inputs;
    let output = &sig.output;
    let ident = &sig.ident;

    // Preserve doc comments on the generated function so they are available in the
    // dylib's source for tooling and documentation purposes. Render them as `///`
    // line comments rather than `#[doc = "..."]` attributes for readability.
    use std::fmt::Write as _;
    let doc_lines: String = func
        .attrs()
        .iter()
        .filter(|attr| attr.path().is_ident("doc"))
        .filter_map(extract_doc_string)
        .fold(String::new(), |mut acc, line| {
            let _ = writeln!(acc, "///{line}");
            acc
        });

    let fn_tokens = quote! {
        #[unsafe(no_mangle)]
        pub fn #ident(#inputs) #output #body_tokens
    };

    // `quote!` above always produces a valid `fn` item, so parsing as a
    // `syn::File` is infallible. Run it through `prettyplease` so the emitted
    // source reads like hand-written Rust rather than a single token-spaced line.
    let file = syn::parse2::<syn::File>(fn_tokens).expect("generated fn is valid Rust");
    let formatted = prettyplease::unparse(&file);

    format!("{doc_lines}{formatted}")
}

/// Extract the string value from a `#[doc = "..."]` attribute.
fn extract_doc_string(attr: &syn::Attribute) -> Option<String> {
    if let syn::Meta::NameValue(nv) = &attr.meta
        && let syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Str(s),
            ..
        }) = &nv.value
    {
        return Some(s.value());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EvolvableBlock;

    #[test]
    fn test_build_full_source() {
        let func = quote!(
            /// Should increment the counter by a value in the range 5..20
            fn step(counter: &mut usize) {
                *counter += 1;
                println!("doing stuff in iteration {}", counter);
            }
        );
        let block: EvolvableBlock = syn::parse2(func).expect("parse evolvable block");
        assert_eq!(
            build_full_source(&block.functions[0]),
            "/// Should increment the counter by a value in the range 5..20\n\
             #[unsafe(no_mangle)]\n\
             pub fn step(counter: &mut usize) {\n    \
                 *counter += 1;\n    \
                 println!(\"doing stuff in iteration {}\", counter);\n\
             }\n",
        );
    }
}

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

    let fn_body = quote! {
        #[unsafe(no_mangle)]
        pub fn #ident(#inputs) #output #body_tokens
    };

    format!("{doc_lines}{fn_body}\n").trim_start().to_string()
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
    #[test]
    fn test_build_full_source() {
        todo!()
    }
}

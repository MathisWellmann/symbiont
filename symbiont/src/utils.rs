use quote::ToTokens;
use syn::ItemFn;

/// If `true`, the function has visibiliity `pub`
#[inline(always)]
pub(crate) fn is_pub(item_fn: &ItemFn) -> bool {
    matches!(item_fn.vis, syn::Visibility::Public(_))
}

/// If `true`, the function is annotated with `#[no_mangle]` or `#[unsafe(no_mangle)]`
#[inline]
pub(crate) fn is_no_mangle(item_fn: &syn::ItemFn) -> bool {
    item_fn.attrs.iter().any(|attr| {
        attr.path().is_ident("no_mangle")
            || format!("{}", attr.meta.to_token_stream()).contains("no_mangle")
    })
}

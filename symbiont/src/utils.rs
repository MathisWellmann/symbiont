// SPDX-License-Identifier: MPL-2.0
use syn::ItemFn;

/// If `true`, the function has visibiliity `pub`
#[inline(always)]
pub(crate) fn is_pub(item_fn: &ItemFn) -> bool {
    matches!(item_fn.vis, syn::Visibility::Public(_))
}

/// If `true`, the function is annotated with `#[unsafe(no_mangle)]`
#[inline]
pub(crate) fn is_no_mangle(item_fn: &ItemFn) -> bool {
    item_fn.attrs.iter().any(|attr| {
        // Only match exactly #[unsafe(no_mangle)]
        attr.path().is_ident("unsafe")
            && matches!(&attr.meta, syn::Meta::List(list) if list.tokens.to_string() == "no_mangle")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unsafe_no_mangle_detected() {
        let code: ItemFn = syn::parse_quote! {
            #[unsafe(no_mangle)]
            pub fn step(counter: &mut usize) {}
        };
        assert!(is_no_mangle(&code));
    }

    #[test]
    fn test_plain_no_mangle_rejected() {
        let code: ItemFn = syn::parse_quote! {
            #[no_mangle]
            pub fn step(counter: &mut usize) {}
        };
        assert!(!is_no_mangle(&code));
    }

    #[test]
    fn test_no_attribute_returns_false() {
        let code: ItemFn = syn::parse_quote! {
            pub fn step(counter: &mut usize) {}
        };
        assert!(!is_no_mangle(&code));
    }

    #[test]
    fn test_other_attribute_returns_false() {
        let code: ItemFn = syn::parse_quote! {
            #[allow(dead_code)]
            pub fn step(counter: &mut usize) {}
        };
        assert!(!is_no_mangle(&code));
    }
}

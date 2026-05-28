// SPDX-License-Identifier: MPL-2.0
//! Proc macros for the `symbiont` crate.
//!
//! Provides the [`evolvable!`] function-like macro that declares
//! hot-reloadable functions and generates dispatch wrappers.

mod evolvable;
mod full_source;
mod utils;

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::{
    FnArg,
    ReturnType,
};

use crate::{
    evolvable::EvolvableBlock,
    full_source::build_full_source,
    utils::format_signature,
};

/// Declare hot-reloadable functions that are compiled into a temporary dylib and loaded at runtime.
///
/// # Examples
///
/// ```rust,ignore
/// symbiont::evolvable! {
///     fn step(counter: &mut usize) {
///         *counter += 1;  // default implementation
///     }
///
///     fn compute(x: f64) -> f64;  // bodyless, defaults to todo!()
/// }
/// ```
///
/// This generates:
/// - A `SYMBIONT_DECLS` constant with metadata for each function
/// - Wrapper functions that dispatch calls through the loaded dylib
#[proc_macro]
pub fn evolvable(input: TokenStream) -> TokenStream {
    let block = syn::parse_macro_input!(input as EvolvableBlock);

    if block.functions.is_empty() {
        return syn::Error::new(
            Span::call_site(),
            "evolvable! block must contain at least one function",
        )
        .to_compile_error()
        .into();
    }

    let mut decl_entries = Vec::new();
    let mut wrapper_fns = Vec::new();

    for func in &block.functions {
        let sig = func.sig();
        let vis = func.vis();
        let ident = &sig.ident;
        let fn_name_str = ident.to_string();
        let signature_str = format_signature(sig);
        let full_source = build_full_source(func);

        // Generate a unique static name for the cached function pointer
        let static_name = syn::Ident::new(
            &format!("__SYMBIONT_FN_{}", fn_name_str.to_uppercase()),
            ident.span(),
        );

        // Build the argument types and names for the extern fn type and call
        let mut arg_types = Vec::new();
        let mut arg_names = Vec::new();
        for arg in &sig.inputs {
            match arg {
                FnArg::Typed(pat_type) => {
                    arg_types.push(pat_type.ty.as_ref().clone());
                    arg_names.push(pat_type.pat.as_ref().clone());
                }
                FnArg::Receiver(_) => {
                    return syn::Error::new_spanned(
                        arg,
                        "self receivers are not supported in evolvable functions",
                    )
                    .to_compile_error()
                    .into();
                }
            }
        }

        let ret_ty = match &sig.output {
            ReturnType::Default => quote! { () },
            ReturnType::Type(_, ty) => quote! { #ty },
        };

        // Build the EvolvableDecl entry (with reference to the AtomicPtr static)
        decl_entries.push(quote! {
            ::symbiont::EvolvableDecl {
                name: #fn_name_str,
                signature: #signature_str,
                full_source: #full_source,
                fn_ptr: &#static_name,
            }
        });

        // Build the per-function AtomicPtr static and lock-free dispatch wrapper
        let fn_inputs = &sig.inputs;
        let fn_output = &sig.output;

        wrapper_fns.push(quote! {
            #[doc(hidden)]
            static #static_name: ::std::sync::atomic::AtomicPtr<()> =
                ::std::sync::atomic::AtomicPtr::new(::std::ptr::null_mut());

            #vis fn #ident(#fn_inputs) #fn_output {
                // In debug builds, track this call so evolve() can assert
                // no functions are in flight. Compiled away in release.
                #[cfg(debug_assertions)]
                let _call_guard = ::symbiont::__internal::enter_call();

                let ptr = #static_name.load(::std::sync::atomic::Ordering::Acquire);
                debug_assert!(
                    !ptr.is_null(),
                    concat!("symbiont: function '", #fn_name_str, "' not initialized; call Runtime::init() first")
                );
                let f: fn(#(#arg_types),*) -> #ret_ty = unsafe { ::std::mem::transmute(ptr) };
                f(#(#arg_names),*)
            }
        });
    }

    // Render inline items declared inside `evolvable! { ... }` into the
    // per-call constant slice. These items are re-emitted verbatim in the
    // host crate and converted to a string literal that the runtime will
    // splice into the dylib's `lib.rs`.
    let prelude_items = &block.prelude_items;
    let inline_prelude_source = if prelude_items.is_empty() {
        String::new()
    } else {
        let file = syn::File {
            shebang: None,
            attrs: Vec::new(),
            items: prelude_items.clone(),
        };
        prettyplease::unparse(&file)
    };

    let mut prelude_parts: Vec<proc_macro2::TokenStream> = Vec::new();
    if !inline_prelude_source.is_empty() {
        prelude_parts.push(quote! { #inline_prelude_source });
    }

    let expanded = quote! {
        #(#prelude_items)*

        #[doc(hidden)]
        const SYMBIONT_PRELUDE: &[&::core::primitive::str] = &[
            #(#prelude_parts),*
        ];

        const SYMBIONT_DECLS: &[::symbiont::EvolvableDecl] = &[
            #(#decl_entries),*
        ];

        #(#wrapper_fns)*
    };

    expanded.into()
}

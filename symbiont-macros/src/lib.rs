// SPDX-License-Identifier: MPL-2.0
//! Proc macros for the `symbiont` crate.
//!
//! Provides the [`evolvable!`] function-like macro that declares
//! hot-reloadable functions and generates dispatch wrappers.

mod evolvable;
mod full_source;

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{
    ToTokens,
    quote,
};
use syn::{
    FnArg,
    ReturnType,
    Signature,
};

use crate::{
    evolvable::EvolvableBlock,
    full_source::build_full_source,
};

/// Format a `syn::Signature` into a human-readable string like `"fn step(counter: &mut usize)"`.
///
/// This mirrors the format used by `symbiont`'s validation module.
fn format_signature(sig: &Signature) -> String {
    let mut out = String::from("fn ");
    out.push_str(&sig.ident.to_string());

    out.push('(');
    let inputs: Vec<String> = sig.inputs.iter().map(format_fn_arg).collect();
    out.push_str(&inputs.join(", "));
    out.push(')');

    if let ReturnType::Type(_, ty) = &sig.output {
        out.push_str(" -> ");
        out.push_str(&normalize_tokens(ty.to_token_stream().to_string()));
    }

    out
}

fn format_fn_arg(arg: &FnArg) -> String {
    match arg {
        FnArg::Receiver(recv) => {
            if recv.mutability.is_some() {
                "&mut self".into()
            } else {
                "&self".into()
            }
        }
        FnArg::Typed(pat) => {
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

    let expanded = quote! {
        const SYMBIONT_DECLS: &[::symbiont::EvolvableDecl] = &[
            #(#decl_entries),*
        ];

        #(#wrapper_fns)*
    };

    expanded.into()
}
